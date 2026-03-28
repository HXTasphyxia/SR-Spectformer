#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光谱重建项目预测脚本
用于使用训练好的模型进行光谱重建预测
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
from models import SuperResolvedSpectformer
from utils.utils import load_checkpoint
from data.processing_pipeline import apply_optical_encoding
from optics.psf_convolution import render_psf
from optics.generate_psf import Lens


def load_model(checkpoint_path, device, upscale_factor=1, num_classes=31):
    """加载训练好的模型"""
    # 初始化模型
    model = SuperResolvedSpectformer(
        upscale_factor=upscale_factor,
        num_classes=num_classes,
        token_label=False,
        image_size=512  # 根据配置文件设置
    )
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"成功加载模型检查点: {checkpoint_path}")
        print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
        print(f"最佳损失: {checkpoint.get('best_loss', '未知'):.6f}")
        print(f"不匹配的键: {incompatible_keys}")
    else:
        raise FileNotFoundError(f"模型检查点文件不存在: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model


def load_psf(lens_file):
    """加载PSF数据"""
    try:
        lens = Lens(lens_file)
        # 生成PSF，使用默认参数
        point = torch.tensor([0.0, 0.0, -1000.0])
        psf = lens.psf(point)
        return psf
    except Exception as e:
        print(f"加载PSF失败: {e}")
        return None


def preprocess_image(image_path, target_size=(512, 512), apply_optical_encoding_flag=False, lens_file=None):
    """预处理输入图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image = image.resize(target_size, Image.BILINEAR)
    
    # 转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    
    # 如果需要应用光学编码
    if apply_optical_encoding_flag and lens_file:
        print("应用光学编码...")
        # 加载PSF
        psf = load_psf(lens_file)
        if psf is not None:
            # 应用光学编码
            # 注意：这里需要确保PSF的格式正确
            # PSF应该是[31, H_psf, W_psf]的张量
            # 由于我们只有一个PSF，我们需要将其扩展为31个通道
            # 这里我们简化处理，实际应用中可能需要更复杂的处理
            psf_expanded = psf.unsqueeze(0).repeat(31, 1, 1)  # [31, H_psf, W_psf]
            image_tensor = apply_optical_encoding(image_tensor, psf_expanded)
        else:
            print("PSF加载失败，跳过光学编码")
    
    return image_tensor.unsqueeze(0)  # 添加批次维度


def predict_spectral_image(model, rgb_image_tensor, device):
    """使用模型进行光谱重建预测"""
    # 确保输入在正确设备上
    rgb_image_tensor = rgb_image_tensor.to(device)
    
    # 模型前向传播
    with torch.no_grad():
        spectral_output = model(rgb_image_tensor)
    
    return spectral_output


def save_spectral_image(spectral_tensor, output_dir, num_channels=31):
    """保存光谱图像结果为多个通道"""
    # 转换为numpy数组
    spectral_np = spectral_tensor.cpu().numpy()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个通道
    for channel_idx in range(num_channels):
        # 提取单个通道
        channel_data = spectral_np[0, channel_idx, :, :]
        
        # 归一化到0-255范围
        channel_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data)) * 255
        channel_data = channel_data.astype(np.uint8)
        
        # 生成文件名
        output_path = os.path.join(output_dir, f'predicted_spectral_channel_{channel_idx}.png')
        
        # 保存为图像
        image = Image.fromarray(channel_data, mode='L')
        image.save(output_path)
        print(f"光谱图像通道 {channel_idx} 已保存到: {output_path}")


def main():
    """主函数"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='光谱重建预测脚本')
    parser.add_argument('--apply-optical-encoding', action='store_true', 
                        help='是否在预测前应用光学编码')
    parser.add_argument('--lens-file', type=str, default='optics/lens/A_001.json',
                        help='镜头参数文件路径')
    parser.add_argument('--checkpoint-path', type=str, default='output/default/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--input-image-path', type=str, default='data/predict/1.png',
                        help='输入图像路径')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置参数
    upscale_factor = 2  # 从训练配置中读取
    num_classes = 31
    
    # 加载模型
    try:
        model = load_model(args.checkpoint_path, device, upscale_factor, num_classes)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 检查输入图像是否存在
    if not os.path.exists(args.input_image_path):
        print(f"输入图像不存在: {args.input_image_path}")
        return
    
    # 预处理图像
    try:
        rgb_image_tensor = preprocess_image(args.input_image_path, 
                                          apply_optical_encoding_flag=args.apply_optical_encoding,
                                          lens_file=args.lens_file)
        print(f"成功加载并预处理输入图像: {args.input_image_path}")
    except Exception as e:
        print(f"预处理图像失败: {e}")
        return
    
    # 进行预测
    try:
        spectral_output = predict_spectral_image(model, rgb_image_tensor, device)
        print(f"预测完成，输出形状: {spectral_output.shape}")
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        return
    
    # 保存所有通道的结果
    save_spectral_image(spectral_output, args.output_dir, num_channels=31)
    
    print("预测完成!")


if __name__ == "__main__":
    main()