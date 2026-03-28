#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save optical encoded data script
用于对光谱数据应用光学编码并保存结果
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from data.processing_pipeline import apply_optical_encoding


def load_spectral_data(data_dir):
    """
    加载31通道光谱数据
    Args:
        data_dir (str): 数据目录路径
    Returns:
        torch.Tensor: 光谱数据张量，形状为 [31, H, W]
    """
    # 加载所有31个波长通道的PNG文件
    channel_files = []
    for wavelength in range(400, 701, 10):  # 400, 410, ..., 700
        file_path = os.path.join(data_dir, f"{wavelength}.png")
        if os.path.exists(file_path):
            channel_files.append(file_path)
    
    # 验证通道数
    if len(channel_files) != 31:
        raise ValueError(f"应该有31个通道，但找到 {len(channel_files)} 个文件")
    
    # 按波长排序
    channel_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # 加载所有通道图像
    channels = []
    for file_path in channel_files:
        # 加载灰度图像
        img = Image.open(file_path).convert('L')
        channels.append(img)
    
    # 将所有通道堆叠成一个张量
    channel_tensors = [TF.to_tensor(img) for img in channels]  # 每个是 [1, H, W]
    spectral_tensor = torch.cat(channel_tensors, dim=0)  # [31, H, W]
    
    return spectral_tensor


def load_psf_data(psf_dir):
    """
    加载31通道PSF数据
    Args:
        psf_dir (str): PSF目录路径
    Returns:
        torch.Tensor: PSF张量，形状为 [31, H_psf, W_psf]
    """
    # 加载所有31个波长通道的PSF文件
    psf_files = []
    for wavelength in range(400, 701, 10):  # 400, 410, ..., 700
        file_path = os.path.join(psf_dir, f"psf_{wavelength}nm.pt")
        if os.path.exists(file_path):
            psf_files.append(file_path)
    
    # 验证通道数
    if len(psf_files) != 31:
        raise ValueError(f"应该有31个PSF文件，但找到 {len(psf_files)} 个文件")
    
    # 按波长排序
    psf_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('nm')[0]))
    
    # 加载所有PSF张量
    psf_tensors = []
    for file_path in psf_files:
        # 加载PSF张量
        psf = torch.load(file_path)
        psf_tensors.append(psf)
    
    # 将所有PSF张量堆叠成一个张量
    psf_tensor = torch.stack(psf_tensors, dim=0)  # [31, H_psf, W_psf]
    
    return psf_tensor


def save_spectral_image(spectral_tensor, output_dir, num_channels=31):
    """
    保存光谱图像结果为多个通道
    Args:
        spectral_tensor (torch.Tensor): 光谱数据张量，形状为 [31, H, W]
        output_dir (str): 输出目录路径
        num_channels (int): 通道数
    """
    # 转换为numpy数组
    spectral_np = spectral_tensor.cpu().numpy()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个通道
    for channel_idx in range(num_channels):
        # 提取单个通道
        channel_data = spectral_np[channel_idx, :, :]
        
        # 归一化到0-255范围
        channel_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data)) * 255
        channel_data = channel_data.astype(np.uint8)
        
        # 生成文件名
        output_path = os.path.join(output_dir, f'channel_{channel_idx:02d}.png')
        
        # 保存为图像
        image = Image.fromarray(channel_data, mode='L')
        image.save(output_path)
        print(f"光谱图像通道 {channel_idx} 已保存到: {output_path}")


def main():
    """
    主函数
    """
    import argparse
    
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='对光谱数据应用光学编码并保存结果')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录路径')
    parser.add_argument('--psf-dir', type=str, required=True,
                        help='PSF目录路径')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 获取当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整路径
    data_path = os.path.join(current_dir, args.data_dir)
    output_path = os.path.join(current_dir, args.output_dir)
    psf_path = os.path.join(current_dir, args.psf_dir)
    
    # 输出本次运行使用的镜头和数据名称
    data_name = os.path.basename(args.data_dir)
    psf_name = os.path.basename(args.psf_dir)
    print(f"本次运行使用的数据: {data_name}")
    print(f"本次运行使用的镜头: {psf_name}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_path):
        print(f"数据目录不存在: {data_path}")
        return
    
    # 检查PSF目录是否存在
    if not os.path.exists(psf_path):
        print(f"PSF目录不存在: {psf_path}")
        return
    
    # 加载光谱数据
    try:
        spectral_data = load_spectral_data(data_path)
        print(f"成功加载光谱数据，形状: {spectral_data.shape}")
    except Exception as e:
        print(f"加载光谱数据失败: {e}")
        return
    
    # 加载PSF数据
    try:
        psf_data = load_psf_data(psf_path)
        print(f"成功加载PSF数据，形状: {psf_data.shape}")
    except Exception as e:
        print(f"加载PSF数据失败: {e}")
        return
    
    # 应用光学编码
    try:
        encoded_data = apply_optical_encoding(spectral_data, psf_data)
        print(f"成功应用光学编码，形状: {encoded_data.shape}")
    except Exception as e:
        print(f"应用光学编码失败: {e}")
        return
    
    # 保存结果
    try:
        save_spectral_image(encoded_data, output_path)
        print("所有通道已保存完成!")
    except Exception as e:
        print(f"保存结果失败: {e}")
        return


if __name__ == "__main__":
    main()