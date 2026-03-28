# train.py
import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

# 导入自定义模块
from data.dataset import SpectralDataset
from data.optical_dataset import OpticalSpectralDataset
from data.transforms import ToTensor, NormalizeSRF
from data.srfs import SRF_BGR_31_CHANNEL_400_700NM
from data.processing_pipeline import apply_optical_encoding, spectral_to_rgb, downsample_image, split_image_into_patches, reconstruct_image_from_patches, rgb_to_spectral
from models import SuperResolvedSpectformer
from models.fsrcnn import FSRCNN
from utils.losses import WeightedMSELoss
from utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse
from utils.utils import save_checkpoint, load_checkpoint, setup_logger, validate, create_tensorboard_writer

import yaml


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def ensure_numeric_types(config):
    """确保配置中的数值类型正确"""
    # 训练配置数值转换
    if 'training' in config:
        training = config['training']
        if 'learning_rate' in training:
            training['learning_rate'] = float(training['learning_rate'])
        if 'lr_min' in training:
            training['lr_min'] = float(training['lr_min'])
        if 'weight_decay' in training:
            training['weight_decay'] = float(training['weight_decay'])
        if 'epochs' in training:
            training['epochs'] = int(training['epochs'])
        if 'val_freq' in training:
            training['val_freq'] = int(training['val_freq'])
    
    # 数据配置数值转换
    if 'data' in config:
        data = config['data']
        if 'batch_size' in data:
            data['batch_size'] = int(data['batch_size'])
        if 'num_workers' in data:
            data['num_workers'] = int(data['num_workers'])
        if 'image_size' in data:
            data['image_size'] = int(data['image_size'])
    
    # 输出配置数值转换
    if 'output' in config:
        output = config['output']
        if 'checkpoint_freq' in output:
            output['checkpoint_freq'] = int(output['checkpoint_freq'])
        if 'val_split' in output:
            output['val_split'] = float(output['val_split'])
    
    # 损失函数配置数值转换
    if 'loss' in config:
        loss = config['loss']
        if 'mse_weight' in loss:
            loss['mse_weight'] = float(loss['mse_weight'])
        if 'ssim_weight' in loss:
            loss['ssim_weight'] = float(loss['ssim_weight'])
        if 'lpips_weight' in loss:
            loss['lpips_weight'] = float(loss['lpips_weight'])
    
    return config


def run_training(config_path):
    # 加载配置文件
    config = load_config(config_path)
    config = ensure_numeric_types(config)  # 确保数值类型正确

    # 使用安全的配置参数获取方式，避免KeyError
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    output_config = config.get('output', {})
    
    batch_size = data_config.get('batch_size', 4)
    num_workers = data_config.get('num_workers', 2)
    lr = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    epochs = training_config.get('epochs', 50)
    lr_min = training_config.get('lr_min', 1e-6)
    val_freq = training_config.get('val_freq', 5)
    image_size = data_config.get('image_size', 512)
    upscale_factor = int(data_config.get('sr_upscale', 1))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(os.path.join(output_config['dir'], output_config['log_file']))
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_config['dir'], exist_ok=True)
    
    # 创建TensorBoard写入器
    writer = create_tensorboard_writer(os.path.join(output_config['dir'], 'tensorboard'))

    
    # 加载光谱响应函数
    # 加载光谱响应函数，SRF_BGR_31_CHANNEL_400_700NM是3通道，每个通道31个数值
    srfs = np.array(SRF_BGR_31_CHANNEL_400_700NM)

    # 确保数据根目录路径为绝对路径
    data_root = os.path.abspath(data_config['data_root'])

    # 获取PSF路径配置
    psf_path = data_config.get('psf_path', None)
    
    # 获取光学编码配置
    use_optical_encoding = data_config.get('use_optical_encoding', False)
    lens_file = data_config.get('lens_file', None)
    
    # 如果启用光学编码但没有提供PSF路径，则尝试生成PSF
    if use_optical_encoding and psf_path is None and lens_file is not None:
        # 导入镜头类
        from optics.generate_psf import Lens
        
        # 生成PSF
        print(f"正在为镜头文件 {lens_file} 生成PSF...")
        wavelengths = [400 + i * 10 for i in range(31)]  # 400, 410, ..., 700 (in nm)
        wavelengths_um = [w / 1000.0 for w in wavelengths]  # Convert to micrometers
        point = [0.0, 0.0, -1000.0]  # Point source position
        
        # Create lens object
        lens = Lens(lens_file)
        
        # Generate PSF for each wavelength
        psf_tensors = []
        for i, wvln in enumerate(wavelengths_um):
            print(f"  生成波长 {wavelengths[i]}nm ({wvln}um) 的PSF...")
            psf = lens.psf(point, wvln=wvln)
            psf_tensors.append(psf)
        
        # Stack all PSF tensors into a single tensor [31, H, W]
        psf_tensor = torch.stack(psf_tensors, dim=0)
        # 将PSF张量移动到指定设备
        psf_tensor = psf_tensor.to(device)
        print("PSF生成完成。")
    elif use_optical_encoding and psf_path is not None:
        # 加载预生成的PSF
        psf_tensor = torch.load(psf_path, map_location=device)
        # 确保PSF张量在正确的设备上
        psf_tensor = psf_tensor.to(device)
        print(f"已加载PSF文件: {psf_path}")
    else:
        # 不使用光学编码
        psf_tensor = None
        print("未启用光学编码。")
    
    # 加载数据集
    # 根据用户要求的工作流程，我们统一使用原始数据集，然后在训练循环中进行处理
    train_dataset = SpectralDataset(
        root_dir=data_root,
        image_size=image_size,
        srfs=srfs,
        is_train=True
    )
    
    val_dataset = SpectralDataset(
        root_dir=data_root,
        image_size=image_size,
        srfs=srfs,
        is_train=False
    )
    
    # 使用随机划分
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # 使用随机划分
    indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建子集
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if device.type == 'cuda' else False, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True if device.type == 'cuda' else False, 
        drop_last=False
    )
    
    logger.info(f"数据加载完成 - 训练集: {len(train_subset)}样本, 验证集: {len(val_subset)}样本")
    logger.info(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}")

    # 初始化模型
    logger.info("初始化SpectFormer模型")
    model = SuperResolvedSpectformer(
        upscale_factor=upscale_factor,
        num_classes=31,
        token_label=False,
        image_size=image_size,
        spectformer_variant='s'  # 指定模型变体
    )
    model = model.to(device)
    
    # 初始化FSRCNN超分网络
    logger.info("初始化FSRCNN模型")
    fsrcnn = FSRCNN(upscale_factor=upscale_factor)
    fsrcnn = fsrcnn.to(device)
    
    logger.info(f"模型初始化完成 - SpectFormer参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"模型初始化完成 - FSRCNN参数数量: {sum(p.numel() for p in fsrcnn.parameters()):,}")

    # 初始化损失函数和优化器
    logger.info("初始化损失函数和优化器")
    criterion = WeightedMSELoss(weights=srfs, device=device)
    # 为两个模型分别创建优化器
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(fsrcnn.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # 确保lr_min是浮点数
    lr_min = float(lr_min)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    
    # 开始训练循环
    logger.info(f"开始训练 - 总epochs: {epochs}, 批次大小: {batch_size}, 设备: {device}")
    logger.info(f"学习率: {lr}, 最小学习率: {lr_min}, 权重衰减: {weight_decay}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"===== 开始训练 epoch {epoch+1}/{epochs} =====")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (rgb_images, spectral_targets, _) in enumerate(tqdm(train_loader, desc=f'训练 epoch {epoch+1}/{epochs}')):
            # 确保数据在正确设备上
            spectral_targets = spectral_targets.to(device)  # [B, 31, H, W]
            
            # 获取批次大小和设备
            batch_size = spectral_targets.shape[0]
            device = spectral_targets.device
            
            # 使用已加载的PSF（如果需要光学编码）
            # psf_tensor已经在训练开始前根据配置加载或生成
            
            # 存储处理后的输入和目标
            processed_inputs = []
            processed_targets = []
            
            # 对批次中的每个样本进行处理
            for i in range(batch_size):
                # 获取单个样本
                single_spectral_target = spectral_targets[i]  # [31, H, W]
                
                # 1. 判断是否进行光学编码
                if psf_tensor is not None:
                    # 计算PSF并进行卷积将图像变成经过光学成像后的图像作为输入
                    single_spectral_input = apply_optical_encoding(single_spectral_target, psf_tensor)
                else:
                    # 不进行光学编码，直接使用原始数据
                    single_spectral_input = single_spectral_target
                
                # 2. 将无论经过或者不经过光学编码的数据按照srf从31通道映射为RGB3通道
                rgb_input = spectral_to_rgb(single_spectral_input)  # [3, H, W]
                rgb_target = spectral_to_rgb(single_spectral_target)  # [3, H, W]
                
                # 3. 将图像的分辨率按照上采样因子的大小进行降采样，以模拟图像分辨率的降低
                # 注意：用户要求的是降采样，但后面又要求上采样，这里我们先降采样再上采样
                downsampled_rgb_input = downsample_image(rgb_input, upscale_factor)  # [3, H//upscale_factor, W//upscale_factor]
                downsampled_rgb_target = downsample_image(rgb_target, upscale_factor)  # [3, H//upscale_factor, W//upscale_factor]
                
                # 4. 判断图像的尺寸是否过大，过大的标准是与512进行对比
                _, H, W = downsampled_rgb_input.shape
                if H > 512 or W > 512:
                    # 5. 如果图像大小大于512*512，则将图像裁切为数个512*512的块状图像
                    # 这些块状图像按分割位置直接拼接，无重叠区域
                    input_patches, input_positions = split_image_into_patches(downsampled_rgb_input, patch_size=512, overlap=0)
                    target_patches, target_positions = split_image_into_patches(downsampled_rgb_target, patch_size=512, overlap=0)
                    
                    # 6. 将这些512*512的块状图像传递给图像超分网络fsrcnn
                    sr_patches = []
                    for patch in input_patches:
                        # 从[3,512,512]变成[3,H,W](按照上采样因子的大小确定,现在将上采样因子规定为2)
                        sr_patch = fsrcnn(patch.unsqueeze(0))  # [1, 3, 512*upscale_factor, 512*upscale_factor]
                        sr_patches.append(sr_patch.squeeze(0))  # [3, 512*upscale_factor, 512*upscale_factor]
                    
                    # 7. 将fsrcnn的结果输入给sprcformer网络将其从[3,H,W]重建为[31,H,W]
                    recon_patches = []
                    for patch in sr_patches:
                        # 将RGB图像转换回31通道光谱图像
                        spectral_patch = rgb_to_spectral(patch)  # [31, 512*upscale_factor, 512*upscale_factor]
                        recon_patch = model(spectral_patch.unsqueeze(0))  # [1, 31, 512*upscale_factor, 512*upscale_factor]
                        recon_patches.append(recon_patch.squeeze(0))  # [31, 512*upscale_factor, 512*upscale_factor]
                    
                    # 8. 根据分割的位置将这些块拼接为原读入图像大小
                    # 重建完整图像
                    recon_input = reconstruct_image_from_patches(recon_patches, input_positions, (31, H*upscale_factor, W*upscale_factor))
                    processed_inputs.append(recon_input)
                    processed_targets.append(single_spectral_target)
                else:
                    # 图像大小等于或小于512*512，不需要裁切
                    # 直接传递给图像超分网络fsrcnn
                    sr_input = fsrcnn(downsampled_rgb_input.unsqueeze(0))  # [1, 3, H*upscale_factor, W*upscale_factor]
                    sr_input = sr_input.squeeze(0)  # [3, H*upscale_factor, W*upscale_factor]
                    
                    # 将fsrcnn的结果输入给sprcformer网络将其从[3,H,W]重建为[31,H,W]
                    spectral_input = rgb_to_spectral(sr_input)  # [31, H*upscale_factor, W*upscale_factor]
                    recon_input = model(spectral_input.unsqueeze(0))  # [1, 31, H*upscale_factor, W*upscale_factor]
                    recon_input = recon_input.squeeze(0)  # [31, H*upscale_factor, W*upscale_factor]
                    
                    processed_inputs.append(recon_input)
                    processed_targets.append(single_spectral_target)
            
            # 如果有处理后的数据，则进行训练
            if len(processed_inputs) > 0 and len(processed_targets) > 0:
                # 堆叠处理后的输入和目标
                processed_inputs = torch.stack(processed_inputs, dim=0)  # [B', 31, H, W]
                processed_targets = torch.stack(processed_targets, dim=0)  # [B', 31, H, W]
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(processed_inputs)  # 模型处理31通道输入
                
                # 计算损失
                loss = criterion(outputs, processed_targets)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 记录训练进度
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                    writer.add_scalar('Train/BatchLoss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.6f}")
        writer.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
        
        # 验证阶段
        if (epoch + 1) % val_freq == 0:
            logger.info(f"===== 开始验证 epoch {epoch+1}/{epochs} =====")
            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            val_ssim = 0.0
            val_rmse = 0.0
            
            with torch.no_grad():
                for batch_idx, (rgb_images, spectral_targets, _) in enumerate(tqdm(val_loader, desc=f'验证 epoch {epoch+1}/{epochs}')):
                    # 确保数据在正确设备上
                    spectral_targets = spectral_targets.to(device)  # [B, 31, H, W]
                    
                    # 获取批次大小和设备
                    batch_size = spectral_targets.shape[0]
                    device = spectral_targets.device
                    
                    # 使用已加载的PSF（如果需要光学编码）
                    # psf_tensor已经在训练开始前根据配置加载或生成
                    
                    # 存储处理后的输入和目标
                    processed_inputs = []
                    processed_targets = []
                    
                    # 对批次中的每个样本进行处理
                    for i in range(batch_size):
                        # 获取单个样本
                        single_spectral_target = spectral_targets[i]  # [31, H, W]
                        
                        # 1. 判断是否进行光学编码
                        if psf_tensor is not None:
                            # 计算PSF并进行卷积将图像变成经过光学成像后的图像作为输入
                            single_spectral_input = apply_optical_encoding(single_spectral_target, psf_tensor)
                        else:
                            # 不进行光学编码，直接使用原始数据
                            single_spectral_input = single_spectral_target
                        
                        # 2. 将无论经过或者不经过光学编码的数据按照srf从31通道映射为RGB3通道
                        rgb_input = spectral_to_rgb(single_spectral_input)  # [3, H, W]
                        rgb_target = spectral_to_rgb(single_spectral_target)  # [3, H, W]
                        
                        # 3. 将图像的分辨率按照上采样因子的大小进行降采样，以模拟图像分辨率的降低
                        downsampled_rgb_input = downsample_image(rgb_input, upscale_factor)  # [3, H//upscale_factor, W//upscale_factor]
                        downsampled_rgb_target = downsample_image(rgb_target, upscale_factor)  # [3, H//upscale_factor, W//upscale_factor]
                        
                        # 4. 判断图像的尺寸是否过大，过大的标准是与512进行对比
                        _, H, W = downsampled_rgb_input.shape
                        if H > 512 or W > 512:
                            # 5. 如果图像大小大于512*512，则将图像裁切为数个512*512的块状图像
                            input_patches, input_positions = split_image_into_patches(downsampled_rgb_input, patch_size=512, overlap=0)
                            target_patches, target_positions = split_image_into_patches(downsampled_rgb_target, patch_size=512, overlap=0)
                            
                            # 6. 将这些512*512的块状图像传递给图像超分网络fsrcnn
                            sr_patches = []
                            for patch in input_patches:
                                sr_patch = fsrcnn(patch.unsqueeze(0))  # [1, 3, 512*upscale_factor, 512*upscale_factor]
                                sr_patches.append(sr_patch.squeeze(0))  # [3, 512*upscale_factor, 512*upscale_factor]
                            
                            # 7. 将fsrcnn的结果输入给sprcformer网络将其从[3,H,W]重建为[31,H,W]
                            recon_patches = []
                            for patch in sr_patches:
                                # 将RGB图像转换回31通道光谱图像
                                spectral_patch = rgb_to_spectral(patch)  # [31, 512*upscale_factor, 512*upscale_factor]
                                recon_patch = model(spectral_patch.unsqueeze(0))  # [1, 31, 512*upscale_factor, 512*upscale_factor]
                                recon_patches.append(recon_patch.squeeze(0))  # [31, 512*upscale_factor, 512*upscale_factor]
                            
                            # 8. 根据分割的位置将这些块拼接为原读入图像大小
                            # 重建完整图像
                            recon_input = reconstruct_image_from_patches(recon_patches, input_positions, (31, H*upscale_factor, W*upscale_factor))
                            target_full = reconstruct_image_from_patches(target_patches, target_positions, (31, H*upscale_factor, W*upscale_factor))
                            processed_inputs.append(recon_input)
                            processed_targets.append(target_full)
                        else:
                            # 图像大小等于或小于512*512，不需要裁切
                            # 直接传递给图像超分网络fsrcnn
                            sr_input = fsrcnn(downsampled_rgb_input.unsqueeze(0))  # [1, 3, H*upscale_factor, W*upscale_factor]
                            sr_input = sr_input.squeeze(0)  # [3, H*upscale_factor, W*upscale_factor]
                            
                            # 将fsrcnn的结果输入给sprcformer网络将其从[3,H,W]重建为[31,H,W]
                            spectral_input = rgb_to_spectral(sr_input)  # [31, H*upscale_factor, W*upscale_factor]
                            recon_input = model(spectral_input.unsqueeze(0))  # [1, 31, H*upscale_factor, W*upscale_factor]
                            recon_input = recon_input.squeeze(0)  # [31, H*upscale_factor, W*upscale_factor]
                            
                            # 对目标图像也进行相同的降采样处理
                            downsampled_target = downsample_image(rgb_target, upscale_factor)  # [3, H//upscale_factor, W//upscale_factor]
                            # 超分处理
                            sr_target = fsrcnn(downsampled_target.unsqueeze(0))  # [1, 3, H, W]
                            sr_target = sr_target.squeeze(0)  # [3, H, W]
                            # 转换回光谱图像
                            spectral_target = rgb_to_spectral(sr_target)  # [31, H, W]
                            
                            processed_inputs.append(recon_input)
                            processed_targets.append(spectral_target)
                    
                    # 如果有处理后的数据，则进行验证
                    if len(processed_inputs) > 0 and len(processed_targets) > 0:
                        # 堆叠处理后的输入和目标
                        processed_inputs = torch.stack(processed_inputs, dim=0)  # [B', 31, H, W]
                        processed_targets = torch.stack(processed_targets, dim=0)  # [B', 31, H, W]
                        
                        # 前向传播
                        outputs = model(processed_inputs)
                        
                        # 计算损失
                        loss = criterion(outputs, processed_targets)
                        val_loss += loss.item()
                        
                        # 计算评估指标
                        psnr = calculate_psnr(outputs, processed_targets)
                        ssim = calculate_ssim(outputs, processed_targets)
                        rmse = calculate_rmse(outputs, processed_targets)
                        
                        val_psnr += psnr
                        val_ssim += ssim
                        val_rmse += rmse
            
            # 计算平均验证指标
            avg_val_loss = val_loss / len(val_loader)
            avg_val_psnr = val_psnr / len(val_loader)
            avg_val_ssim = val_ssim / len(val_loader)
            avg_val_rmse = val_rmse / len(val_loader)
            
            logger.info(f"Epoch {epoch+1} 验证损失: {avg_val_loss:.6f}")
            logger.info(f"Epoch {epoch+1} PSNR: {avg_val_psnr:.2f} dB")
            logger.info(f"Epoch {epoch+1} SSIM: {avg_val_ssim:.4f}")
            logger.info(f"Epoch {epoch+1} RMSE: {avg_val_rmse:.6f}")
            
            # 记录验证指标
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/PSNR', avg_val_psnr, epoch)
            writer.add_scalar('Val/SSIM', avg_val_ssim, epoch)
            writer.add_scalar('Val/RMSE', avg_val_rmse, epoch)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, avg_val_loss, os.path.join(output_config['dir'], 'best_model.pth'))
                logger.info(f"Epoch {epoch+1} 保存最佳模型")
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch+1} 结束 - 学习率: {current_lr:.6f}")
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # 保存检查点
        if (epoch + 1) % output_config.get('checkpoint_freq', 10) == 0:
            save_checkpoint(model, optimizer, epoch, avg_train_loss, os.path.join(output_config['dir'], f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Epoch {epoch+1} 保存检查点")
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} 耗时: {epoch_time:.2f}秒")
    
    # 训练结束
    logger.info("训练完成")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectral Reconstruction Training')
    parser.add_argument('--config', type=str, required=True, help='YAML配置文件路径')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    run_training(args.config)