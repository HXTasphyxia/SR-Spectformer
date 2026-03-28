# Lens Adaptive Multi-Dimensional High-Resolution Snapshot Spectral Imaging

本项目是一个基于深度学习的光谱重建系统，结合了FSRCNN超分辨率网络和SpectFormer模型，用于从RGB图像重建高分辨率光谱图像。

## 项目结构

```
.
├── configs/              # 配置文件目录
├── data/                 # 数据集目录
├── models/               # 模型定义
├── output/               # 训练输出目录
├── utils/                # 工具函数
├── train.py             # 训练脚本
├── run.py               # 运行训练脚本
├── predict.py           # 预测脚本
├── requirements.txt     # 依赖列表
└── README.md           # 项目说明文档
```

## 环境配置

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 训练模型

1. 准备数据集并按要求格式存放
2. （可选）生成PSF文件：
   ```bash
   python generate_psf.py --lens optics/lens/A_001.json
   ```
3. 运行训练脚本：
   ```bash
   python main.py --config configs/default.yaml
   ```

   ### 光学编码配置

   在配置文件中，可以通过以下配置项来控制光学编码：

   **禁用光学编码（默认）：**
   ```yaml
   data:
     use_optical_encoding: false
   ```

   **启用光学编码并使用镜头文件生成PSF：**
   ```yaml
   data:
     use_optical_encoding: true
     lens_file: "optics/lens/A_001.json"
   ```

   **启用光学编码并使用预生成的PSF文件：**
   ```yaml
   data:
     use_optical_encoding: true
     psf_path: "psf_outputs/A_001/psf_31channels.pt"
   ```

## 使用预测脚本

项目包含一个预测脚本 `predict.py`，用于使用训练好的模型进行光谱重建预测。

### 运行预测

```bash
python predict.py
```

### 预测脚本说明

- 自动加载 `output/default/best_model.pth` 作为模型权重
- 默认使用 `data/TOSHI/book 2/group1/rgb.jpg` 作为输入图像
- 预测结果保存在 `predictions/` 目录下
- 保存的图像为光谱数据的31个通道，每个通道分别保存为独立的PNG文件

### 自定义预测

可以通过修改 `predict.py` 中的以下参数来自定义预测：

- `checkpoint_path`: 模型检查点路径
- `input_image_path`: 输入RGB图像路径
- `upscale_factor`: 超分辨率因子

### 注意事项

1. 模型检查点文件必须与代码中的模型结构匹配
2. 输入图像需要是RGB格式的彩色图像
3. 预测结果为31通道的光谱数据，每个通道分别保存为独立的PNG文件
4. 如果遇到模型加载问题，可以调整`load_state_dict`的`strict`参数
5. 确保预测脚本中的`upscale_factor`与训练配置一致（默认为2）

## 保存光学编码数据

项目包含一个脚本 `save_optical_encoded_data.py`，用于读取镜头文件对数据集中某一个数据进行光学编码，并将经过光学编码的数据保存为31张代表不同通道的灰度图。

### 运行脚本

```bash
python save_optical_encoded_data.py
```

### 脚本参数

- `--dataset-root`: 数据集根目录 (默认: data/TOSHI)
- `--output-dir`: 输出目录 (默认: optical_encoded_output)
- `--lens-file`: 镜头参数文件路径 (默认: optics/lens/A_001.json)
- `--data-index`: 要处理的数据索引 (默认: 0)

### 示例

```bash
python save_optical_encoded_data.py --dataset-root data/TOSHI --output-dir optical_encoded_output --lens-file optics/lens/A_001.json --data-index 0
```

## 模型架构

1. **FSRCNN**: 用于RGB图像的超分辨率处理
2. **SpectFormer**: 基于Transformer的光谱重建网络

## 损失函数

损失函数已修改为对每个通道分别计算MSE并累加，以确保网络能够还原所有通道的信息。

## 输出结果

- `best_model.pth`: 最佳模型权重
- `checkpoint_epoch_*.pth`: 训练过程中的检查点
- `tensorboard/`: TensorBoard日志
- `train.log`: 训练日志