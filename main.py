import os
# 设置环境变量解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from train import run_training

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='光谱重建训练主入口')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='训练配置文件路径 (默认: configs/default.yaml)')
    args = parser.parse_args()

    # 调用训练函数
    run_training(args.config)