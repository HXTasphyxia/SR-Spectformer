import torch
print("PyTorch版本:", torch.__version__)
print("是否可用GPU:", torch.cuda.is_available())  # 应返回True
print("GPU数量:", torch.cuda.device_count())
print("当前GPU名称:", torch.cuda.get_device_name(0))  # 显示显卡型号