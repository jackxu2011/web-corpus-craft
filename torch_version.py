import torch
# 查看 torch 版本
print("PyTorch 版本:", torch.__version__)
# 查看 CUDA 是否可用（若为 GPU 环境）
print("CUDA 可用状态:", torch.cuda.is_available())
# 查看 CUDA 版本（若 CUDA 可用）
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("GPU 设备名称:", torch.cuda.get_device_name(0))  # 查看第1块GPU名称
