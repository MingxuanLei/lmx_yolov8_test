import torch
print(torch.__version__)          # 应显示 2.6.0 或更高
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_capability())  # (12, 0) ，且不应有警告