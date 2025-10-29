import torch
import intel_extension_for_pytorch as ipex

print("PyTorch version:", torch.__version__)
print("IPEX version:", ipex.__version__)
print("XPU available:", torch.xpu.is_available())
print("XPU device count:", torch.xpu.device_count())

if torch.xpu.is_available():
    print("XPU device name:", torch.xpu.get_device_name(0))
    # 测试简单的 XPU 操作
    x = torch.randn(3, 3).to('xpu')
    print("Test tensor on XPU:", x.device)
    print("XPU test successful!")
else:
    print("\nXPU is NOT available. Possible reasons:")
    print("1. Intel GPU driver not installed")
    print("2. Intel Extension for PyTorch not properly installed")
    print("3. No Intel GPU hardware detected")

