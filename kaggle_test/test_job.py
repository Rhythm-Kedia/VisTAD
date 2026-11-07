import json
import os
import platform
import sys
import datetime

print("=" * 50)
print("     KAGGLE AUTOMATION AGENT SMOKE TEST")
print("=" * 50)
print(f"Timestamp:       {datetime.datetime.now().isoformat()} UTC")
print(f"Python Version:  {sys.version.split()[0]}")
print(f"OS Platform:     {platform.platform()}")

# Verify PyTorch availability
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_avail}")
    if cuda_avail:
        print(f"Device Name:     {torch.cuda.get_device_name(0)}")
except ImportError:
    cuda_avail = False
    print("PyTorch not installed.")

# Write test result artifact to /kaggle/working/
output_data = {
    "status": "SUCCESS",
    "timestamp": datetime.datetime.now().isoformat(),
    "python_version": sys.version,
    "cuda_available": cuda_avail,
    "message": "Automated push-and-run workflow executed flawlessly!"
}

output_path = "/kaggle/working/smoke_test_result.json"
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n[+] Created output artifact: {output_path}")
print("=" * 50)
