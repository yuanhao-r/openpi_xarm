import sys
# print("Python sys.path:")
# for p in sys.path:
#     print(f"  - {p}")

import lerobot
# print(f"\nlerobot模块位置: {lerobot.__file__}")

import os

# 找到lerobot的安装位置
lerobot_path = os.path.dirname(lerobot.__file__)
print(f"lerobot包路径: {lerobot_path}")

# 检查是否存在datasets目录
datasets_path = os.path.join(lerobot_path, 'datasets')
print(f"datasets路径存在: {os.path.exists(datasets_path)}")

if os.path.exists(datasets_path):
    print("datasets目录内容:")
    for item in os.listdir(datasets_path):
        print(f"  - {item}")