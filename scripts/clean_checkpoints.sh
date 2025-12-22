#!/bin/bash

# 配置部分
# TARGET_DIR="/root/autodl-tmp/openpi/scripts/test-dir"
TARGET_DIR="~/autodl-tmp/openpi/checkpoints/pi05_xarm_1212_night/my_experiment_1218"  # 模型保存目录
FILE_PATTERN="*00"              # 要筛选的文件后缀，例如 *.pth, *.ckpt
KEEP_COUNT=2                     # 保留最新的多少个文件

# 逻辑部分
# 1. ls -dt: 按时间排序(最新的在最前)列出完整路径
# 2. tail -n +$((KEEP_COUNT + 1)): 跳过前N个文件，选出剩下的（即旧文件）
# 3. xargs -r rm -f: 如果有输入则执行删除，没有输入不报错

ls -dt "$TARGET_DIR"/$FILE_PATTERN | tail -n +$((KEEP_COUNT + 1)) | xargs -r rm -rf
