import os
import shutil
import re

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SOURCE_BASE = "/home/openpi/data/data_raw/test/raw"
TARGET_BASE = "/home/openpi/data/data_raw/exp21_data_auto_queue_PutAndRecord_0115"

# 起始和结束的 episode 编号
START_NUM = 1593
END_NUM = 1684

# 目标起始编号
TARGET_START_NUM = 6830

def rename_and_move():
    # 1. 确保目标目录存在
    if not os.path.exists(TARGET_BASE):
        os.makedirs(TARGET_BASE)
        print(f"[Init] 创建目标目录: {TARGET_BASE}")

    print(f"[Process] 开始从 {START_NUM} 处理到 {END_NUM}...")

    success_count = 0
    
    # 2. 遍历指定的范围
    for i in range(START_NUM, END_NUM + 1):
        old_folder_name = f"episode_{i}"
        old_path = os.path.join(SOURCE_BASE, old_folder_name)
        
        # 检查源文件夹是否存在
        if not os.path.exists(old_path):
            print(f"[Warning] 跳过: 找不到源文件夹 {old_path}")
            continue
            
        # 3. 计算新编号和新路径
        # 偏移量 = 6830 - 1593 = 5237
        new_num = i + (TARGET_START_NUM - START_NUM)
        new_folder_name = f"episode_{new_num}"
        new_path = os.path.join(TARGET_BASE, new_folder_name)
        
        # 4. 执行移动并重命名
        try:
            # 使用 shutil.move，它支持跨分区移动并重命名
            shutil.move(old_path, new_path)
            print(f"[Success] {old_folder_name} -> {new_path}")
            success_count += 1
        except Exception as e:
            print(f"[Error] 移动 {old_folder_name} 失败: {e}")

    print("-" * 50)
    print(f"任务完成！成功移动并重命名了 {success_count} 个文件夹。")

if __name__ == "__main__":
    rename_and_move()