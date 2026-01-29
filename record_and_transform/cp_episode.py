import os
import shutil

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
# 源根目录
SOURCE_BASE = "/home/openpi/data/data_raw/test"
# 目标根目录
TARGET_BASE = "/home/openpi/data/data_raw/exp21_data_auto_queue_PutAndRecord_0115/raw"

# 范围定义
START_NUM = 6830
END_NUM = 6921
# 目标起始编号
TARGET_START_NUM = 6830

def copy_and_rename():
    # 1. 确保目标根目录存在
    if not os.path.exists(TARGET_BASE):
        os.makedirs(TARGET_BASE)
        print(f"[Init] 创建目标目录: {TARGET_BASE}")

    print(f"[Process] 开始从 {START_NUM} 处理到 {END_NUM}...")
    success_count = 0

    # 2. 计算偏移量
    offset = TARGET_START_NUM - START_NUM

    for i in range(START_NUM, END_NUM + 1):
        old_folder_name = f"episode_{i}"
        old_path = os.path.join(SOURCE_BASE, old_folder_name)
        
        # 检查源文件夹是否存在
        if not os.path.exists(old_path):
            print(f"[Warning] 跳过: 找不到源文件夹 {old_path}")
            continue
            
        # 3. 计算新路径
        new_num = i + offset
        new_folder_name = f"episode_{new_num}"
        new_path = os.path.join(TARGET_BASE, new_folder_name)
        
        # 4. 执行复制
        try:
            # 如果目标文件夹已存在，先删除（可选，防止合并冲突）
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
                
            shutil.copytree(old_path, new_path)
            print(f"[Success] Copied: {old_folder_name} -> {new_folder_name}")
            success_count += 1
        except Exception as e:
            print(f"[Error] 复制 {old_folder_name} 失败: {e}")

    print("-" * 50)
    print(f"任务完成！共成功复制并重命名了 {success_count} 个文件夹。")

if __name__ == "__main__":
    copy_and_rename()