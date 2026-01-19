import os
import json
import glob

def update_instructions(base_path, start_ep, end_ep, new_instruction):
    """
    批量更新指定范围内 episode 的 instruction 字段
    """
    # 确保基础路径存在
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
        return

    print(f"开始处理 episode_{start_ep} 到 episode_{end_ep} ...")
    
    success_count = 0
    
    for i in range(start_ep, end_ep + 1):
        ep_dir = os.path.join(base_path, f"episode_{i}")
        file_path = os.path.join(ep_dir, "data.jsonl")
        
        # 检查该 episode 文件夹和文件是否存在
        if not os.path.exists(file_path):
            print(f"跳过: {file_path} (文件不存在)")
            continue
            
        # 读取内容并修改
        updated_lines = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    # 解析 JSON
                    data = json.loads(line)
                    # 修改 instruction
                    data["instruction"] = new_instruction
                    # 重新转换为字符串
                    updated_lines.append(json.dumps(data, ensure_ascii=False))
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(updated_lines) + "\n")
            
            print(f"已完成: {ep_dir}")
            success_count += 1
            
        except Exception as e:
            print(f"处理 {file_path} 时出错: {e}")

    print(f"\n任务完成！成功更新了 {success_count} 个 episode。")

if __name__ == "__main__":
    # --- 配置参数 ---
    RAW_DATA_PATH = "/home/openpi/data/data_raw/exp20_data_auto_queue_PutAndRecord_0114/raw"
    # RAW_DATA_PATH = "/home/openpi/record_and_transform/test"
    START_EPISODE = 0     # 起始编号
    END_EPISODE = 1017       # 结束编号（包含此编号）
    NEW_TEXT = "pick up the small upright valve"
    # ----------------

    update_instructions(RAW_DATA_PATH, START_EPISODE, END_EPISODE, NEW_TEXT)