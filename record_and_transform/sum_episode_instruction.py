import os
import json
from collections import Counter

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
BASE_DIR = "/home/openpi/data/data_raw/exp21_data_auto_queue_PutAndRecord_0115/raw"

# 定义工件映射
INSTRUCTION_MAP = {
    "pick up the hollow rectangular housing": "工件 A (Hollow Rectangular Housing)",
    "pick up the silver metal cylinder": "工件 B (Silver Metal Cylinder)",
    "pick up the small upright valve": "工件 C (Small Upright Valve)",
    "pick up the flat triangular plate": "工件 D (Flat Triangular Plate)"
}

def analyze_instructions():
    if not os.path.exists(BASE_DIR):
        print(f"[Error] 找不到目录: {BASE_DIR}")
        return

    # 获取所有以 episode_ 开头的文件夹
    episode_folders = [f for f in os.listdir(BASE_DIR) 
                      if os.path.isdir(os.path.join(BASE_DIR, f)) and f.startswith("episode_")]
    
    total_episodes = len(episode_folders)
    stats = Counter()
    others = []
    error_count = 0

    print(f"正在分析 {total_episodes} 个 episodes...")

    for ep in episode_folders:
        jsonl_path = os.path.join(BASE_DIR, ep, "data.jsonl")
        
        if not os.path.exists(jsonl_path):
            print(f"[Warning] {ep} 下缺少 data.jsonl")
            error_count += 1
            continue

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                # 读取第一行
                first_line = f.readline()
                if not first_line:
                    print(f"[Warning] {ep} 的 data.jsonl 是空的")
                    error_count += 1
                    continue
                
                data = json.loads(first_line)
                instr = data.get("instruction", "").strip()
                
                # 统计
                if instr in INSTRUCTION_MAP:
                    stats[instr] += 1
                else:
                    stats["Others/Unknown"] += 1
                    others.append(f"{ep}: {instr}")
        except Exception as e:
            print(f"[Error] 解析 {ep} 失败: {e}")
            error_count += 1

    # -----------------------------------------------------------------------------
    # 打印结果报告
    # -----------------------------------------------------------------------------
    print("\n" + "="*50)
    print("                数据统计报告")
    print("="*50)
    
    four_parts_total = 0
    for key, label in INSTRUCTION_MAP.items():
        count = stats[key]
        four_parts_total += count
        print(f"{label:<40}: {count} 个")

    print("-" * 50)
    print(f"四个工件总计 (A+B+C+D)              : {four_parts_total} 个")
    print(f"目录中 Episode 总数 (文件夹数量)      : {total_episodes} 个")
    
    if others:
        print(f"未匹配/其他指令数量                  : {stats['Others/Unknown']} 个")
        # print("未匹配示例:", others[:3]) # 如果需要看具体的未匹配项可以取消注释

    if error_count:
        print(f"读取失败/空文件数量                  : {error_count} 个")

    print("-" * 50)
    if four_parts_total == total_episodes:
        print("✅ 验证结果: 四个工件总数【等于】Episode 总数。")
    else:
        diff = total_episodes - four_parts_total
        print(f"❌ 验证结果: 不相等！差距为 {diff} 个 (可能包含未匹配项或错误文件)。")
    print("="*50)

if __name__ == "__main__":
    analyze_instructions()