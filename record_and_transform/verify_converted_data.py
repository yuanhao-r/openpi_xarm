# import pandas as pd
# from pathlib import Path

# # ================= 配置区域 =================
# DATASET_PATH = "/home/openpi/data/data_converted/exp19_lerobot_autoPut_data_0113night_224_224/xarm_autoPut_pi05_dataset"
# TARGET_EPISODE_IDX = 0
# # ===========================================

# def export_raw_data():
#     base_path = Path(DATASET_PATH)
#     # 核心修改：根据你的截图，parquet 在 data/chunk-000 下
#     data_dir = base_path / "data" / "chunk-000"
    
#     print(f"📂 正在扫描目录: {data_dir}")

#     # 1. 查找所有的 parquet 文件
#     parquet_files = sorted(list(data_dir.glob("episode_*.parquet")))
    
#     if not parquet_files:
#         print(f"❌ 错误：在 {data_dir} 找不到 episode_xxxxxx.parquet 文件")
#         return

#     # 2. 找到目标 episode 的文件
#     # 格式为 episode_000000.parquet，所以我们要把数字补齐到6位
#     target_filename = f"episode_{TARGET_EPISODE_IDX:06d}.parquet"
#     target_file = data_dir / target_filename

#     if not target_file.exists():
#         print(f"❌ 错误：找不到目标文件 {target_filename}")
#         print(f"当前目录下存在的文件示例: {[f.name for f in parquet_files[:3]]}")
#         return

#     print(f"📖 正在读取: {target_filename}")
#     df = pd.read_parquet(target_file)

#     # 3. 导出 CSV
#     output_filename = f"episode_{TARGET_EPISODE_IDX}_full_data.csv"
#     # 保存到当前脚本运行的目录
#     # --- 新增：限制浮点数精度 ---
#     # 仅针对浮点数类型的列进行四舍五入，保留 6 位小数
#     df = df.round(4) 
#     # --------------------------

#     # 保存到当前脚本运行的目录
#     df.to_csv(output_filename, index=False)
    
#     print(f"\n✅ 导出成功！")
#     print(f"📍 保存位置: {Path.cwd() / output_filename}")
    
#     # 4. 数据预览
#     print("\n--- 核心数据预览 (前5行) ---")
#     # 自动识别列名（OpenPI 格式通常包含 state, action 等）
#     available_cols = df.columns.tolist()
#     # 挑选一些关键列展示，防止列太多刷屏
#     preview_cols = [c for c in available_cols if 'state' in c or 'action' in c or 'index' in c][:8]
#     print(df[preview_cols].head().to_string(index=False))

# if __name__ == "__main__":
#     export_raw_data()

import pandas as pd
from pathlib import Path
import json
import numpy as np

# ================= 配置区域 =================
DATASET_PATH = "/home/openpi/data/data_converted/exp1_lerobot_autoPut_data_0128night_224_224/xarm_autoPut_pi05_dataset"
TARGET_EPISODE_IDX = 0
# ===========================================

def export_raw_data():
    base_path = Path(DATASET_PATH)
    data_dir = base_path / "data" / "chunk-000"
    
    target_filename = f"episode_{TARGET_EPISODE_IDX:06d}.parquet"
    target_file = data_dir / target_filename

    if not target_file.exists():
        print(f"❌ 错误：找不到文件 {target_file}")
        return

    print(f"📖 正在读取: {target_filename}")
    df = pd.read_parquet(target_file)

    # --- 核心修复：移除图像列 ---
    # 图像数据通常很大且是二进制，无法直接转 JSON
    cols_to_keep = [c for c in df.columns if 'image' not in c.lower()]
    df_numeric = df[cols_to_keep].copy()
    print(f"✂️ 已过滤图像列，保留字段: {cols_to_keep}")

    # --- 精度处理：缩减至 4 位小数 ---
    def round_nested(val):
        if isinstance(val, (list, np.ndarray)):
            return [round(float(x), 8) for x in val]
        elif isinstance(val, (float, np.float32, np.float64)):
            return round(float(val), 8)
        return val

    # 使用新的 .map() 代替被弃用的 .applymap()
    df_numeric = df_numeric.map(round_nested)

    # --- 导出为 JSON ---
    output_filename = f"episode_{TARGET_EPISODE_IDX}_numeric_data.json"
    
    try:
        df_numeric.to_json(output_filename, orient='records', force_ascii=False, indent=2)
        print(f"\n✅ 导出成功！")
        print(f"📍 文件位置: {Path.cwd() / output_filename}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")

    # --- 预览第一帧 ---
    print("\n--- 第一帧数值数据预览 ---")
    print(json.dumps(df_numeric.iloc[0].to_dict(), indent=4))

if __name__ == "__main__":
    export_raw_data()