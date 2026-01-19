import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import sys
# 确保包含 openpi 文件夹的上一层目录 (src) 在路径中
sys.path.append("/home/openpi/src")
from openpi.shared import image_tools

# 配置项
SCALE_FACTOR = 1.0
image_resolution = (224, 224)
SCAN_INTERVAL = 5  # 每隔5秒扫描一次新增episode
MAX_RETRY = 3      # 单个episode转换失败重试次数

# 新增：创建空图像（用于填充缺失的摄像头数据）
def create_empty_image(resolution):
    """创建黑色空图像（RGB格式）"""
    H, W = resolution
    return np.zeros((H, W, 3), dtype=np.uint8)

def create_dataset(repo_id, root_dir, robot_type="xarm", incremental=False):
    """创建/加载数据集（增量模式不删除旧数据）"""
    root_dir = Path(root_dir)
    output_path = root_dir / repo_id
    
    # 增量模式：保留原有数据，仅加载；全量模式：清空重建
    if not incremental and output_path.exists():
        print(f"[全量模式] 清理旧数据集: {output_path}")
        shutil.rmtree(output_path)

    features = {
        # # 绝对值
        # "observation.state": {
        #     "dtype": "float32",
        #     "shape": (7,), 
        #     "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        # },
        # "action": {
        #     "dtype": "float32",
        #     "shape": (7,),
        #     "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        # },
        #------------------修改--------------------#
        #------------------修改--------------------#
        #------------------修改--------------------#
        # Input State: 现在代表【相对于本回合起始点的位移】(Relative to Start)
        # 形状: 7维 (dx, dy, dz, dRx, dRy, dRz, gripper_abs)
        "observation.state": {
            "dtype": "float32",
            "shape": (7,), 
            "names": ["rel_x", "rel_y", "rel_z", "rel_roll", "rel_pitch", "rel_yaw", "gripper"],
        },
        
        # Output Action: 现在代表【相对于上一帧的增量】(Delta)
        # 形状: 7维 (dx, dy, dz, dRx, dRy, dRz, gripper_abs)
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dr", "dp", "dyaw", "gripper"],
        },
        
        # 语言指令
        "language_instruction": {
            "dtype": "string",
            "shape": (1,),
            "names": ["instruction"],
        },
    }
    
    # 摄像头配置
    cameras = ["cam_left_wrist", "cam_right_wrist"]
    for cam in cameras:
        H, W = image_resolution
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
        }
    
    # 增量模式：加载已有数据集；全量模式：创建新数据集
    if incremental and output_path.exists():
        print(f"[增量模式] 加载已有数据集: {output_path}")
        dataset = LeRobotDataset(str(output_path))
        # 重置数据集状态以追加新episode
        dataset._current_episode_frames = []
        dataset._current_task = None
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_path, 
            fps=10,
            robot_type=robot_type,
            features=features,
            use_videos=True, 
        )
    return dataset

def get_episode_number(episode_dir):
    """提取episode名称中的数字（用于数值排序）"""
    try:
        # 从 "episode_0" 中提取 "0" 并转为整数
        return int(episode_dir.name.split("_")[-1])
    except (ValueError, IndexError):
        # 异常情况返回极大值，排到最后
        return float('inf')

def load_episode_data(episode_dir):
    """加载单个episode的原始数据（修复图像统计逻辑）"""
    print(f"\n📂 当前处理的episode绝对路径: {episode_dir.absolute()}")
    data_file = episode_dir / "data.jsonl"
    
    if not data_file.exists():
        print(f"警告: {episode_dir.name} 缺少data.jsonl文件，跳过")
        return None, None, None, None
    
    # 尝试加载 jsonl
    try:
        with open(data_file, "r") as f:
            lines = [json.loads(line) for line in f]
    except Exception as e:
        print(f"❌ 读取jsonl文件失败: {e}")
        return None, None, None, None

    print(f"📝 {episode_dir.name} 原始data.jsonl行数: {len(lines)}")
    
    cartesian_abs_list = [] 
    gripper_list = []
    instructions = []

    # 逐行处理并捕获错误
    for i, line in enumerate(lines):
        try:
            gripper_state = line.get("gripper_state", 0.0)
            
            # 检查字段是否存在
            if "cartesian_pos" not in line:
                print(f"⚠️ 第 {i} 行缺少 'cartesian_pos' 字段，跳过。内容: {line.keys()}")
                continue
                
            raw_cart = line["cartesian_pos"]
            
            # 检查数据长度
            if len(raw_cart) < 6:
                print(f"⚠️ 第 {i} 行 'cartesian_pos' 长度不足: {raw_cart}")
                continue

            cart_pos = np.array(raw_cart, dtype=np.float32)
            cart_pos[:3] /= 1000.0  # mm -> m
            
            # 只有数据有效才添加到列表
            cartesian_abs_list.append(cart_pos)
            gripper_list.append(gripper_state)
            instructions.append(line.get("instruction", ""))
            
        except Exception as e:
            print(f"❌ 处理第 {i} 行数据时出错: {e}")
            continue

    # 检查是否解析到了数据
    if len(cartesian_abs_list) == 0:
        print(f"❌ 警告: {episode_dir.name} 解析后有效数据为0！请检查jsonl格式。")
        return None, None, None, None

    # 转为 numpy 数组
    cartesian_abs_arr = np.array(cartesian_abs_list, dtype=np.float32)
    gripper_arr = np.array(gripper_list, dtype=np.float32).reshape(-1, 1)
    
    print(f"✅ 成功解析状态数据: {cartesian_abs_arr.shape}")

    # ========== 图像加载逻辑 ==========
    import re
    images = {}
    cameras = ["cam_left_wrist", "cam_right_wrist"]
    
    for cam in cameras:
        cam_dir = episode_dir / "images" / cam
        if not cam_dir.exists():
            print(f"警告: {episode_dir.name} 缺少{cam}图像目录")
            images[cam] = np.array([])
            continue
            
        import os
        os.sync()
        
        img_files = []
        for file in cam_dir.iterdir():
            if file.is_file() and file.suffix.lower() == ".jpg":
                img_files.append(file)
        
        def extract_number(file_path):
            nums = re.findall(r'\d+', file_path.name)
            return int(nums[0]) if nums else float('inf')
        
        img_files.sort(key=extract_number)
        
        # 加载图像
        cam_imgs = []
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cam_imgs.append(img)
        
        # 长度对齐检查
        if len(cam_imgs) != len(cartesian_abs_arr):
            print(f"⚠️ {cam} 图像数({len(cam_imgs)}) 与 状态数({len(cartesian_abs_arr)}) 不一致")
            
        images[cam] = np.array(cam_imgs)
    
    return cartesian_abs_arr, gripper_arr, images, instructions

def convert_single_episode(episode_dir, dataset, sample_data_list):
    """转换单个episode并追加到数据集"""
    # 加载数据
    # st_abs, st_delta, images, instructions = load_episode_data(episode_dir)
    cart_abs, gripper, images, instructions = load_episode_data(episode_dir)
    if cart_abs is None:
        return False
    
    # 计算最小长度（对齐所有数据）
    min_len = len(cart_abs)
    for cam, imgs in images.items():
        min_len = min(min_len, len(imgs)) if len(imgs) > 0 else min_len
    
    if min_len == 0:
        print(f"警告: {episode_dir.name} 无有效数据，跳过")
        return False
    
    # 统一裁切到最小长度
    cart_abs = cart_abs[:min_len]
    gripper = gripper[:min_len]
    instructions = instructions[:min_len]
    for cam in images:
        if len(images[cam]) > 0:
            images[cam] = images[cam][:min_len]
    
    ## A. 计算 State: Relative to Start (当前绝对 - 第一帧绝对)
    start_pose = cart_abs[0] # 记录起始点
    # 广播减法：每一帧都减去起始点
    state_pos_rel = cart_abs - start_pose 
    # 拼接夹爪：[rel_x, rel_y, rel_z, rel_r, rel_p, rel_y, gripper]
    state_final_np = np.hstack([state_pos_rel, gripper])
    # B. 计算 Action: Delta (下一帧绝对 - 当前帧绝对)
    action_pos_delta = np.zeros_like(cart_abs)
    # Delta[t] = Pose[t+1] - Pose[t]
    action_pos_delta[:-1] = cart_abs[1:] - cart_abs[:-1]
    # 最后一帧增量设为0（或者复制上一帧）
    action_pos_delta[-1] = action_pos_delta[-2] if min_len > 1 else 0.0
    # Action中的夹爪：预测下一步的绝对状态
    action_gripper = np.zeros_like(gripper)
    action_gripper[:-1] = gripper[1:]
    action_gripper[-1] = gripper[-1]
    # 拼接 Action
    action_final_np = np.hstack([action_pos_delta, action_gripper])
    # 打印信息
    print(f"\n=== 处理 Episode: {episode_dir.name} ===")
    print(f"模式: Cartesian Relative State & Delta Action")
    print(f"State维度: {state_final_np.shape}, Action维度: {action_final_np.shape}")
    # 保存示例数据 (用于调试查看)
    if len(sample_data_list) < 5: # 只存前几个防止文件过大
        sample_frame = {
            "episode": episode_dir.name,
            "frame_idx": 0,
            "state_sample": state_final_np[0].tolist(), # 应该是接近0
            "action_sample": action_final_np[0].tolist(),
            "instruction": instructions[0]
        }
        sample_data_list.append(sample_frame)

    
    
    
    
    
    # 写入每一帧数据
    for i in range(min_len):
        frame = {
            # 绝对值
            "observation.state": torch.from_numpy(state_final_np[i]),
            "action": torch.from_numpy(action_final_np[i]),
            # 语言指令
            "language_instruction": instructions[i],
        }
        
        # 处理每个摄像头的图像（兼容缺失情况）
        cameras = ["cam_left_wrist", "cam_right_wrist"]
        for cam in cameras:
            cam_imgs = images.get(cam, np.array([]))
            # 图像缺失/索引越界：用空图像填充
            if len(cam_imgs) == 0 or i >= len(cam_imgs):
                img = create_empty_image(image_resolution)
            else:
                img = cam_imgs[i]
                # 缩放+补边到目标分辨率
                if img.shape[:2] != image_resolution:
                    img = image_tools.resize_with_pad(img, *image_resolution)
                    img = np.array(img)
            frame[f"observation.images.{cam}"] = img
        
        dataset.add_frame(frame, task=instructions[i])
    
    # 保存当前episode（增量写入）
    dataset.save_episode()
    print(f"✅ {episode_dir.name} 转换完成并追加到数据集")
    return True

def load_existing_sample_data(sample_file):
    """加载已有的示例数据（用于增量追加）"""
    if not sample_file.exists():
        return []
    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        print(f"警告: 读取{sample_file}失败，重新创建")
        return []

def incremental_convert(raw_dir, repo_id, output_dir):
    """增量转换主函数：持续扫描新增episode并追加"""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_path = output_dir / repo_id
    sample_file = output_path / "sample_data.json"
    
    # 初始化：创建/加载数据集
    dataset = create_dataset(repo_id, output_dir, incremental=True)
    
    # 加载已转换的episode列表
    converted_episodes = set()
    if output_path.exists():
        # 从示例数据中读取已转换的episode
        sample_data = load_existing_sample_data(sample_file)
        converted_episodes = {item["episode"] for item in sample_data}
        print(f"📌 已转换的episode数量: {len(converted_episodes)}")
    
    # 加载示例数据列表
    sample_data_list = load_existing_sample_data(sample_file)
    
    print(f"\n🚀 启动增量转换模式，每隔{SCAN_INTERVAL}秒扫描一次新增episode")
    print(f"原始数据目录: {raw_dir}")
    print(f"输出数据集目录: {output_path}")
    print("按 Ctrl+C 停止转换\n")
    
    try:
         while True:
            # 扫描所有episode目录
            all_episode_dirs = [
                d for d in raw_dir.iterdir() 
                if d.is_dir() and d.name.startswith("episode_")
            ]
            
            # ========== 新增：过滤掉未完成写入的episode ==========
            completed_episodes = []
            for d in all_episode_dirs:
                # 1. 检查data.jsonl是否存在且写入完成（最后修改时间超过2秒）
                data_file = d / "data.jsonl"
                if not data_file.exists():
                    continue
                data_mtime = data_file.stat().st_mtime
                if time.time() - data_mtime < 2:  # 延迟2秒，确保写入完成
                    continue
                
                # 2. 检查图像目录是否存在且写入完成
                # cam_dir = d / "images" / "cam_high"
                cam_dir = d / "images" / "cam_left_wrist"
                if not cam_dir.exists():
                    continue
                # 取最后一张图像的修改时间（验证写入完成）
                img_files = list(cam_dir.glob("*.jpg"))
                if len(img_files) == 0:
                    continue
                last_img_mtime = max(f.stat().st_mtime for f in img_files)
                if time.time() - last_img_mtime < 2:
                    continue
                
                completed_episodes.append(d)
            
            # ========== 按数值排序 ==========
            completed_episodes.sort(key=get_episode_number)
            
            # 筛选未转换的episode
            new_episodes = [
                d for d in completed_episodes 
                if d.name not in converted_episodes
            ]
            
            # 处理新增episode
            if new_episodes:
                # 打印新增episode列表（按顺序）
                new_ep_names = [d.name for d in new_episodes]
                print(f"\n🔍 发现{len(new_episodes)}个新增episode: {new_ep_names}")
                for ep_dir in tqdm.tqdm(new_episodes, desc="转换中"):
                    # 转换并追加
                    success = convert_single_episode(ep_dir, dataset, sample_data_list)
                    if success:
                        converted_episodes.add(ep_dir.name)
                        # 实时保存示例数据
                        with open(sample_file, "w", encoding="utf-8") as f:
                            json.dump(sample_data_list, f, indent=4, ensure_ascii=False)
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 暂无新增episode，等待中...", end="\r")
            
            # 等待下一次扫描
            time.sleep(SCAN_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 用户终止程序")
    finally:
        # 最终保存示例数据
        print(f"\n💾 保存最终示例数据到: {sample_file}")
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_data_list, f, indent=4, ensure_ascii=False)
        # 验证最终数据集
        print("\n=== 验证最终数据集 ===")
        final_dataset = LeRobotDataset(str(output_path))
        print(f"数据集总帧数: {len(final_dataset)}")
        print(f"已转换episode数量: {len(converted_episodes)}")

def verify_converted_data(output_dir, repo_id):
    """验证转换后的数据集"""
    print("\n=== 验证数据集 ===")
    dataset_path = Path(output_dir) / repo_id
    if not dataset_path.exists():
        print(f"❌ 数据集不存在: {dataset_path}")
        return
    
    dataset = LeRobotDataset(str(dataset_path))
    print(f"✅ 数据集加载成功，总帧数: {len(dataset)}")
    
    if len(dataset) > 0:
        sample_data = dataset[0]
        print(f"\n--- 第0帧数据样例 ---")
        print(f"observation.state: {np.round(sample_data['observation.state'], 4)}")
        print(f"action: {np.round(sample_data['action'], 4)}")
        print(f"language_instruction: {sample_data['language_instruction']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, 
                        default="/home/openpi/data/data_raw/exp19_data_auto_queue_PutAndRecord_0113/raw", 
                        help="原始数据目录")
    # parser.add_argument("--raw-dir", type=str, 
    #                     default="/home/openpi/data/data_raw/test/raw", 
    #                     help="原始数据目录")
    parser.add_argument("--repo-id", type=str, 
                        default="xarm_autoPut_pi05_dataset", 
                        help="数据集名称")
    parser.add_argument("--output-dir", type=str, 
                        default="/home/openpi/data/data_converted/exp19_lerobot_autoPut_data_0113night_224_224", 
                        help="输出目录")
    parser.add_argument("--scan-interval", type=int, default=5, 
                        help="扫描新增episode的间隔（秒）")
    args = parser.parse_args()
    
    # 覆盖全局扫描间隔
    SCAN_INTERVAL = args.scan_interval
    
    # 启动增量转换
    incremental_convert(args.raw_dir, args.repo_id, args.output_dir)
    
    # 验证最终结果
    verify_converted_data(args.output_dir, args.repo_id)
