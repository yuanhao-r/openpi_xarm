# import os
# import sys
# import json
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # OpenPI 路径设置 (根据你的环境)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("/home/openpi/src")
# # 如果 openpi-client 位置不同，请自行调整
# openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
# sys.path.append(os.path.abspath(openpi_client_path))

# from openpi.training import config as _config
# from openpi.policies import policy_config
# from openpi_client import image_tools

# # ================= 配置区域 =================
# # 1. 模型配置
# CONFIG_NAME = "pi05_xarm_1212_night"  # 你的训练配置名
# CHECKPOINT_DIR = "/home/openpi/checkpoints/exp19/4000" # 你的 checkpoint 路径

# # 2. 数据源配置
# # JSON 文件路径
# JSON_DATA_PATH = "/home/openpi/record_and_transform/episode_0_numeric_data.json"

# # 图片根目录 (去掉最后的 episode_000000，代码里会自动拼)
# IMG_ROOT_LEFT = "/home/openpi/data/data_converted/exp19_lerobot_autoPut_data_0113night_224_224/xarm_autoPut_pi05_dataset/images/observation.images.cam_left_wrist"
# IMG_ROOT_RIGHT = "/home/openpi/data/data_converted/exp19_lerobot_autoPut_data_0113night_224_224/xarm_autoPut_pi05_dataset/images/observation.images.cam_right_wrist"

# EPISODE_FOLDER = "episode_000000" # 图片所在的子文件夹名

# # 3. 图像预处理参数 (必须和训练时一致)
# RESIZE_W, RESIZE_H = 224, 224
# # 你的裁剪参数
# CROP_CONFIGS = {
#     "cam_left_wrist": (118, 60, 357, 420),
#     "cam_right_wrist": (136, 57, 349, 412)
# }
# # ===========================================

# def load_json_data(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     # 按 frame_index 排序，防止乱序
#     data.sort(key=lambda x: x['frame_index'])
#     return data

# def load_image(root_dir, episode_folder, frame_idx, crop_params=None):
#     # 假设文件名格式为 frame_000000.png，根据实际情况修改
#     # LeRobot/OpenPi 通常转换后的图片格式是 frame_{:06d}.png
#     img_name = f"frame_{frame_idx:06d}.png" 
#     path = os.path.join(root_dir, episode_folder, img_name)
    
#     if not os.path.exists(path):
#         # 尝试 jpg
#         path = path.replace(".png", ".jpg")
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Image not found: {path}")

#     img = cv2.imread(path)
#     if img is None:
#         raise ValueError(f"Failed to read image: {path}")
    
#     # BGR -> RGB
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Crop
#     if crop_params:
#         x, y, w, h = crop_params
#         img = img[y:y+h, x:x+w]
        
#     # Resize (OpenPi 通常需要 224x224)
#     img = cv2.resize(img, (RESIZE_W, RESIZE_H))
    
#     return image_tools.convert_to_uint8(img)

# def main():
#     # 1. 加载模型
#     print(f"Loading Policy: {CONFIG_NAME} from {CHECKPOINT_DIR}...")
#     config = _config.get_config(CONFIG_NAME)
#     policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
#     print("Policy loaded successfully.")

#     # 2. 加载数据
#     print(f"Loading data from {JSON_DATA_PATH}...")
#     json_data = load_json_data(JSON_DATA_PATH)
#     print(f"Loaded {len(json_data)} frames.")

#     # 准备存储结果
#     gt_actions = []     # Ground Truth (JSON里的 action)
#     pred_actions = []   # Model Prediction (模型推理出的 action)
#     pred_actions_integrated = [] # 预测积分后的位置
    
#     # 初始位置 (用于积分)
#     # 假设 JSON 里的 observation.state 是相对于起始点的位姿，
#     # 那么第一帧的绝对位置可以认为是 0 (或者我们需要把 state 加回去)
#     # 为了简化对比，我们只对比“增量”和“相对位姿的变化趋势”
    
#     print("\nStarting Offline Inference Loop...")
    
#     for i, frame in enumerate(json_data):
#         frame_idx = frame['frame_index']
#         timestamp = frame['timestamp']
        
#         # --- A. 构造 Observation ---
#         # 1. 图片
#         try:
#             img_left = load_image(IMG_ROOT_LEFT, EPISODE_FOLDER, frame_idx, CROP_CONFIGS["cam_left_wrist"])
#             img_right = load_image(IMG_ROOT_RIGHT, EPISODE_FOLDER, frame_idx, CROP_CONFIGS["cam_right_wrist"])
#         except Exception as e:
#             print(f"Error loading images for frame {frame_idx}: {e}")
#             break

#         # 2. State
#         # 注意：这里直接使用 JSON 里的 state。
#         # 如果训练时用了 transform，这里要确保喂进去的是 raw 数据，policy 内部会处理 transform
#         input_state = np.array(frame['observation.state'], dtype=np.float32)
        
#         # 3. Prompt
#         prompt = frame['language_instruction']

#         # --- B. 推理 ---
#         # 构造输入字典
#         obs_dict = {
#             "cam_left_wrist": img_left,
#             "cam_right_wrist": img_right,
#             "state": input_state,
#             "prompt": prompt
#         }
#         # 补充 cam_high 占位符 (如果模型需要)
#         obs_dict["cam_high"] = np.zeros((224, 224, 3), dtype=np.uint8)

#         result = policy.infer(obs_dict)
        
#         # --- C. 提取数据 ---
#         # 1. Ground Truth Action
#         gt_action = np.array(frame['action'], dtype=np.float32)
#         gt_actions.append(gt_action)
        
#         # 2. Predicted Action
#         # 模型输出的是 chunk [T, 7]。我们取第一步 (action_chunk[0]) 作为当前帧的预测。
#         # 也可以取 action_chunk[k] 对应未来的预测，但为了对比 JSON 的 action (它是单步的)，取 [0] 最直观。
#         pred_chunk = np.array(result['actions'])
#         pred_action_step0 = pred_chunk[0]
#         pred_actions.append(pred_action_step0)
        
#         print(f"Frame {frame_idx}: GT[0]={gt_action[0]:.6f}, Pred[0]={pred_action_step0[0]:.6f}")

#     # 转换 list 为 numpy array
#     gt_actions = np.array(gt_actions)       # [N, 7]
#     pred_actions = np.array(pred_actions)   # [N, 7]
    
#     # ================= 绘图分析 =================
#     print("\nPlotting results...")
    
#     # 创建图表：3行 (X, Y, Z)，2列 (Raw Action, Cumulative Sum)
#     fig, axs = plt.subplots(3, 2, figsize=(15, 12))
#     dims = ['X', 'Y', 'Z']
    
#     for i in range(3): # 只画 X, Y, Z
#         # --- 左侧：原始输出对比 ---
#         # 如果你的原始 JSON action 是 Delta (速度)，这里展示的就是速度对比
#         ax_raw = axs[i, 0]
#         ax_raw.plot(gt_actions[:, i], label='GT Action (Data)', color='black', linewidth=2, alpha=0.6)
#         ax_raw.plot(pred_actions[:, i], label='Model Pred', color='red', linestyle='--')
#         ax_raw.set_title(f"{dims[i]} - Raw Model Output")
#         ax_raw.legend()
#         ax_raw.grid(True)
        
#         # --- 右侧：积分轨迹对比 (Cumulative Sum) ---
#         # 如果 Action 是 Delta，积分后就是位置 (Position Relative to Start)
#         # 这样我们可以验证模型是不是真的学会了轨迹
#         ax_cum = axs[i, 1]
        
#         gt_cumsum = np.cumsum(gt_actions[:, i])
#         pred_cumsum = np.cumsum(pred_actions[:, i])
        
#         ax_cum.plot(gt_cumsum, label='GT Integrated (Pos)', color='blue', linewidth=2, alpha=0.6)
#         ax_cum.plot(pred_cumsum, label='Pred Integrated (Pos)', color='orange', linestyle='--')
        
#         # 同时画出 input state，看看 state 和积分轨迹是否吻合
#         # JSON 里的 state 是相对于 Start 的位置
#         # 如果 Action 是 Delta，那么 cumsum(Action) 应该大致等于 State (忽略初始偏差)
#         # 注意：这里假设 state 数据存在且对应
#         state_vals = [f['observation.state'][i] for f in json_data[:len(gt_actions)]]
#         ax_cum.plot(state_vals, label='Input State (Ref)', color='green', linestyle=':', alpha=0.5)
        
#         ax_cum.set_title(f"{dims[i]} - Trajectory Reconstruction (CumSum)")
#         ax_cum.legend()
#         ax_cum.grid(True)

#     plt.tight_layout()
#     plt.savefig("offline_test_analysis.png")
#     print(f"Analysis saved to: {os.path.abspath('offline_test_analysis.png')}")

# if __name__ == "__main__":
#     main()


import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# OpenPI 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("/home/openpi/src")
openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
sys.path.append(os.path.abspath(openpi_client_path))

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# ================= 配置区域 =================
CONFIG_NAME = "pi05_xarm_1212_night"
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp20/4000"

# JSON 文件路径
JSON_DATA_PATH = "/home/openpi/record_and_transform/episode_0_numeric_data.json"

# 图片根目录 (请确认路径是否正确)
IMG_ROOT_LEFT = "/home/openpi/data/data_converted/exp20_lerobot_autoPut_data_0114night_224_224/xarm_autoPut_pi05_dataset/images/observation.images.cam_left_wrist"
IMG_ROOT_RIGHT = "/home/openpi/data/data_converted/exp20_lerobot_autoPut_data_0114night_224_224/xarm_autoPut_pi05_dataset/images/observation.images.cam_right_wrist"

EPISODE_FOLDER = "episode_000000"

RESIZE_W, RESIZE_H = 224, 224
CROP_CONFIGS = {
    "cam_left_wrist": (118, 60, 357, 420),
    "cam_right_wrist": (136, 57, 349, 412)
}
# ===========================================

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    data.sort(key=lambda x: x['frame_index'])
    return data

def load_image(root_dir, episode_folder, frame_idx, crop_params=None):
    img_name = f"frame_{frame_idx:06d}.png" 
    path = os.path.join(root_dir, episode_folder, img_name)
    if not os.path.exists(path):
        path = path.replace(".png", ".jpg")
        if not os.path.exists(path):
            # 容错：如果找不到图片，生成纯黑图片防止报错中断
            print(f"Warning: Image not found {path}, using black image.")
            return np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8)

    img = cv2.imread(path)
    if img is None:
        return np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop_params:
        x, y, w, h = crop_params
        img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (RESIZE_W, RESIZE_H))
    return image_tools.convert_to_uint8(img)

def main():
    print(f"Loading Policy: {CONFIG_NAME} from {CHECKPOINT_DIR}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    print(f"Loading data from {JSON_DATA_PATH}...")
    json_data = load_json_data(JSON_DATA_PATH)
    
    gt_actions = []
    pred_actions = []
    all_states = [] # 用于存储每一帧的 State 真值

    print("\nStarting Offline Inference Loop...")
    for i, frame in enumerate(json_data):
        frame_idx = frame['frame_index']
        
        # 1. 准备 State (真值)
        input_state = np.array(frame['observation.state'], dtype=np.float32)
        all_states.append(input_state)
        
        # 2. 准备图片
        img_left = load_image(IMG_ROOT_LEFT, EPISODE_FOLDER, frame_idx, CROP_CONFIGS["cam_left_wrist"])
        img_right = load_image(IMG_ROOT_RIGHT, EPISODE_FOLDER, frame_idx, CROP_CONFIGS["cam_right_wrist"])
        
        # 3. 推理
        obs_dict = {
            "cam_left_wrist": img_left,
            "cam_right_wrist": img_right,
            "state": input_state,
            "prompt": frame['language_instruction'],
            "cam_high": np.zeros((224, 224, 3), dtype=np.uint8)
        }
        result = policy.infer(obs_dict)
        
        # 4. 记录数据
        gt_action = np.array(frame['action'], dtype=np.float32)
        gt_actions.append(gt_action)
        
        pred_chunk = np.array(result['actions'])
        pred_actions.append(pred_chunk[0]) # 取第一步预测
        
        if i % 10 == 0:
            print(f"Processed frame {frame_idx}/{len(json_data)}")

    # 转 Numpy
    gt_actions = np.array(gt_actions)       # [N, 7]
    pred_actions = np.array(pred_actions)   # [N, 7]
    all_states = np.array(all_states)       # [N, 7]
    
    # ================= 绘图分析 (3列布局) =================
    print("\nPlotting results (3 Columns)...")
    
    fig, axs = plt.subplots(3, 3, figsize=(20, 12))
    dims = ['X', 'Y', 'Z']
    
    for i in range(3): # Loop dimensions
        # --- Col 1: Raw Action (速度) ---
        ax_raw = axs[i, 0]
        ax_raw.plot(gt_actions[:, i], label='GT Action (Velocity)', color='black', linewidth=2, alpha=0.6)
        ax_raw.plot(pred_actions[:, i], label='Model Pred', color='red', linestyle='--')
        ax_raw.set_title(f"{dims[i]} - Raw Velocity")
        ax_raw.legend(fontsize='small')
        ax_raw.grid(True)
        
        # --- 计算积分 ---
        # 假设 Action 是 State 的差分，那么 cumsum(Action) 应该还原 State
        # 初始 State 通常是 0 (因为是 Rel_Start)
        gt_cumsum = np.cumsum(gt_actions[:, i])
        pred_cumsum = np.cumsum(pred_actions[:, i])

        # --- Col 2: Integrated Path (积分轨迹) ---
        ax_cum = axs[i, 1]
        ax_cum.plot(gt_cumsum, label='GT Integration', color='blue', linewidth=2, alpha=0.6)
        ax_cum.plot(pred_cumsum, label='Pred Integration', color='orange', linestyle='--')
        ax_cum.set_title(f"{dims[i]} - Integrated Path")
        ax_cum.legend(fontsize='small')
        ax_cum.grid(True)

        # --- Col 3: State vs Integration (真值校验) ---
        # 这里的关键是看 GT CumSum 是否等于 GT State
        # 如果重合，说明你的数据逻辑是自洽的
        # 如果 Pred CumSum 偏离了 GT State，那就是模型的问题
        ax_state = axs[i, 2]
        ax_state.plot(all_states[:, i], label='GT State (Recorded)', color='green', linewidth=3, alpha=0.4)
        ax_state.plot(gt_cumsum, label='GT Action CumSum', color='blue', linestyle=':', linewidth=2)
        ax_state.plot(pred_cumsum, label='Pred Path (Model)', color='red', linestyle='--')
        
        ax_state.set_title(f"{dims[i]} - State vs Pred Path")
        ax_state.legend(fontsize='small')
        ax_state.grid(True)

    plt.tight_layout()
    plt.savefig("offline_test_analysis_3col.png")
    print(f"Analysis saved to: {os.path.abspath('offline_test_analysis_3col.png')}")

if __name__ == "__main__":
    main()