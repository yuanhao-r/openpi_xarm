import numpy as np
import os, sys
# OpenPI 依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
openpi_client_path = os.path.join(current_dir, "../../packages/openpi-client/src")
sys.path.append("/home/openpi/src")
sys.path.append(os.path.abspath(openpi_client_path))
from openpi.training import config as _config
from openpi.policies import policy_config

# 配置
CONFIG_NAME = "pi05_xarm_1212_night"
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp19/4000" # 你的模型路径

def test_model_numerics():
    print(f"Loading: {CHECKPOINT_DIR}")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    # 构造一个“完美的”输入：全部为 0
    # 理论上，如果处于起始点，Delta 应该是 0 或非常小
    dummy_state = np.zeros(7, dtype=np.float32)
    # 随便造个黑图，只要格式对就行
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    print("\n>>> Testing Inference with ZERO inputs...")
    result = policy.infer({
        "cam_left_wrist": dummy_img,
        "cam_right_wrist": dummy_img,
        "state": dummy_state,
        "prompt": "pick up the industrial components"
    })
    
    action = np.array(result["actions"])[0]
    print(f"\n[Model Output] First Step Action: {action}")
    
    # 检查量级
    max_val = np.max(np.abs(action[:3]))
    print(f"[Analysis] XYZ Max Delta: {max_val:.6f}")
    
    if max_val > 0.05: # 如果一步大于 5cm
        print("❌ 异常！对于静止状态，模型输出了巨大的位移。")
        print("可能性 1: 数据集单位是 mm，但模型被当成 m 训练。")
        print("可能性 2: Normalization 统计数据出错（方差过小）。")
    else:
        print("✅ 数值量级正常 (毫米级)。")

if __name__ == "__main__":
    test_model_numerics()