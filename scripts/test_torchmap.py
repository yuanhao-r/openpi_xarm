import torch
# PyTorch 2.7.1 原生支持 _pytree 模块，直接导入即可
from torch.utils._pytree import tree_map

def main():
    # ===================== 1. 配置设备（自动检测GPU/CPU） =====================
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # 优先用第0块GPU
        print(f"✅ 检测到GPU，使用设备：{device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  未检测到GPU，使用CPU设备：{device}")

    # ===================== 2. 模拟真实场景的 observation 嵌套结构 =====================
    # 完全贴合 OpenPI/LeRobot 机械臂的 observation 格式（包含字典、子字典、tensor、非tensor元素）
    observation = {
        # 机械臂关节状态（7维）
        "observation.state": torch.randn(7, dtype=torch.float32),
        # 机械臂动作（7维）
        "action": torch.randn(7, dtype=torch.float32),
        # 多摄像头图像（3通道，224x224）
        "observation.images": {
            "cam_high": torch.randn(3, 224, 224, dtype=torch.float32),
            "cam_left_wrist": torch.randn(3, 224, 224, dtype=torch.float32),
            "cam_right_wrist": torch.randn(3, 224, 224, dtype=torch.float32)
        },
        # 语言指令（非tensor元素，需跳过）
        "language_instruction": "move the object to the target position",
        # 帧索引（列表+整数，非tensor）
        "frame_index": [123, 456, 789],
        # 混合类型元组（tensor + 字符串）
        "meta_info": (torch.tensor([0.1, 0.2]), "episode_506")
    }

    # ===================== 3. 核心：迁移所有tensor到目标设备（替代JAX代码） =====================
    print("\n🔄 开始迁移observation到目标设备...")
    # tree_map 完全复刻 jax.tree.map 的功能，遍历所有嵌套结构的叶子节点
    observation = tree_map(
        # 仅对tensor执行to(device)，非tensor元素直接返回（避免报错）
        lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
        observation
    )

    # ===================== 4. 验证迁移结果 =====================
    print("\n✅ 验证迁移结果：")
    # 定义递归验证函数（遍历所有嵌套结构，检查tensor设备）
    def validate_device(obj, parent_key=""):
        if isinstance(obj, torch.Tensor):
            assert obj.device == device, f"❌ {parent_key} 设备错误！当前：{obj.device}，预期：{device}"
            print(f"✅ {parent_key}: {obj.shape} → 设备：{obj.device}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                validate_device(v, f"{parent_key}.{k}" if parent_key else k)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            for idx, item in enumerate(obj):
                validate_device(item, f"{parent_key}[{idx}]" if parent_key else f"[{idx}]")
        else:
            # 非tensor元素，打印类型即可
            print(f"ℹ️  {parent_key}: 非Tensor类型 → {type(obj)}")

    # 执行验证
    validate_device(observation)

    # ===================== 5. 测试总结 =====================
    print("\n🎉 所有Tensor均已成功迁移到目标设备！测试通过！")

if __name__ == "__main__":
    # 设置PyTorch CUDA日志级别（避免冗余输出）
    # torch.cuda.set_logging_level(torch.logging.Level.ERROR)
    main()
