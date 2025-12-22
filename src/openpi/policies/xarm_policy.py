import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    """
    辅助函数：将 LeRobot 加载的图像 (可能是 Float CHW) 转换为 OpenPi 需要的 Uint8 HWC 格式。
    """
    image = np.asarray(image)
    # 如果是浮点数 (0-1)，转为 0-255 整数
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # 如果是 CHW (Channel first)，转为 HWC (Height first)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class XArmInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 1. 提取状态 (7维: 6关节 + 1夹爪)
        state = data["state"]
        
        # 2. 处理图像
        # 对应你在 RepackTransform 中定义的键名
        img_high = _parse_image(data["cam_high"])
        img_left = _parse_image(data["cam_left_wrist"])
        img_right = _parse_image(data["cam_right_wrist"])

        # 3. 构建输入字典
        inputs = {
            "state": state,
            "image": {
                # 映射逻辑：
                # cam_high -> base_0_rgb (第三人称主视角)
                "base_0_rgb": img_high,
                # cam_left_wrist -> left_wrist_0_rgb (左腕)
                "left_wrist_0_rgb": img_left,
                # cam_right_wrist -> right_wrist_0_rgb (右腕)
                "right_wrist_0_rgb": img_right,
            },
            # 告诉模型这三个摄像头的数据都是有效的
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # 4. 处理动作 (如果存在)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # 5. 处理文本提示
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XArmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # xArm 是 7 自由度 (6关节 + 1夹爪)，取前7维
        return {"actions": np.asarray(data["actions"][:, :7])}