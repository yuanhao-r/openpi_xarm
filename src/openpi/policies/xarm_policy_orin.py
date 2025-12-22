import dataclasses
from typing import ClassVar
import einops
import numpy as np
from openpi import transforms

@dataclasses.dataclass(frozen=True)
class XArmInputs(transforms.DataTransformFn):
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        images = {}
        image_masks = {}
        
        def process_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        cam_map = {
            "base_0_rgb": "cam_high",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        
        for model_key, dataset_key in cam_map.items():
            if dataset_key in in_images:
                images[model_key] = process_image(in_images[dataset_key])
                image_masks[model_key] = np.True_
            else:
                images[model_key] = np.zeros((224, 224, 3), dtype=np.uint8)
                image_masks[model_key] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        return inputs

@dataclasses.dataclass(frozen=True)
class XArmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])}