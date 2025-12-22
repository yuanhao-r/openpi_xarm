from dataclasses import dataclass
import os

from openpi.training import config as train_config
from openpi.models import model as model_config
from openpi.shared import download

@dataclass
class XArmInputs(model_config.ModelInputs):
    # 7-dim state (6 joints + 1 padding/gripper)
    state_dim: int = 7
    # 3 cameras
    num_cameras: int = 3

@dataclass
class XArmOutputs(model_config.ModelOutputs):
    # 7-dim action (6 joints + 1 padding/gripper)
    action_dim: int = 7

def get_config(training_steps: int = 10000) -> train_config.TrainConfig:
    return train_config.TrainConfig(
        # Data config
        data=train_config.DataConfig(
            # Update this with your actual dataset repo ID
            repo_id="xarm_convert", 
            # Map dataset features to model inputs
            image_key_mapping={
                "cam_high": "base_0_rgb",
                "cam_left_wrist": "left_wrist_0_rgb",
                "cam_right_wrist": "right_wrist_0_rgb",
            },
            state_key="observation.state",
            action_key="action",
        ),
        
        # Model config
        model=model_config.Pi0Config(
            inputs=XArmInputs(),
            outputs=XArmOutputs(),
        ),
        
        # Training parameters
        batch_size=32, # Adjust based on VRAM
        num_train_steps=training_steps,
        save_interval=1000,
        eval_interval=1000,
        learning_rate=1e-4,
    )
