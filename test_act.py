#!lerobot_env/bin/python3

import numpy as np
import torch
import cv2
import torchvision
from mujoco_env.act_y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import dataset_to_policy_features
from config import SEED, REPO_NAME, ROOT, TASK_NAME, xml_path, CKPT_PATH

device = 'cuda'

dataset_metadata = LeRobotDatasetMetadata(REPO_NAME, root=ROOT)
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")
# Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
# we'll just use the defaults and so no arguments other than input/output features need to be passed.
# Temporal ensemble to make smoother trajectory predictions
cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size= 10, n_action_steps=1, temporal_ensemble_coeff = 0.9)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
# We can now instantiate our policy with this config and the dataset stats.
policy = ACTPolicy.from_pretrained(CKPT_PATH, config = cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

# Load environment
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

step = 0
PnPEnv.reset(seed=SEED)
policy.reset()
policy.eval()
save_image = True
img_transform = torchvision.transforms.ToTensor()
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Check if the task is completed
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # Reset the environment and action queue
            policy.reset()
            PnPEnv.reset(seed=SEED)
            step = 0
            save_image = False
        # Get the current state of the environment
        state = PnPEnv.get_ee_pose()
        # Get the current image from the environment
        image, wirst_image = PnPEnv.grab_image()
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        image = img_transform(image)
        wrist_image = cv2.resize(wirst_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        wrist_image = img_transform(wrist_image)
        data = {
            'observation.state': torch.tensor(np.array([state]), dtype=torch.float32).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [TASK_NAME],
            'timestamp': torch.tensor([step/20]).to(device)
        }
        # Select an action
        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()
        # print(f"Step: {step}, Action: {action}")
        # Take a step in the environment
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step+= 1
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break