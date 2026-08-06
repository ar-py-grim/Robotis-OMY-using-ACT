#!lerobot_env/bin/python3

import numpy as np
import json
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.act_y_env import SimpleEnv
from config import REPO_NAME, ROOT, xml_path

dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# 'single' -> replay one specific episode on loop (set EPISODE_INDEX below)
# 'all'    -> cycle through every episode in the dataset, one after another
VISUALIZE_MODE = 'single'  # 'single' or 'all'

NUM_EPISODES = dataset.meta.total_episodes

print(f"[data_visualize] Dataset '{dataset.repo_id}' has {NUM_EPISODES} episode(s) "
      f"(valid index range: 0 to {NUM_EPISODES-1})")

# only used when VISUALIZE_MODE == 'single'
EPISODE_INDEX = int(input('Enter valid episode index: '))

if VISUALIZE_MODE == 'single':
    if not (0 <= EPISODE_INDEX < NUM_EPISODES):
        raise ValueError(f"valid values are 0 to {NUM_EPISODES-1}.")


class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sampler for a single episode
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        episode = dataset.meta.episodes[episode_index]
        from_idx = episode["dataset_from_index"]
        to_idx = episode["dataset_to_index"]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def make_dataloader(episode_index):
    """Build a fresh sampler + dataloader for the given episode index."""
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=1,
    batch_size=1, sampler=episode_sampler,
    )
    return episode_sampler, dataloader


# Pick the starting episode based on the mode
if VISUALIZE_MODE == 'single':
    current_episode_index = EPISODE_INDEX

elif VISUALIZE_MODE == 'all':
    current_episode_index = 0
else:
    raise ValueError(f"Unknown VISUALIZE_MODE: {VISUALIZE_MODE!r}. Use 'single' or 'all'.")

episode_sampler, dataloader = make_dataloader(current_episode_index)

PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

print(f"[data_visualize] mode={VISUALIZE_MODE} | episode={current_episode_index}"
      f"{f'/{NUM_EPISODES-1}' if VISUALIZE_MODE == 'all' else ''}")

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Get the action from dataset
        data = next(iter_dataloader)
        if step == 0:
            # Reset the object pose based on the dataset
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:])
        # Get the action from dataset
        action = data['action'].numpy()
        obs = PnPEnv.step(action[0])

        # Visualize the image from dataset to rgb_overlay
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3x256x256 -> 256x256x3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        PnPEnv.render()
        step+= 1

        if step == len(episode_sampler):
            # if 'all', loop to next episode
            # if 'single', loop same episode
            if VISUALIZE_MODE == 'all':
                current_episode_index = (current_episode_index+1)%NUM_EPISODES

            episode_sampler, dataloader = make_dataloader(current_episode_index)
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0

            print(f"[data_visualize] mode={VISUALIZE_MODE} | episode={current_episode_index}"
                  f"{f'/{NUM_EPISODES-1}' if VISUALIZE_MODE == 'all' else ''}")

PnPEnv.env.close_viewer()


# Save Stats.json for other versions
def _to_json_safe(obj):
    """Recursively convert numpy arrays into plain Python types for json.dump."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    return obj.tolist() if isinstance(obj, np.ndarray) else obj

stats = dataset.meta.stats
PATH = dataset.root/'meta'/'stats.json'
stats_safe = _to_json_safe(stats)

with open(PATH, 'w') as f:
    json.dump(stats_safe, f, indent=4)