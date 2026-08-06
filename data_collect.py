#!lerobot_env/bin/python3

import numpy as np
import os
import shutil
import cv2
from mujoco_env.act_y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from config import SEED, REPO_NAME, ROOT, TASK_NAME, xml_path

NUM_DEMO = 50

# Define the environment
PnPEnv = SimpleEnv(xml_path, seed = SEED, state_type = 'joint_angle')
create_new = True

if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans == 'y':
        import shutil
        shutil.rmtree(ROOT)
        print("Deleted")

    else:
        create_new = False


if create_new:
    dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root = ROOT, 
                robot_type="omy",
                # 20 frames per second
                fps=20, 
                features={
                    "observation.image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channels"],
                    },
                    "observation.wrist_image": {
                        "dtype": "image",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["state"], # x, y, z, roll, pitch, yaw
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["action"], # 6 joint angles and 1 gripper
                    },
                    "obj_init": {
                        "dtype": "float32",
                        "shape": (6,),
                        "names": ["obj_init"], # just the initial position of the object. Not used in training.
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,
        )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset.resume(repo_id=REPO_NAME, root=ROOT)

action = np.zeros(7)
episode_id = 0
record_flag = False # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Teleoperate the robot and get delta end-effector pose with gripper
        action, reset, done = PnPEnv.teleop_robot()

        if not record_flag and sum(action)!= 0:
            record_flag = True
            print("Start recording")

        if reset:
            # Reset the environment and clear the episode buffer
            # This can be done by pressing 'z' key
            PnPEnv.reset(seed=SEED)
            # PnPEnv.reset()
            dataset.clear_episode_buffer()
            record_flag = False
        # Step the environment
        # Get the end-effector pose and images
        ee_pose = PnPEnv.get_ee_pose()
        agent_image,wrist_image = PnPEnv.grab_image()
        # resize to 256x256
        agent_image = cv2.resize(agent_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        wrist_image = cv2.resize(wrist_image, (256, 256), interpolation=cv2.INTER_LINEAR)
        joint_q = PnPEnv.step(action)
        if record_flag:
            # Add the frame to the dataset
            dataset.add_frame( {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose, 
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": TASK_NAME,
                }
            )
        PnPEnv.render(teleop=True)

        if done:
            # manually decide when this episode is done
            dataset.save_episode()
            PnPEnv.reset(seed=SEED)
            episode_id+= 1
            record_flag = False
            print("Episode saved")


PnPEnv.env.close_viewer()
# Finalize the dataset
dataset.finalize()
# Clean up the images folder
shutil.rmtree(dataset.root/'images')