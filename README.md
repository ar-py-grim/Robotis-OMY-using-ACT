**Packages used**
1. mujoco == 3.6
2. lerobot == 0.5.1
3. opencv-python == 4.13.0.92
4. torch == 2.10.0
5. matplotlib == 3.10.8

<img width="3072" height="1657" alt="image" src="https://github.com/user-attachments/assets/bcb65334-414c-459e-a898-16f74b09bb01" />

### 1. To collect data
Run **data_collect.py** for data collection. 
Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROW keys for the rest of the rotations.
SPACEBAR will change your gripper's state, P key will pause recording, and Z key will reset your environment with discarding the current episode data.

Pre-recorded dataset available at https://huggingface.co/datasets/ar-py-grim/omy_pnp/tree/main
place **demo_data/** inside workspace.

### 2. Visualize data
Data recorded can be replayed either entire dataset or a certain episode

### 3. Train Model
Run **train_act.py** for training the model on the data stored in **demo_data/**. 

Pre-trained model is at https://huggingface.co/ar-py-grim/omy_pnp_act/tree/main
place **ckpt/act_y/** inside workspace. 
<img width="999" height="999" alt="ACT pred v gt" src="https://github.com/user-attachments/assets/38de3d4a-4b59-468b-b79b-684a02dbee01" />

### 4. Train Model
Run **test_act.py** to run the trained model

### Implementations Link
https://youtu.be/lryWgsryg5A
