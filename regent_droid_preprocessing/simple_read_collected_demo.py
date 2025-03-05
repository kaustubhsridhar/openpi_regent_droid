import numpy as np
import json 
import h5py
import os

fol = "collected_demos/2025-03-04/2025-03-04_00-17-49"

traj_h5 = h5py.File(f"{fol}/trajectory.h5", "r")

print(f'{traj_h5.keys()=}') # ['action', 'observation']
print(f'{traj_h5["action"].keys()=}') # ['cartesian_position', 'cartesian_velocity', 'gripper_position', 'gripper_velocity', 'joint_position', 'joint_velocity', 'robot_state', 'target_cartesian_position', 'target_gripper_position']
print(f'{traj_h5["observation"].keys()=}') # ['camera_extrinsics', 'camera_intrinsics', 'camera_type', 'controller_info', 'robot_state', 'timestamp']

print(f'{traj_h5["action"]["cartesian_position"].shape=}') # (561, 6)
print(f'{traj_h5["action"]["cartesian_velocity"].shape=}') # (561, 6)
print(f'{traj_h5["action"]["gripper_position"].shape=}') # (561,)
print(f'{traj_h5["action"]["gripper_velocity"].shape=}') # (561,)
print(f'{traj_h5["action"]["joint_position"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["joint_velocity"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["robot_state"].keys()=}') # ['cartesian_position', 'gripper_position', 'joint_positions', 'joint_torques_computed', 'joint_velocities', 'motor_torques_measured', 'prev_command_successful', 'prev_controller_latency_ms', 'prev_joint_torques_computed', 'prev_joint_torques_computed_safened']
print(f'{traj_h5["action"]["robot_state"]["cartesian_position"].shape=}') # (561, 6)
print(f'{traj_h5["action"]["robot_state"]["gripper_position"].shape=}') # (561,)
print(f'{traj_h5["action"]["robot_state"]["joint_positions"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["robot_state"]["joint_torques_computed"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["robot_state"]["joint_velocities"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["robot_state"]["motor_torques_measured"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["robot_state"]["prev_command_successful"].shape=}') # (561,)
print(f'{traj_h5["action"]["robot_state"]["prev_controller_latency_ms"].shape=}') # (561,)
print(f'{traj_h5["action"]["robot_state"]["prev_joint_torques_computed"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["robot_state"]["prev_joint_torques_computed_safened"].shape=}') # (561, 7)
print(f'{traj_h5["action"]["target_cartesian_position"].shape=}') # (561, 6)
print(f'{traj_h5["action"]["target_gripper_position"].shape=}') # (561,)

print(f'\n\n\n')
print(f'{traj_h5["observation"]["camera_extrinsics"].keys()=}') # ['14436910_left', '14436910_left_gripper_offset', '14436910_right', '14436910_right_gripper_offset', '19824535_left', '19824535_right', '23404442_left', '23404442_right', '23748752_left', '23748752_right', '25455306_left', '25455306_right', '26368109_left', '26368109_right', '26405488_left', '26405488_right', '27085680_left', '27085680_right', '29838012_left', '29838012_right']
print(f'{traj_h5["observation"]["camera_intrinsics"].keys()=}') # ['14436910_left', '14436910_right', '25455306_left', '25455306_right', '27085680_left', '27085680_right']
print(f'{traj_h5["observation"]["camera_type"].keys()=}') # ['14436910', '25455306', '27085680']
print(f'{traj_h5["observation"]["controller_info"].keys()=}') # ['controller_on', 'failure', 'movement_enabled', 'success']
print(f'{traj_h5["observation"]["robot_state"].keys()=}') # ['cartesian_position', 'gripper_position', 'joint_positions', 'joint_torques_computed', 'joint_velocities', 'motor_torques_measured', 'prev_command_successful', 'prev_controller_latency_ms', 'prev_joint_torques_computed', 'prev_joint_torques_computed_safened']
print(f'{traj_h5["observation"]["robot_state"]["cartesian_position"].shape=}') # (561, 6)
print(f'{traj_h5["observation"]["robot_state"]["gripper_position"].shape=}') # (561,)
print(f'{traj_h5["observation"]["robot_state"]["joint_positions"].shape=}') # (561, 7)
print(f'{traj_h5["observation"]["robot_state"]["joint_torques_computed"].shape=}') # (561, 7)
print(f'{traj_h5["observation"]["robot_state"]["joint_velocities"].shape=}') # (561, 7)
print(f'{traj_h5["observation"]["robot_state"]["motor_torques_measured"].shape=}') # (561, 7)
print(f'{traj_h5["observation"]["robot_state"]["prev_command_successful"].shape=}') # (561,)
print(f'{traj_h5["observation"]["robot_state"]["prev_controller_latency_ms"].shape=}') # (561,)
print(f'{traj_h5["observation"]["robot_state"]["prev_joint_torques_computed"].shape=}') # (561, 7)
print(f'{traj_h5["observation"]["robot_state"]["prev_joint_torques_computed_safened"].shape=}') # (561, 7)
print(f'{traj_h5["observation"]["timestamp"].keys()=}') # ['cameras', 'control', 'robot_state', 'skip_action']
print(f'{traj_h5["observation"]["timestamp"]["cameras"].keys()=}') # ['14436910_estimated_capture', '14436910_frame_received', '14436910_read_end', '14436910_read_start', '25455306_estimated_capture', '25455306_frame_received', '25455306_read_end', '25455306_read_start', '27085680_estimated_capture', '27085680_frame_received', '27085680_read_end', '27085680_read_start']
print(f'{traj_h5["observation"]["timestamp"]["control"].keys()=}') # ['control_start', 'policy_start', 'sleep_start', 'step_end', 'step_start']
print(f'{traj_h5["observation"]["timestamp"]["robot_state"].keys()=}') # ['read_end', 'read_start', 'robot_timestamp_nanos', 'robot_timestamp_seconds']
print(f'{traj_h5["observation"]["timestamp"]["skip_action"].shape=}') # (561,)

print(f'\n\n\n')
# check if ["observation"]["robot_state"] is the same as ["action"]["robot_state"]
print(f'{np.allclose(traj_h5["observation"]["robot_state"]["cartesian_position"], traj_h5["action"]["robot_state"]["cartesian_position"])=}') # False
print(f'{np.allclose(traj_h5["observation"]["robot_state"]["gripper_position"], traj_h5["action"]["robot_state"]["gripper_position"])=}') # True
print(f'{np.allclose(traj_h5["observation"]["robot_state"]["joint_positions"], traj_h5["action"]["robot_state"]["joint_positions"])=}') # False
print(f'{np.allclose(traj_h5["observation"]["robot_state"]["joint_torques_computed"], traj_h5["action"]["robot_state"]["joint_torques_computed"])=}') # False
print(f'{np.allclose(traj_h5["observation"]["robot_state"]["joint_velocities"], traj_h5["action"]["robot_state"]["joint_velocities"])=}') # False
print(f'{np.allclose(traj_h5["observation"]["robot_state"]["motor_torques_measured"], traj_h5["action"]["robot_state"]["motor_torques_measured"])=}') # False

print(f'\n\n\n')
traj_np = np.load(f"{fol}/trajectory.npz")
print(f'{traj_np.keys()=}') # states, actions_pos, actions_vel
print(f'{traj_np["states"].shape=}')  # (107, 63)
print(f'{traj_np["actions_pos"].shape=}')  # (107, 6)
print(f'{traj_np["actions_vel"].shape=}')  # (107, 6)

print(f'\n\n\n')
# h5 records the skip action too unlike npz
print(f'{traj_h5["observation"]["timestamp"]["skip_action"].dtype=}') # bool
h5_actions_cartesian_position = traj_h5["action"]["cartesian_position"][:]
print(f'{h5_actions_cartesian_position.shape=}') # (561, 6)
skip_bools = traj_h5["observation"]["timestamp"]["skip_action"][:]
use_bools = ~skip_bools
print(f'{use_bools.shape=}') # (561,)
print(f'{np.sum(use_bools)=}') # 107
h5_actions_cartesian_position_use = h5_actions_cartesian_position[use_bools]
print(f'{h5_actions_cartesian_position_use.shape=}') # (107, 6)

# assert h5_actions_cartesian_position_use same as actions_pos upto its first 6 dimensions
print(f'{np.allclose(h5_actions_cartesian_position_use, traj_np["actions_pos"][:, :6])=}') # True

# similarly,
h5_actions_cartesian_velocity_use = traj_h5["action"]["cartesian_velocity"][~traj_h5["observation"]["timestamp"]["skip_action"][:]]
print(f'{h5_actions_cartesian_velocity_use.shape=}') # (107, 6)
print(f'{np.allclose(h5_actions_cartesian_velocity_use, traj_np["actions_vel"][:, :6])=}') # True

