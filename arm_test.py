"""
Kinova Gen3 with 2f85 gripper. 
Test file for getting keypoints in Isaac Gym
----------------
Copyright (c) 2023 SJTU RL2 Lab, Changda Tian
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time

# from arm_gym.envs.base.KinovaGen3 import *

# set random seed
np.random.seed(42)

# set torch options
torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Kinova Jacobian Inverse Kinematics Example")

# define transition matrix
HAND_2_HAMMERMID = torch.tensor([-0.15, 0, 0.14])
HAND_2_HAMMERGRASP = torch.tensor([0, 0, 0.14])
HAND_2_HAMMERHEAD = torch.tensor([-0.14, 0, 0.18])
HAND_2_HAMMERHANDLE = torch.tensor([-0.055, 0, 0.14])
HAND_2_HAMMERTAIL = torch.tensor([0.09, 0, 0.14])
HAND_2_HAMMERCLAW = torch.tensor([-0.145, 0, 0.095])
# NAIL_2_HAILHEAD = torch.tensor([0.028, 0.168, 0.014]) # nail_small

NAIL_2_HAILHEAD = torch.tensor([0.042, 0.253, 0.022]) # nail
# NAIL_2_HAILHEAD = torch.tensor([0, 0.148, 0])
# NAIL_2_HAILHEAD = torch.tensor([0.028, 0.138, 0.034])
# NAIL_2_HAILHEAD = torch.tensor([0.028, 0.029, 0.168])

# define helper functions
def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def quat_2_rotMat(q):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(q, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (q * q).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def nail_knocking_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# asset root
asset_root = "./resources/arms/kinovagen3"

# create table asset
table_dims = gymapi.Vec3(0.8, 0.6, 0.4)
table_options = gymapi.AssetOptions()
table_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_options)

# load kinova asset
kinova_asset_file = "mjcf/kinova_hammer_isaacsim_pull.xml"
kinova_options = gymapi.AssetOptions()
kinova_options.armature = 0.01
kinova_options.fix_base_link = True
kinova_options.disable_gravity = True
kinova_options.flip_visual_attachments = False
kinova_asset = gym.load_asset(sim, asset_root, kinova_asset_file, kinova_options)

# load nail asset
nail_size = 0.045
nail_asset_file = "mjcf/nail.xml"
nail_options = gymapi.AssetOptions()
# nail_options.fix_base_link = True
nail_asset = gym.load_asset(sim,asset_root,nail_asset_file, nail_options)

# load shoe asset
shoe_size = 0.002
shoe_asset_file = "mjcf/shoe.xml"
shoe_options = gymapi.AssetOptions()
shoe_asset = gym.load_asset(sim,asset_root,shoe_asset_file, shoe_options)

# configure kinova dofs
kinova_dof_props = gym.get_asset_dof_properties(kinova_asset)
kinova_dof_dict = gym.get_asset_joint_dict(kinova_asset)

kinova_dof_types = []
for i in range(len(kinova_dof_props)):
    kinova_dof_types.append(gym.get_asset_dof_type(kinova_asset,i))

kinova_lower_limits = kinova_dof_props["lower"]
for i in range(len(kinova_lower_limits)):
    if kinova_lower_limits[i]< -math.pi:
        kinova_lower_limits[i]=-math.pi

kinova_upper_limits = kinova_dof_props["upper"]
for i in range(len(kinova_upper_limits)):
    if kinova_upper_limits[i]>math.pi:
        kinova_upper_limits[i]=math.pi
kinova_ranges = kinova_upper_limits - kinova_lower_limits

# kinova_mids = [0.0, -1.0, 0.0, +2.6, -1.57, 0.0, 0.0,0,0,0,0,0,0,0,0]
kinova_mids = [0, 0.4, np.pi, -np.pi+1.4, 0, -1, np.pi/2,0,0,0,0,0,0,0,0]

# use position drive for all dofs
kinova_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
kinova_dof_props["stiffness"][:].fill(400.0)
kinova_dof_props["damping"][:].fill(40.0)

# default dof states and position targets
kinova_num_dofs = gym.get_asset_dof_count(kinova_asset)
default_dof_pos = np.zeros(kinova_num_dofs, dtype=np.float32)
default_dof_pos = kinova_mids

print("default_dof_pos ",default_dof_pos)

default_dof_state = np.zeros(kinova_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# get link index of endeffector hand, which we will use as end effector
kinova_link_dict = gym.get_asset_rigid_body_dict(kinova_asset)
print(kinova_link_dict)
kinova_hand_index = kinova_link_dict["base"] # todo change with end effector
kinova_hammer_head_index = kinova_link_dict["HammerHead"]

# configure env grid
num_envs = 16
num_dof = 15
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)


table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5 * table_dims.x, 0.0, 0.5 * table_dims.z)

kinova_pose = gymapi.Transform()
kinova_pose.p = gymapi.Vec3(table_pose.p.x-0.5*table_dims.x+0.05, 0, table_dims.z)

box_pose = gymapi.Transform()
nail_pose = gymapi.Transform()
shoe_pose = gymapi.Transform()

envs = []
actor_handles = []
j2_idxs = []
j4_idxs = []
j6_idxs = []
nail_idxs = []
nail_head_idxs = []
shoe_idxs = []
shoe_mid_idxs = []
hand_idxs = []
hammer_idxs = []
hammer_head_idxs = []
base_pos_list = []
base_rot_list = []
hand_pos_list = []
hand_rot_list = []
hammer_pos_list = []
hammer_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    #################### create env ####################
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    #################### add table ####################
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    actor_handles.append(table_handle)

    #################### add nail ####################
    nail_pose.p.x = table_pose.p.x + np.random.uniform(-0.05, 0.2)
    nail_pose.p.y = table_pose.p.y + np.random.uniform(-0.12, 0.12)
    nail_pose.p.z = table_dims.z + 0.4 * nail_size
    nail_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    # nail_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-0.001, 0.001))
    nail_handle = gym.create_actor(env, nail_asset, nail_pose, "nail", i, 1)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, nail_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    actor_handles.append(nail_handle)

    # get global index of nail in rigid body state tensor
    nail_idx = gym.get_actor_rigid_body_index(env, nail_handle, 0, gymapi.DOMAIN_SIM)
    nail_idxs.append(nail_idx)

    nail_head_idx = gym.find_actor_rigid_body_index(env, nail_handle, "NailHead", gymapi.DOMAIN_SIM)
    nail_head_idxs.append(nail_head_idx)

     #################### add shoe ####################
    shoe_pose.p.x = table_pose.p.x + np.random.uniform(-0.05, 0.2)
    shoe_pose.p.y = table_pose.p.y + np.random.uniform(-0.12, 0.12)
    shoe_pose.p.z = table_dims.z + 0.4 * nail_size
    shoe_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    # shoe_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-0.001, 0.001))
    shoe_handle = gym.create_actor(env, shoe_asset, shoe_pose, "shoe", i, 1)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, shoe_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    actor_handles.append(shoe_handle)

    # get global index of shoe in rigid body state tensor
    shoe_idx = gym.get_actor_rigid_body_index(env, shoe_handle, 0, gymapi.DOMAIN_SIM)
    shoe_idxs.append(nail_idx)

    shoe_mid_idx = gym.find_actor_rigid_body_index(env, shoe_handle, "shoe", gymapi.DOMAIN_SIM)
    shoe_mid_idxs.append(shoe_mid_idx)

    #################### add kinova ####################
    kinova_handle = gym.create_actor(env, kinova_asset, kinova_pose, "kinova", i, 2)
    actor_handles.append(kinova_handle)

    # set dof properties
    gym.set_actor_dof_properties(env, kinova_handle, kinova_dof_props)
    # set initial dof states
    gym.set_actor_dof_states(env, kinova_handle, default_dof_state, gymapi.STATE_ALL)
    # set initial position targets
    gym.set_actor_dof_position_targets(env, kinova_handle, default_dof_pos)

    # get initial base pose
    base_handle = gym.find_actor_rigid_body_handle(env, kinova_handle, "base_link")
    base_pose = gym.get_rigid_transform(env, base_handle)
    base_pos_list.append([base_pose.p.x, base_pose.p.y, base_pose.p.z])
    base_rot_list.append([base_pose.r.x, base_pose.r.y, base_pose.r.z, base_pose.r.w])
    # get initial hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, kinova_handle, "base")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    hand_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    hand_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
    # get initial hammer pose
    hammer_handle = gym.find_actor_rigid_body_handle(env, kinova_handle, "hammer")
    hammer_pose = gym.get_rigid_transform(env, hammer_handle)
    hammer_pos_list.append([hammer_pose.p.x, hammer_pose.p.y, hammer_pose.p.z])
    hammer_rot_list.append([hammer_pose.r.x, hammer_pose.r.y, hammer_pose.r.z, hammer_pose.r.w])


    # get global index of some part in rigid body state tensor
    j2_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "half_arm_1_link", gymapi.DOMAIN_SIM)
    j4_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "forearm_link", gymapi.DOMAIN_SIM)
    j6_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "spherical_wrist_2_link", gymapi.DOMAIN_SIM)
    hand_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "base", gymapi.DOMAIN_SIM)
    hammer_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "hammer", gymapi.DOMAIN_SIM)
    hammer_head_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "HammerHead", gymapi.DOMAIN_SIM)
    j2_idxs.append(j2_idx)
    j4_idxs.append(j4_idx)
    j6_idxs.append(j6_idx)
    hand_idxs.append(hand_idx)
    hammer_idxs.append(hammer_idx)
    hammer_head_idxs.append(hammer_head_idx)


# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(hand_pos_list).view(num_envs, -1).to(device)
init_rot = torch.Tensor(hand_rot_list).view(num_envs, -1).to(device)
# print("init_pos:",init_pos)
# print("init_rot:",init_rot)

# hand orientation for hammering
hammer_q = torch.stack(num_envs * [torch.tensor([0.0, -1.0, 0.0, 0.0])]).to(device).view((num_envs, -1))

# initial base position and orientation tensors
init_base_pos = torch.Tensor(base_pos_list).view(num_envs, -1).to(device)
init_base_rot = torch.Tensor(base_rot_list).view(num_envs, -1).to(device)

# initial hammer position and orientation tensors
init_hammer_pos = torch.Tensor(hammer_pos_list).view(num_envs, -1).to(device)
init_hammer_rot = torch.Tensor(hammer_rot_list).view(num_envs, -1).to(device)

# nail head coords, used to determine hammering pos
# nail_half_size = 0.5 * nail_size
# nail_head_coord = torch.Tensor([nail_half_size, nail_half_size, nail_half_size])
# nail_head = torch.stack(num_envs * [nail_head_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base kinova, tensor has shape (num envs, 10, 6, 7) todo
_jacobian = gym.acquire_jacobian_tensor(sim, "kinova")
jacobian = gymtorch.wrap_tensor(_jacobian)

# print("jacobian",jacobian)
print("jacobian_shape",jacobian.shape)

# jacobian entries corresponding to kinova hand
# j_eef = jacobian[:, kinova_hand_index - 1, :]
j_eef = jacobian[:, kinova_hammer_head_index - 1, :]

# print("j_eef:",j_eef)
print("jeef[0,0:7]",j_eef[0,:])

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)
print(rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, kinova_num_dofs, 1)

# get contact forces tensor
_net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
net_contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(num_envs, -1, 3)

tip_names = ["HammerHead"]
tip_indices = torch.zeros(len(tip_names), dtype=torch.long, device=device, requires_grad=False)
for i in range(len(tip_names)):
    tip_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[2], tip_names[i])
print("tip_indices: ",tip_indices)

nail_names = ["NailHead"]
nail_indices = torch.zeros(len(nail_names), dtype=torch.long, device=device, requires_grad=False)
for i in range(len(nail_names)):
    nail_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[1], nail_names[i])
print("nail_indices: ",nail_indices)

# table_names = ["table"]
# table_indices = torch.zeros(len(table_names), dtype=torch.long, device=device, requires_grad=False)
# for i in range(len(table_names)):
#     table_indices[i] = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], table_names[i])
# print("table_indices: ",table_indices)
# print(1/0)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

counter = 0

# simulation loops
while not gym.query_viewer_has_closed(viewer):
    counter+=1
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim,True)
    

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim)

    j2_pos = rb_states[j2_idxs, :3]
    j4_pos = rb_states[j4_idxs, :3]
    j6_pos = rb_states[j6_idxs, :3]

    nail_pos = rb_states[nail_idxs, :3]
    nail_rot = rb_states[nail_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    hammer_pos = rb_states[hammer_idxs, :3]
    hammer_rot = rb_states[hammer_idxs, 3:7]

    hammer_mid = hand_pos.view(num_envs,3,1) + quat_2_rotMat(hand_rot).view(num_envs,3,3).to(torch.float32).to(device) @ HAND_2_HAMMERMID.view(3,1).to(torch.float32).to(device)
    hammer_head = hand_pos.view(num_envs,3,1) + quat_2_rotMat(hand_rot).view(num_envs,3,3).to(torch.float32).to(device) @ HAND_2_HAMMERHEAD.view(3,1).to(torch.float32).to(device)
    hammer_tail = hand_pos.view(num_envs,3,1) + quat_2_rotMat(hand_rot).view(num_envs,3,3).to(torch.float32).to(device) @ HAND_2_HAMMERTAIL.view(3,1).to(torch.float32).to(device)
    hammer_claw = hand_pos.view(num_envs,3,1) + quat_2_rotMat(hand_rot).view(num_envs,3,3).to(torch.float32).to(device) @ HAND_2_HAMMERCLAW.view(3,1).to(torch.float32).to(device)
    hammer_grasp = hand_pos.view(num_envs,3,1) + quat_2_rotMat(hand_rot).view(num_envs,3,3).to(torch.float32).to(device) @ HAND_2_HAMMERGRASP.view(3,1).to(torch.float32).to(device)

    # print("j2 pos: ", j2_pos)
    # print("j4 pos: ", j4_pos)
    # print("j6 pos: ", j6_pos)
    # print("hammer_mid: ",hammer_mid)
    # print("hammer_head: ",hammer_head)
    # print("hammer_tail: ",hammer_tail)
    # print("hammer_claw: ",hammer_claw)
    # print("hammer_grasp: ",hammer_grasp)
    # print("hammer_pos: ",hammer_pos)

    hammer_mid_pos = hammer_mid.view(num_envs,3)
    hammer_head_pos = hammer_head.view(num_envs,3)
    hammer_tail_pos = hammer_tail.view(num_envs,3)
    hammer_claw_pos = hammer_claw.view(num_envs,3)
    hammer_grasp_pos = hammer_grasp.view(num_envs,3)

    nail_head = nail_pos.view(num_envs,3,1) + quat_2_rotMat(nail_rot).view(num_envs,3,3).to(torch.float32).to(device) @ NAIL_2_HAILHEAD.view(3,1).to(torch.float32).to(device) 
    nail_head_pos = nail_head.view(num_envs,3)
    
    if 20<= counter <= 30:
        if counter == 20:
            delta_hammer_head_2_hand = hammer_head_pos - hand_pos
            tan_alpha = delta_hammer_head_2_hand[:,2] / delta_hammer_head_2_hand[:,0]
            # print("hammer_head_pos ",hammer_head_pos)
            # print("hand_pos ",hand_pos)
            # print("hand_rot ",hand_rot)
            # print("hammer_2_hand_angle ",torch.arctan(tan_alpha))

        # to_nail = nail_pos - hand_pos
        to_nail = nail_head_pos - hammer_head_pos
        nail_dist = torch.norm(to_nail, dim=-1).unsqueeze(-1)
        nail_dir = to_nail / nail_dist
        nail_dot = nail_dir @ down_dir.view(3,1)

        # how far the hand should be from box for knocking
        knock_offset = 0.05

        yaw_q = nail_knocking_yaw(nail_rot, nail_head_pos)
        # print("yaw_q",yaw_q)
        nail_yaw_dir = quat_axis(yaw_q, 0)
        hand_yaw_dir = quat_axis(hand_rot, 0)
        yaw_dot = torch.bmm(nail_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

        # determine if we have reached the initial position; if so allow the hand to start moving to the box
        to_init = init_pos - hand_pos
        init_dist = torch.norm(to_init, dim=-1)
        hand_restart =  (init_dist > 0.5).squeeze(-1)
        return_to_start = hand_restart.unsqueeze(-1)
        
        # if hand is above box, descend to grasp offset
        # otherwise, seek a position above the box
        above_box = ((nail_dot >= 0.99) & (yaw_dot >= 0.95) & (nail_dist < knock_offset * 3)).squeeze(-1)
        grasp_pos = nail_head_pos.clone()
        grasp_pos[:, 2] = torch.where(above_box, nail_head_pos[:, 2] + knock_offset, nail_head_pos[:, 2] + knock_offset * 2.5)

        # compute goal position and orientation
        # print("return_to_start: ", return_to_start)
        goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
        goal_rot = torch.where(return_to_start, init_rot, hammer_q)

        # print("goal_pos: ",goal_pos)
        # print("goal_rot: ",goal_rot)

        # compute position and orientation error
        pos_err = goal_pos - hammer_head_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 15, 1)

        # update position targets
        pos_target = dof_pos + u

        # set new position targets
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    if 30<= counter <= 40:
        # knock
        goal_pos = nail_head_pos.clone()
        goal_rot = torch.where(return_to_start, init_rot, hammer_q)

        # compute position and orientation error
        pos_err = goal_pos - hammer_head_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 15, 1)

        # update position targets
        pos_target = dof_pos + u

        # set new position targets
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    if counter == 41:
        counter = 0
        pos_vel = torch.zeros((num_envs,15),device=device)
        gym.set_dof_velocity_target_tensor(sim,gymtorch.unwrap_tensor(pos_vel))
        


    elif counter == 250:
         gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(torch.cat([init_pos, init_rot], -1).unsqueeze(-1)))
    
    nail_head_pos = rb_states[nail_head_idxs,:][:,:3]
    # print("nail_head_pos: ", nail_head_pos[0])
    hammer_head_pos = rb_states[hammer_head_idxs,:][:,:3]
    dis = torch.sqrt(torch.sum(torch.square(nail_head_pos-hammer_head_pos), dim=1))

    # update viewer
    print("count: ", counter, "dis:", dis[3])
    print("count: ", counter, "nail force: ",net_contact_forces[:,nail_indices,:][3])
    print("count: ", counter, "hammer force: ",net_contact_forces[:,tip_indices,:][3],"\n")

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)