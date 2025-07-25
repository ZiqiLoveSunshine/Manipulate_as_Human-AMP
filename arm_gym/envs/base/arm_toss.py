# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2023 SJTU, Changda Tian

from arm_gym import ARM_GYM_ROOT_DIR
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from arm_gym.utils.KinovaGen3 import forward_kinematics
import torch

from arm_gym import ARM_GYM_ROOT_DIR
from arm_gym.envs.base.base_task import BaseTask

from arm_gym.utils.math import quat_apply_yaw, quat_2_rotMat, angle_of_vectors
from arm_gym.utils.helpers import class_to_dict
from .arm_toss_config import ArmTossCfg
from rsl_rl.datasets.keypoint_loader_toss import AMPLoader


# define transition matrix
HAND_2_PANMID = torch.tensor([-0.15, 0, 0.15])
HAND_2_PANGRASP = torch.tensor([0, 0, 0.14])


class ArmToss(BaseTask):
    def __init__(self, cfg: ArmTossCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)

        # define prims present in the scene
        prim_names = ["table", "kinova", "pita"]
        # mapping from name to gym indices
        self.gym_indices = dict.fromkeys(prim_names)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.flip_pita_times = 0

        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        if self.cfg.env.include_history_amp_steps is not None:
            self.amp_obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.amp_obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, amp_obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, amp_obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        
        for _ in range(self.cfg.control.decimation):
            self.gym.refresh_dof_state_tensor(self.sim)
            # print("step, dof_state: ", self.kinova_dof_state[0])
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf

        if self.cfg.env.include_history_amp_steps is not None: # add amp observation buffer
            self.amp_obs_buf_history.reset(reset_env_ids, self.amp_obs_buf[reset_env_ids])
            self.amp_obs_buf_history.insert(self.amp_obs_buf)
            amp_obs = self.amp_obs_buf_history.get_obs_vec(np.arange(self.include_history_amp_steps))
        else:
            amp_obs = self.amp_obs_buf

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return policy_obs, amp_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        return policy_obs
    
    def get_amp_observations(self):
        if self.cfg.env.include_history_amp_steps is not None:
            amp_obs = self.amp_obs_buf_history.get_obs_vec(np.arange(self.include_history_amp_steps))
        else:
            amp_obs = self.amp_obs_buf
        return amp_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """

        # print("refresh_net_contact_force: ",self.gym.refresh_net_contact_force_tensor(self.sim))
        # print("refresh_actor_root_state_tensor: ",self.gym.refresh_actor_root_state_tensor(self.sim))
        # print("refresh_rigid_body_state_tensor: ",self.gym.refresh_rigid_body_state_tensor(self.sim))
        # print("refresh_dof_state_tensor: ",self.gym.refresh_dof_state_tensor(self.sim))
        # print("refresh_force_sensor_tensor: ",self.gym.refresh_force_sensor_tensor(self.sim))

        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        print("env num to be reset: ",len(env_ids))
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_amp_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        # self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        self.pita_drop_buf = self.rt_states[self.pita_idxs,:][:,2] < self.cfg.table.z_size + 0.05
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.flip_pan_buf = self.detect_pan_flip()

        
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.pita_drop_buf
        self.reset_buf |= self.flip_pan_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        if self.cfg.env.reference_state_initialization:
            # frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            # self._reset_dofs_amp(env_ids, frames)
            self._reset_dofs(env_ids)
            self._reset_pita(env_ids)
        else:
            self._reset_dofs(env_ids)
            self._reset_pita(env_ids)

        self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # print("reward: ",name, rew)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations, not for amp, is for traditional
        """
        hand_pos = self.rb_states[self.hand_idxs, :3]
        hand_rot = self.rb_states[self.hand_idxs, 3:7]
        j2_pos = self.rb_states[self.j2_idxs, :3]
        j4_pos = self.rb_states[self.j4_idxs, :3]
        j6_pos = self.rb_states[self.j6_idxs, :3]
        # print(hand_pos)
        # print("hand_rot: ", hand_rot)
        pan_grasp = hand_pos.view(-1,3,1) + quat_2_rotMat(hand_rot).view(-1,3,3).to(torch.float32) @ HAND_2_PANGRASP.view(3,1).to(torch.float32).to(self.device)
        pan_mid = hand_pos.view(-1,3,1) + quat_2_rotMat(hand_rot).view(-1,3,3).to(torch.float32) @ HAND_2_PANMID.view(3,1).to(torch.float32).to(self.device)

        pan_grasp = pan_grasp.view(-1,1,3).squeeze()
        pan_mid = pan_mid.view(-1,1,3).squeeze()

        # print("pan shape: ", pan_grasp.shape)
        # print("j2 shape: ", j2_pos.shape)
        vec_panHandleMid = pan_mid - pan_grasp
        vec_panHandleMid = vec_panHandleMid.view(-1,3)

        vec_j4j6 = j6_pos - j4_pos
        vec_j2j4 = j4_pos - j2_pos


        x_axis = torch.tensor((1,0,0), device=self.device).unsqueeze(0).repeat(vec_panHandleMid.shape[0],1)
        y_axis = torch.tensor((0,1,0), device=self.device).unsqueeze(0).repeat(vec_panHandleMid.shape[0],1)
        z_axis = torch.tensor((0,0,1), device=self.device).unsqueeze(0).repeat(vec_panHandleMid.shape[0],1)

        ang_panHandleMid_j4j6 = angle_of_vectors(vec_panHandleMid, vec_j4j6).to(self.device)
        ang_j4j6_j2j4 = angle_of_vectors(vec_j4j6,vec_j2j4).to(self.device)
        
        ang_j2j4_x = angle_of_vectors(vec_j2j4,x_axis).to(self.device)
        ang_j2j4_y = angle_of_vectors(vec_j2j4,y_axis).to(self.device)
        ang_j2j4_z = angle_of_vectors(vec_j2j4,z_axis).to(self.device)
        ang_j4j6_x = angle_of_vectors(vec_j4j6,x_axis).to(self.device)
        ang_j4j6_y = angle_of_vectors(vec_j4j6,y_axis).to(self.device)
        ang_j4j6_z = angle_of_vectors(vec_j4j6,z_axis).to(self.device)
        ang_pan_x = angle_of_vectors(vec_panHandleMid, x_axis).to(self.device)
        ang_pan_y = angle_of_vectors(vec_panHandleMid, y_axis).to(self.device)
        ang_pan_z = angle_of_vectors(vec_panHandleMid, z_axis).to(self.device)

        hand_pos = self.rb_states[self.hand_idxs, :3]
        pita_pos = self.rb_states[self.pitaRoot_idxs, :3]
        pan_mid_pos = self.rb_states[self.pan_mid_idxs,:3]
        self.pan_mid_r = self.rb_states[self.pan_mid_idxs,3:7]

        self.privileged_obs_buf = torch.cat((hand_pos, pita_pos, pan_mid_pos,
                                             ang_panHandleMid_j4j6, ang_j4j6_j2j4,
                                             ang_j2j4_x, ang_j2j4_y, ang_j2j4_z,
                                             ang_j4j6_x, ang_j4j6_z,
                                             ang_pan_x, ang_pan_z,
                                            self.commands[:, :3] * self.commands_scale,
                                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                            self.dof_vel * self.obs_scales.dof_vel,
                                            self.actions,
                                            ),dim=-1)
        
        # print("pita_pos: ", pita_pos)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec
        # print(((2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec)[0])
        # print(1/0)
        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 6:
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)

    def compute_amp_observations(self):
        hand_pos = self.rb_states[self.hand_idxs, :3]
        hand_rot = self.rb_states[self.hand_idxs, 3:7]
        j2_pos = self.rb_states[self.j2_idxs, :3]
        j4_pos = self.rb_states[self.j4_idxs, :3]
        j6_pos = self.rb_states[self.j6_idxs, :3]
        # print(hand_pos)
        # print("hand_rot: ", hand_rot)
        pan_grasp = hand_pos.view(-1,3,1) + quat_2_rotMat(hand_rot).view(-1,3,3).to(torch.float32) @ HAND_2_PANGRASP.view(3,1).to(torch.float32).to(self.device)
        pan_mid = hand_pos.view(-1,3,1) + quat_2_rotMat(hand_rot).view(-1,3,3).to(torch.float32) @ HAND_2_PANMID.view(3,1).to(torch.float32).to(self.device)

        pan_grasp = pan_grasp.view(-1,1,3).squeeze()
        pan_mid = pan_mid.view(-1,1,3).squeeze()

        # print("pan shape: ", pan_grasp.shape)
        # print("j2 shape: ", j2_pos.shape)
        vec_panHandleMid = pan_mid - pan_grasp
        vec_panHandleMid = vec_panHandleMid.view(-1,3)
        vec_j4j6 = j6_pos - j4_pos
        vec_j2j4 = j4_pos - j2_pos

        x_axis = torch.tensor((1,0,0), device=self.device).unsqueeze(0).repeat(vec_panHandleMid.shape[0],1)
        y_axis = torch.tensor((0,1,0), device=self.device).unsqueeze(0).repeat(vec_panHandleMid.shape[0],1)
        z_axis = torch.tensor((0,0,1), device=self.device).unsqueeze(0).repeat(vec_panHandleMid.shape[0],1)

        ang_panHandleMid_j4j6 = angle_of_vectors(vec_panHandleMid, vec_j4j6).to(self.device)
        ang_j4j6_j2j4 = angle_of_vectors(vec_j4j6,vec_j2j4).to(self.device)
        
        ang_j2j4_x = angle_of_vectors(vec_j2j4,x_axis).to(self.device)
        ang_j2j4_y = angle_of_vectors(vec_j2j4,y_axis).to(self.device)
        ang_j2j4_z = angle_of_vectors(vec_j2j4,z_axis).to(self.device)
        ang_j4j6_x = angle_of_vectors(vec_j4j6,x_axis).to(self.device)
        ang_j4j6_y = angle_of_vectors(vec_j4j6,y_axis).to(self.device)
        ang_j4j6_z = angle_of_vectors(vec_j4j6,z_axis).to(self.device)
        ang_pan_x = angle_of_vectors(vec_panHandleMid, x_axis).to(self.device)
        ang_pan_y = angle_of_vectors(vec_panHandleMid, y_axis).to(self.device)
        ang_pan_z = angle_of_vectors(vec_panHandleMid, z_axis).to(self.device)


        pan_grasp = pan_grasp.view(-1,3)
        pan_mid = pan_mid.view(-1,3)

        obs = torch.cat((ang_j2j4_x, ang_j2j4_y, ang_j2j4_z, ang_j4j6_x, ang_j4j6_z, 
                         ang_pan_x, ang_pan_z, ang_panHandleMid_j4j6, ang_j4j6_j2j4), dim=-1)        
        self.amp_obs_buf = torch.clone(obs)



    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        self._create_ground_plane()
        self._create_table()
        self._create_pita()
        self._create_kinova()

        self._create_envs()

    def set_camera(self, position, lookat, select_env = None):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, select_env, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = max(props["lower"][i].item(),-np.pi)
                self.dof_pos_limits[i, 1] = min(props["upper"][i].item(),np.pi)
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
        return props

    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["x_range"][0], self.command_ranges["x_range"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["y_range"][0], self.command_ranges["y_range"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["z_range"][0], self.command_ranges["z_range"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["a_range"][0], self.command_ranges["a_range"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.commands[env_ids, :4] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.01).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_pos[env_ids,:] = self.default_dof_pos
        self.dof_vel[env_ids,:] = 0.

        self.kinova_dof_pos[env_ids,:] = self.default_dof_pos
        self.kinova_dof_vel[env_ids,:] = 0.
        # print("I'm in")
        # print("dof_state: ", self.kinova_dof_state[0])

        # print("dof_state shape",self.dof_state.shape)
        # print("kinova dof_state shape",self.kinova_dof_state.shape)

        kinova_indices = self.gym_indices["kinova"][env_ids].to(torch.int32)
        # print("kinova_indices: ",kinova_indices)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.kinova_dof_state),
                                              gymtorch.unwrap_tensor(kinova_indices), len(kinova_indices))

    def _reset_pita(self,env_ids):
        """ Resets pita position of selected environmments

        Args:
            env_ids (List[int]): Environemnt ids
        """
        for i in env_ids:
            self.nl_pose.p = gymapi.Vec3(self.init_pan_mid[0], self.init_pan_mid[1], self.init_pan_mid[2] - self.cfg.asset.pita_offset)
            self.nl_pose.r = gymapi.Quat.from_euler_zyx(np.random.uniform(-np.pi, np.pi),0,np.pi/2)


            pita_p = torch.tensor([self.nl_pose.p.x,self.nl_pose.p.y,self.nl_pose.p.z],device=self.device)
            pita_r = torch.tensor([self.nl_pose.r.w,self.nl_pose.r.x,self.nl_pose.r.y,self.nl_pose.r.z],device=self.device)
            pita_0 = torch.zeros(6,device=self.device)
            pita_state = torch.concat((pita_p,pita_r,pita_0))

            self.pita_states[i] = pita_state

        pita_indices = self.gym_indices["pita"][env_ids].to(torch.int32)


        for i in range(len(pita_indices)):
            self.rt_states[pita_indices[i]] = self.pita_states[i]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.rt_states),
                                              gymtorch.unwrap_tensor(pita_indices), len(pita_indices))




    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        print("gen noise: ", self.privileged_obs_buf[0].shape)
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.hand_pos * noise_level * self.obs_scales.hand_pos
        noise_vec[3:6] = noise_scales.pita_pos * noise_level * self.obs_scales.pita_pos
        noise_vec[6:9] = noise_scales.pan_mid_pos * noise_level * self.obs_scales.pan_mid_pos
        noise_vec[9:18] = noise_scales.angle *noise_level *self.obs_scales.angle
        noise_vec[18:21] = 0. # commands
        noise_vec[21:36] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[36:51] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[51:66] = 0. # previous actions
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rt_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # update information
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.rb_states = gymtorch.wrap_tensor(rb_states)
        self.rt_states = gymtorch.wrap_tensor(rt_states)
        self.root_states = self.rb_states[self.root_idxs,:]
        self.pita_states = self.rt_states[self.pita_idxs,:]
        self.pita_root_states = self.rb_states[self.pitaRoot_idxs,:]
        self.pan_mid_states = self.rb_states[self.pan_mid_idxs,:]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.kinova_dof_state = self.dof_state.view(self.num_envs,-1,2)[:,:self.num_kinova_dofs]

        self.kinova_dof_pos = self.kinova_dof_state[..., 0]
        self.kinova_dof_vel = self.kinova_dof_state[..., 1]

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.pita_pos, self.obs_scales.pita_pos, self.obs_scales.pita_pos], device=self.device, requires_grad=False,) # TODO change this
        self.last_contacts = torch.zeros(self.num_envs, len(self.tip_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self._compute_randomized_gains(self.num_envs)

    def _compute_randomized_gains(self, num_envs):
        p_mult = ((
            self.cfg.domain_rand.stiffness_multiplier_range[0] -
            self.cfg.domain_rand.stiffness_multiplier_range[1]) *
            torch.rand(num_envs, self.num_actions, device=self.device) +
            self.cfg.domain_rand.stiffness_multiplier_range[1]).float()
        d_mult = ((
            self.cfg.domain_rand.damping_multiplier_range[0] -
            self.cfg.domain_rand.damping_multiplier_range[1]) *
            torch.rand(num_envs, self.num_actions, device=self.device) +
            self.cfg.domain_rand.damping_multiplier_range[1]).float()
        
        return p_mult * self.p_gains, d_mult * self.d_gains


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_table(self):
        """ Adds a table for manipulation to the simulation, sets parameters based on the cfg.
        """
        self.tb_dims = gymapi.Vec3(self.cfg.table.x_size,self.cfg.table.y_size,self.cfg.table.z_size)
        tb_asset_options = gymapi.AssetOptions()
        tb_asset_options.fix_base_link = True
        self.tb_asset = self.gym.create_box(self.sim,self.tb_dims.x, self.tb_dims.y,self.tb_dims.z,tb_asset_options)

        self.tb_pose = gymapi.Transform()
        self.tb_pose.p = gymapi.Vec3(self.cfg.table.x_pos,self.cfg.table.y_pos,self.cfg.table.z_pos)


    def _create_pita(self):
        """ Adds a pita to the simulation, sets parameters based on the cfg.
        # """
        # self.nl_size = self.cfg.pita.size
        nl_file = self.cfg.asset.pita_file
        nl_asset_options = gymapi.AssetOptions()
        nl_asset_options.fix_base_link = False
        self.nl_asset = self.gym.load_asset(self.sim,ARM_GYM_ROOT_DIR, nl_file ,nl_asset_options)
        self.nl_pose = gymapi.Transform()
        
    def _create_kinova(self):
        
        asset_path = self.cfg.asset.file.format(ARM_GYM_ROOT_DIR=ARM_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # asset_options.convex_decomposition_from_submeshes = True
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)

        self.kinova_pose = gymapi.Transform()
        self.kinova_pose.p = gymapi.Vec3(self.tb_pose.p.x-0.5*self.tb_dims.x+0.05, 0, self.tb_dims.z)

    def _quat2list(self,q):
        return [q.w,q.x,q.y,q.z]

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        
        # kinova_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        # print(kinova_link_dict)

        default_dof_pos = np.array([0, 0.4, np.pi, -np.pi+1.4, 0, -1, np.pi/2,0,0,0,0,0,0,0,0],dtype=np.float32)
        default_dof_state = np.zeros(self.num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        self.num_kinova_dofs = self.gym.get_asset_dof_count(self.robot_asset)

        # initialize gym indices buffer as a list
        # note: later the list is converted to torch tensor for ease in interfacing with IsaacGym.
        for asset_name in self.gym_indices.keys():
            self.gym_indices[asset_name] = list()

        tip_names = [self.cfg.asset.tip_name]
        pita_names = [self.cfg.asset.pita_name]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        self._get_env_origins()
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.actor_handles = []
        self.envs = []
        self.root_idxs = []
        self.hand_idxs = []
        self.j6_idxs = []
        self.j4_idxs = []
        self.j2_idxs = []
        self.pitaRoot_idxs = []
        self.pita_idxs = []
        self.pan_mid_idxs = []
        self.initial_pan_orientation = []

        for i in range(self.num_envs):
            ############################### create env instance ###############################
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)
            
            ############################### add kinova ###############################
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

            kinova_handle = self.gym.create_actor(env_handle, self.robot_asset, self.kinova_pose, "kinova", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, kinova_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, kinova_handle)
             
            self.gym.set_actor_rigid_body_properties(env_handle, kinova_handle, body_props, recomputeInertia=True)
            self.actor_handles.append(kinova_handle)
            kinova_idx = self.gym.get_actor_index(env_handle,kinova_handle,gymapi.DOMAIN_SIM)
            # set initial dof states
            self.gym.set_actor_dof_states(env_handle, kinova_handle, default_dof_state, gymapi.STATE_ALL)

            # get global index of some part in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env_handle, kinova_handle, "base", gymapi.DOMAIN_SIM)
            j2_idx = self.gym.find_actor_rigid_body_index(env_handle, kinova_handle, "half_arm_1_link", gymapi.DOMAIN_SIM)
            j4_idx = self.gym.find_actor_rigid_body_index(env_handle, kinova_handle, "forearm_link", gymapi.DOMAIN_SIM)
            j6_idx = self.gym.find_actor_rigid_body_index(env_handle, kinova_handle, "spherical_wrist_2_link", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)
            self.j2_idxs.append(j2_idx)
            self.j4_idxs.append(j4_idx)
            self.j6_idxs.append(j6_idx)
            
            # get global index of root in rigid body state tensor
            root_idx = self.gym.find_actor_rigid_body_index(env_handle, kinova_handle, "base_link", gymapi.DOMAIN_SIM)
            self.root_idxs.append(root_idx)

            ############################### add table ###############################
            table_handle = self.gym.create_actor(env_handle,self.tb_asset,self.tb_pose, "table", i, 0)
            self.actor_handles.append(table_handle)
            table_idx = self.gym.get_actor_index(env_handle,table_handle,gymapi.DOMAIN_SIM)

            # get global index of pan head in rigid body state tensor
            pan_mid_idx = self.gym.find_actor_rigid_body_index(env_handle, kinova_handle, "PanMid", gymapi.DOMAIN_SIM)
            self.pan_mid_idxs.append(pan_mid_idx)

            self.init_pan_mid = [0.6285,      0.0013,      0.5914]
            self.init_pan_ori = gymapi.Quat.from_euler_zyx(0,0,np.pi/2)
            self.initial_pan_orientation.append(self.init_pan_ori)

            ############################### add pita ###############################
            # self.nl_pose.p = gymapi.Vec3(self.tb_pose.p.x + np.random.uniform(-0.06, 0.06), self.tb_pose.p.y + np.random.uniform(-0.06, 0.06), self.tb_dims.z - 0.13) # big pita
            self.nl_pose.p = gymapi.Vec3(self.init_pan_mid[0], self.init_pan_mid[1], self.init_pan_mid[2] + 0.02) # small pita
            self.nl_pose.r = gymapi.Quat.from_euler_zyx(np.random.uniform(-np.pi, np.pi),0,np.pi/2)
            pita_handle = self.gym.create_actor(env_handle, self.nl_asset, self.nl_pose, "pita",i,0)
            nl_color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, pita_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, nl_color)
            self.actor_handles.append(pita_handle)
            pita_idx = self.gym.get_actor_index(env_handle,pita_handle,gymapi.DOMAIN_SIM)
            self.pita_idxs.append(pita_idx)

            # get global index of pita in rigid body state tensor
            pitaRoot_idx = self.gym.find_actor_rigid_body_index(env_handle, pita_handle, "PitaMid", gymapi.DOMAIN_SIM)
            self.pitaRoot_idxs.append(pitaRoot_idx)


            # add instances to list
            self.gym_indices["table"].append(table_idx)
            self.gym_indices["kinova"].append(kinova_idx)
            self.gym_indices["pita"].append(pita_idx)
        # print(self.root_idxs)
        # convert gym indices from list to tensor
        for asset_name, asset_indices in self.gym_indices.items():
            self.gym_indices[asset_name] = torch.tensor(asset_indices, dtype=torch.long, device=self.device)

        ## add sensor for calculating rewards
        self.tip_indices = torch.zeros(len(tip_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.PitaMid_indices = torch.zeros(len(pita_names),dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(tip_names)):
            self.tip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], tip_names[i])
        
        for i in range(len(pita_names)):
            self.PitaMid_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[2], pita_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    # def _draw_debug_vis(self):
    #     """ Draws visualizations for dubugging (slows down simulation a lot).
    #         Default behaviour: draws height measurement points
    #     """
    #     # draw height lines
    #     if not self.terrain.cfg.measure_heights:
    #         return
    #     self.gym.clear_lines(self.viewer)
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)
    #     sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
    #     for i in range(self.num_envs):
    #         base_pos = (self.root_states[i, :3]).cpu().numpy()
    #         heights = self.measured_heights[i].cpu().numpy()
    #         height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
    #         for j in range(heights.shape[0]):
    #             x = height_points[j, 0] + base_pos[0]
    #             y = height_points[j, 1] + base_pos[1]
    #             z = heights[j]
    #             sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
    #             gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    # ----------- helper functions -----------
    def detect_flip(self):
        # Compute the current orientations of the pita and the pan
        current_pan_orientation = self.rb_states[self.pan_mid_idxs, :][..., 3:7]
        current_pita_orientation = self.rb_states[self.pitaRoot_idxs, :][..., 3:7]

        pan_orientation_inv = torch.cat((current_pan_orientation[..., :1], -current_pan_orientation[..., 1:]), dim=-1)

        # Step 2: Quaternion Multiplication to get the relative orientation
        # Manual implementation of quaternion multiplication
        a, b = pan_orientation_inv, current_pita_orientation
        w = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
        xyz = (a[..., 0:1] * b[..., 1:] + b[..., 0:1] * a[..., 1:] +
            torch.cross(a[..., 1:], b[..., 1:], dim=-1))
        current_relative_orientation = torch.cat((w, xyz), dim=-1)

        # Step 3: Convert Quaternion to Euler Angles (Z-Y-X, i.e., yaw-pitch-roll)
        # Manual implementation of quaternion to Euler conversion
        q = current_relative_orientation
        sinr_cosp = 2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2])
        cosr_cosp = 1 - 2 * (q[..., 2]**2 + q[..., 3]**2)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q[..., 0] * q[..., 2] - q[..., 3] * q[..., 1])
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2 * (q[..., 0] * q[..., 1] + q[..., 2] * q[..., 3])
        cosy_cosp = 1 - 2 * (q[..., 1]**2 + q[..., 2]**2)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        # Here we're interested in the yaw (Z-axis rotation)
        current_z = yaw

        # Calculate the z-axis orientation difference in degrees
        z_diff_deg = torch.rad2deg(torch.abs(current_z)) % 360
        z_diff_deg = ((z_diff_deg + 180) % 360) - 180
        z_diff_deg = torch.abs(z_diff_deg)

        # Detect a flip if the z-axis orientation difference exceeds 160 degrees
        flip_detected = z_diff_deg > 150

        # If dealing with batches, check if any flips are detected across the batch
        return flip_detected.any() if flip_detected.dim() > 0 else flip_detected

    def detect_pan_flip(self):
        # Compute the current orientation of the pan
        current_pan_orientation = self.rb_states[self.pan_mid_idxs, :][..., 3:7]
        pan_orientation_init = torch.tensor([self._quat2list(self.initial_pan_orientation[i]) for i in range(len(self.initial_pan_orientation))], dtype=torch.float32, device=self.device)
        # Assuming self.initial_pan_orientation stores the quaternion of the initial pan orientation
        # Inverting the initial orientation
        initial_pan_orientation_inv = torch.cat((pan_orientation_init[..., :1], -pan_orientation_init[..., 1:]), dim=-1)

        # Perform quaternion multiplication in batch
        a, b = initial_pan_orientation_inv, current_pan_orientation
        w = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
        xyz = (a[..., 0:1] * b[..., 1:] + b[..., 0:1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1))
        pan_relative_orientation = torch.cat((w, xyz), dim=-1)

        # Assuming a simple criterion for pan flip based on z component change
        # Here, modify to suit how you define a "flip" for the pan
        # This is a placeholder; replace with your condition
        flipped_condition = torch.abs(pan_relative_orientation[..., 2]) > 0.5*3.14  # Example condition

        return flipped_condition.any()


    #------------ reward functions----------------
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        self.collision_buf = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1., dim=1)
        # return self.reset_buf * ~self.time_out_buf
        return self.reset_buf
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_reach(self):
        # Get the pita head position
        pita_root_pos = self.rb_states[self.pitaRoot_idxs, :][:, :3]
        # Offset for panmid under the pita head by 1cm along the z-axis
        # Assuming the z-axis is the third column in your position vectors and z is up
        # Subtracting 0.01m (or 1cm) to move "under" the pita head in the z direction
        target_pos_panmid = pita_root_pos.clone()
        target_pos_panmid[:, 2] -= 0.001

        # Get the current position of the pan panmid
        pan_mid_pos = self.rb_states[self.pan_mid_idxs, :][:, :3]
        
        # Calculate the distance between the pan panmid and the target position
        dis = torch.sqrt(torch.sum(torch.square(target_pos_panmid - pan_mid_pos), dim=1))
        print("dis of pita and pan mid: ",dis)
        # Reward calculation
        # Using 1 - tanh of distance as the reward function to ensure reward is between 0 and 1
        # The closer the panmid is to the target position, the higher the reward
        r = 1 - torch.tanh(dis)
        
        return r
    
    def _reward_contact_force(self):
        force_vectors_pita = self.contact_forces[:, self.PitaMid_indices, :]
        force_pita_z = force_vectors_pita[..., 2]  # Z component

        # Sum the absolute z-axis forces across the relevant dimension
        force_pita_z_sum = torch.sum(torch.abs(force_pita_z), dim=1)

        # Normalize using a sigmoid function to map the sum to (0, 1)
        # The sigmoid function maps any real-valued number into the range (0, 1),
        # which makes it useful for this kind of soft normalization.
        normalized_force = torch.sigmoid(force_pita_z_sum)

        return normalized_force

    def _reward_flip(self):
        # Reward for successful toss
        flip_detected = self.detect_flip()  # This may return a tensor

        # Ensure flip_detected is properly handled as a tensor
        # Assuming flip_detected can be a tensor with one element in the case of a single environment
        # Or it has elements for each environment in a batched simulation
        if flip_detected.dim() > 0:
            # Handle batched case: flip_detected is a tensor
            # Create a reward tensor of the same shape, filled with zeros
            r = torch.zeros_like(flip_detected, dtype=torch.float)
            # Where a flip is detected, set the reward to 1.0
            r[flip_detected] = 1.0
            # If you're counting flips, ensure this works with batches
            self.flip_pita_times += flip_detected.sum().item()  # Summing flips detected across all environments
        else:
            # Handle single case: flip_detected is a single value tensor or a boolean
            r = 1.0 if flip_detected else 0.0
            self.flip_pita_times += int(flip_detected)  # Increment if flip_detected is True

        print("flip pita times: ", self.flip_pita_times)
        return r


    
