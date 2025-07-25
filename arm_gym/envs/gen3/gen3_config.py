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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import glob
from arm_gym.envs.base.arm_config import ArmCfg, ArmCfgPPO

MOTION_FILES = glob.glob('datasets/hammer_motions/*')

class gen3RoughCfg( ArmCfg ):

    class env( ArmCfg.env ):
        num_envs = 2048 
        include_history_steps = None  # Number of steps of history to include.
        include_history_amp_steps = None # Number of steps of history to include

        # num_envs = 64 
        # include_history_steps = 2  # Number of steps of history to include.
        # include_history_amp_steps = None # Number of steps of history to include

        num_observations = 66
        num_privileged_obs = 66
        amp_num_observations = 9
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class init_state( ArmCfg.init_state ):
        pos = [0.0, 0.0, 0.44] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'joint_1': 0,   # [rad]
            'joint_2': 0.4,   # [rad]
            'joint_3': np.pi ,  # [rad]
            'joint_4': -np.pi+1.4,   # [rad]
            'joint_5': 0,     # [rad]
            'joint_6': -1.,   # [rad]
            'joint_7': np.pi/2,     # [rad]
            'right_driver_joint': 0.601,   # [rad]
            'right_coupler_joint': 0,   # [rad]
            'right_spring_link_joint': 0.585,  # [rad]
            'right_follower_joint': -0.585,   # [rad]
            'left_driver_joint': 0.601,   # [rad]
            'left_coupler_joint': 0,   # [rad]
            'left_spring_link_joint': 0.595,  # [rad]
            'left_follower_joint': -0.595,   # [rad]
        }

    class control( ArmCfg.control ):
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_1': 100.0, 'joint_2': 100.0,'joint_3': 100.0, 'joint_4': 100.0,'joint_5': 80.0, 'joint_6': 70.0,'joint_7': 20.0}  # [N*m/rad]
        damping = {'joint_1': 0.3, 'joint_2': 0.3,'joint_3': 0.3, 'joint_4': 0.3,'joint_5': 0.2, 'joint_6': 0.2,'joint_7': 0.1}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset( ArmCfg.asset ):
        file = '{ARM_GYM_ROOT_DIR}/resources/arms/kinovagen3/mjcf/kinova_hammer_isaacsim.xml'
        # nail_file = 'resources/arms/kinovagen3/mjcf/nail_small.xml'
        # nail_offset = 0.1

        shoe_file = 'resources/arms/kinovagen3/mjcf/shoe.xml'
        shoe_offset = 0.01
        nail_file = 'resources/arms/kinovagen3/mjcf/nail.xml'
        nail_offset = 0.13
        target_force = 100
        tip_name = "HammerHead"
        nail_name = "NailHead"
        shoe_name = "shoe"
        penalize_contacts_on = ["half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link", "bracelet_link"]
        terminate_after_contacts_on = ["NailHead","HammerHead","half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link"]
        # terminate_after_contacts_on = [
        #     "base", "half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link", "bracelet_link", "hammer"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( ArmCfg.rewards ):
        # soft_dof_pos_limit = 0.9
        class scales( ArmCfg.rewards.scales ):
            termination = -10
            reach = 1
            knock_force = 1e4
            torques = -0.000001
            collision = -0.1
            action_rate = -0.001

class gen3RoughCfgPPO( ArmCfgPPO ):
    class algorithm( ArmCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( ArmCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_gen3'
        max_iterations = 500000 # number of policy updates

  