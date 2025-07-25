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
# Copyright (c) 2023 FORTH, Changda Tian

import glob
import numpy as np

from arm_gym.envs.base.arm_toss_config import ArmTossCfg, ArmTossCfgPPO

MOTION_FILES = glob.glob('datasets/tossing_motions/*')


class gen3AMPTossCfg( ArmTossCfg ):

    class env( ArmTossCfg.env ):
        # num_envs = 2048 
        num_envs = 2400
        include_history_steps = None  # Number of steps of history to include.
        include_history_amp_steps = 5 # Number of steps of history to include

        # num_envs = 128 
        # include_history_steps = None  # Number of steps of history to include.
        # include_history_amp_steps = None # Number of steps of history to include

        num_observations = 66
        num_privileged_obs = 66
        amp_num_observations = 9
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class init_state( ArmTossCfg.init_state ):
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

    class control( ArmTossCfg.control ):
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_1': 100.0, 'joint_2': 100.0,'joint_3': 100.0, 'joint_4': 100.0,'joint_5': 80.0, 'joint_6': 70.0,'joint_7': 20.0}  # [N*m/rad]
        damping = {'joint_1': 0.3, 'joint_2': 0.3,'joint_3': 0.3, 'joint_4': 0.3,'joint_5': 0.2, 'joint_6': 0.2,'joint_7': 0.1}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset( ArmTossCfg.asset ):
        file = '{ARM_GYM_ROOT_DIR}/resources/arms/kinovagen3/mjcf/kinova_pan_toss.xml'
        # pita_file = 'resources/arms/kinovagen3/mjcf/pita_small.xml'
        # pita_offset = 0.1

        pita_file = 'resources/arms/kinovagen3/mjcf/pita.xml'
        pita_offset = 0.004
        target_force = 100
        tip_name = "PanMid"
        pita_name = "PitaMid"
        penalize_contacts_on = ["half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link", "bracelet_link"]
        terminate_after_contacts_on = ["half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link"]
        
        # terminate_after_contacts_on = [
        #     "base", "half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link", "bracelet_link", "hammer"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            hand_pos = 0.005
            pan_mid_pos = 1.0
            pita_pos = 1.0
            angle = 0.0001
            dof_pos = 0.001
            dof_vel = 0.15

    class rewards( ArmTossCfg.rewards ):
        class scales( ArmTossCfg.rewards.scales ):
            termination = -1
            reach = 1e2
            contact_force = 1
            flip = 1e3
            torques = -0.000001
            collision = -0.1
            action_rate = -0.001

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: x y z for hammer head, alpha for hammer head and hand pitch angle.
        resampling_time = 10. # time before command are changed[s]
        class ranges:
            x_range = [-0.1, 0.7] # min max [m]
            y_range = [-0.3, 0.3]   # min max [m]
            z_range = [0, 0.6]   # min max [m]
            a_range = [-1.57, 1.57]

class gen3AMPTossCfgPPO( ArmTossCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunnerToss'
    class algorithm( ArmTossCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( ArmTossCfgPPO.runner ):
        run_name = ''
        experiment_name = 'gen3_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 500000 # number of policy updates

        amp_reward_coef = 1.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.6
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.02] * 15

  