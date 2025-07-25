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

from arm_gym import ARM_GYM_ROOT_DIR, ARM_GYM_ENVS_DIR
# from arm_gym.envs.gen3.gen3_config import gen3RoughCfg, gen3RoughCfgPPO
from .base.arm import Arm
from .base.arm_pull import ArmPull
from .base.arm_toss import ArmToss

from .gen3.gen3_config import gen3RoughCfg, gen3RoughCfgPPO
from .gen3.gen3_amp_config import gen3AMPCfg, gen3AMPCfgPPO
from .gen3.gen3_pull_amp_config import gen3AMPPullCfg, gen3AMPPullCfgPPO
from .gen3.gen3_toss_amp_config import gen3AMPTossCfg, gen3AMPTossCfgPPO

import os

from arm_gym.utils.task_registry import task_registry

task_registry.register( "gen3", Arm, gen3RoughCfg(), gen3RoughCfgPPO() )
task_registry.register( "gen3_amp", Arm, gen3AMPCfg(), gen3AMPCfgPPO() )
task_registry.register( "gen3_pull_amp", ArmPull, gen3AMPPullCfg(), gen3AMPPullCfgPPO() )
task_registry.register( "gen3_toss_amp", ArmToss, gen3AMPTossCfg(), gen3AMPTossCfgPPO() )