# Manipulate as Human: Learning Task-oriented Manipulation Skills by Adversarial Motion Priors #

Codebase for the "[Manipulate as Human: Learning Task-oriented Manipulation Skills by Adversarial Motion Priors](https://www.cambridge.org/core/journals/robotica/article/manipulate-as-human-learning-taskoriented-manipulation-skills-by-adversarial-motion-priors/74AC9205DD0BB47D7905E4764B4E61F2)" project(under review). This repository contains the code necessary to ground agent skills using small amounts of reference data (4.5 seconds). All experiments are performed using the robot Kinova Gen3 with Gripper 2f85. This repository is modified based on Alescontrela's [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware) repo and Nikita Rudin's [arm_gym](https://github.com/leggedrobotics/arm_gym) repo, and enables us to train policies using [Isaac Gym](https://developer.nvidia.com/isaac-gym).


### Installation ###
1. Create a new python virtual env with python 3.8 i.e. with conda:
    - `conda create -n arm_amp python==3.8`
    - `conda activate arm_amp`
2. Install pytorch 1.13 with cuda-11.7:
    - `pip3 install torch==1.13.0+cu117 torchvision==0.14.0+cu117 tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 (I don't know whether other Previews will work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
4. Install rsl_rl (PPO implementation)
   - Clone this repository
   -  `cd arm_amp/rsl_rl && pip install -e .` 
5. Install arm_gym
   - `cd ../ && pip install -e .`

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`arm.py`) and a config file (`arm_config.py`). The config file contains two classes: one conatianing all the environment parameters (`ArmCfg`) and one for the training parameters (`ArmCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward. The AMP reward parameters are defined in `ArmCfgPPO`, as well as the path to the reference data.
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.
5. Reference data can be found in the `datasets` folder.

### Usage ###
1. Train:  
```python arm_gym/scripts/train.py --task=gen3_amp```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `arm_amp/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python arm_gym/scripts/play.py --task=gen3_amp```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
3. Record video of a trained policy
```python arm_gym/scripts/record_policy.py --task=gen3_amp```
    - This saves a video in the base directory.


## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{Ma_Tian_Gao_2025, 
        title={Manipulate as human: learning task-oriented manipulation skills by adversarial motion priors}, 
        DOI={10.1017/S0263574725001444}, 
        journal={Robotica}, 
        author={Ma, Ziqi and Tian, Changda and Gao, Yue}, 
        year={2025}, 
        pages={1–13}
        }
```