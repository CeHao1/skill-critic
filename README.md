# Skill-Critic: Refining Learned Skills for Hierarchical Reinforcement Learning

#### [[Project Website]](https://sites.google.com/view/skill-critic) [[Paper]](https://arxiv.org/abs/2306.08388)

Ce Hao, Catherine Weaver,  Chen Tang, Kenta Kawamoto, Masayoshi Tomizuka, Wei Zhan

## Requirements
- python 3.8+
- mujoco 2.0 (for RL experiments)
- Ubuntu 18.04+

## Installation Instructions
Create a virtual environment (e.g. conda) with Python>=3.8 and install the following requirements
```
# download the repo
git clone https://github.com/CeHao1/skill-critic.git 
cd skill-critic

# install requirements and package
pip3 install -r requirements.txt
pip3 install -e .
```

To manage the data and checkpoints, we recommand to put them in the current directory as,
```
mkdir ./experiments
mkdir ./data
export EXP_DIR=./experiments
export DATA_DIR=./data
```

## Data collection
For Maze experiments, please download the demonstration data from [SPiRL](https://github.com/clvrai/spirl/tree/master/spirl/data), at [drive](https://drive.google.com/uc?id=1pXM-EDCwFrfgUjxITBsR48FqW9gMoXYZ).
Then place the them in the ```./data/point_maze```.   
For Fetch robot experiments, please download demonstration data from [ReSkill](https://github.com/krishanrana/reskill/tree/main), at [drive](https://drive.google.com/drive/folders/1yTr_6fc-sHXK_CZkm8QIRTV9VgWxKpOE). Then use [converter.py](src/envs/wrapper/reskill_fetch_robot/convert.py) to convert the data format and finally put then in the ```./data/reskill_fetch_robot```.

## Example commands

All results will be written to [WandB](https://www.wandb.com/). Before running any of the commands below, 
create an account and then change the WandB entity and project name at the top of [train_skill.py](src/train/train_skill.py) and
[train_rl.py](src/train/train_rl.py) to match your account.

### Train skill prior


### Train SPiRL baseline



### Train Skill-critic

