export EXP_DIR=./experiments
export DATA_DIR=./data


# ================== Maze Navigation ==================
# SAC
python src/train/train_rl.py \
--path=src/configs/rl/point_maze/maze_nav/SAC --seed=0 --gpu=0 \
--prefix=nav 

# BC+finetune
python src/train/train_rl.py \
--path=src/configs/rl/point_maze/maze_nav/bc_finetune --seed=0 --gpu=0 \
--prefix=nav 


# ================== Maze Trajectory planning ==================
# SAC
python src/train/train_rl.py \
--path=src/configs/rl/point_maze/maze_traj/SAC --seed=0 --gpu=0 \
--prefix=traj 

# BC+finetune
python src/train/train_rl.py \
--path=src/configs/rl/point_maze/maze_traj/bc_finetune --seed=0 --gpu=0 \
--prefix=traj 