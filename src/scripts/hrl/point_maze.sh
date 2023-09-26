export EXP_DIR=./experiments
export DATA_DIR=./data

# ================= Maze navigation =================
# spirl open-loop
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_nav/spirl --seed=0 --gpu=0 \
--prefix=nav 

# spirl closed-loop
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_nav/spirl_cl --seed=0 --gpu=0 \
--prefix=nav 

# skill critic, warm-start
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_nav/skill_critic/warm_start --seed=0 --gpu=0 \
--prefix=nav 

# copy the saved weights to the formal folder, then run the formal
python3 src/train/train_rl.py  \
--path=src/configs/hrl/point_maze/maze_nav/skill_critic/formal \
--gpu=0 --seed=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=nav 


# ================= Maze Trajectory Planning ===========

# spirl open-loop
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_traj/spirl --seed=0 --gpu=0 \
--prefix=traj 

# spirl closed-loop
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_traj/spirl_cl --seed=0 --gpu=0 \
--prefix=traj 

# skill critic, warm-start
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_traj/skill_critic/warm_start --seed=0 --gpu=0 \
--prefix=traj 

# copy the saved weights to the formal folder, then run the formal
python3 src/train/train_rl.py  \
--path=src/configs/hrl/point_maze/maze_traj/skill_critic/formal \
--gpu=0 --seed=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=traj 

