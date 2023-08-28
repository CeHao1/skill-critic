export EXP_DIR=./experiments
export DATA_DIR=./data

# =================  Maze in SPiRL =================

# spirl open-loop
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_spirl/spirl --seed=0 --gpu=0 \
--prefix=spirl 

# spirl closed-loop
python src/train/train_rl.py \
--path=src/configs/hrl/point_maze/maze_spirl/spirl_cl --seed=0 --gpu=0 \
--prefix=spirl 

# skill critic, warm-start
