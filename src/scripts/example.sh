python main.py --prefix "hello" --path "src/configs/example" --cpu_workers 4


export EXP_DIR=./experiments
export DATA_DIR=./data


mpiexec -n 15 python3 src/train/train_rl.py \
--path=src/configs/rl/point_maze/SAC --seed=0 --gpu=0 --prefix=sac_n15  


mpiexec -n 10 python3 src/train/train_rl.py \
--path=src/configs/rl/point_maze/maze_nav/SAC --seed=0 --gpu=0 \
--prefix=maze_nav01
