python main.py --prefix "hello" --path "src/configs/example" --cpu_workers 4


export EXP_DIR=./experiments
export DATA_DIR=./data


python3 src/train/train_rl.py --path=src/configs/rl/point_maze/SAC --seed=0 --gpu=0 --prefix=sac  

