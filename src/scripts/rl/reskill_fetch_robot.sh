export EXP_DIR=./experiments
export DATA_DIR=./data


# table cleanup
python src/train/train_rl.py \
--path=src/configs/rl/reskill_fetch_robot/table_cleanup/SAC --seed=0 --gpu=0 \
--prefix=table_cleanup

# slippery push
python src/train/train_rl.py \
--path=src/configs/rl/reskill_fetch_robot/slippery_push/SAC --seed=0 --gpu=0 \
--prefix=slippery_push