export EXP_DIR=./experiments
export DATA_DIR=./data

# 1. Download the data
# 2. convert data to .h5 format
# see src/envs/wrapper/reskill_fetch_robot/readme.md

# 3. Train skill model


# open-loop spirl skill
python3 src/train/train_skill.py --path=src/configs/skill/reskill_fetch_robot/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_fetch
# when the program finshed, please copy the checkpoint to ./experiments/skill/reskill_fetch_robot/hierarhical/weights

# closed-loop spirl skill
python3 src/train/train_skill.py --path=src/configs/skill/python3 src/train/train_skill.py --path=src/configs/skill/reskill_fetch_robot/hierarchical --val_data_size=160 \
/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_fetch
# when the program finshed, please copy the checkpoint to ./experiments/skill/reskill_fetch_robot/hierarhical_cl/weights


# conditioned decoder skill
python3 src/train/train_skill.py --path=src/configs/skill/python3 src/train/train_skill.py --path=src/configs/skill/reskill_fetch_robot/hierarchical --val_data_size=160 \
/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_fetch
# when the program finshed, please copy the checkpoint to ./experiments/skill/reskill_fetch_robot/hierarhical_cd/weights


