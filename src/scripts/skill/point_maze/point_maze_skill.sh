export EXP_DIR=./experiments
export DATA_DIR=./data

# flat
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/flat --val_data_size=160 \
--gpu=0 --prefix=flat_maze

# open-loop spirl skill
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_maze

# closed-loop spirl skill
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_maze


# conditioned decoder skill
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_maze

