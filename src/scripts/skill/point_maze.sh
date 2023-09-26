export EXP_DIR=./experiments
export DATA_DIR=./data

# flat
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/flat --val_data_size=160 \
--gpu=0 --prefix=flat_maze
# when the program finshed, please copy the checkpoint to ./experiments/skill/point_maze/flat/weights

# open-loop spirl skill
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/hierarchical --val_data_size=160 \
--gpu=0 --prefix=ol_maze
# when the program finshed, please copy the checkpoint to ./experiments/skill/point_maze/hierarchical/weights

# closed-loop spirl skill
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/hierarchical_cl --val_data_size=160 \
--gpu=0 --prefix=cl_maze
# when the program finshed, please copy the checkpoint to ./experiments/skill/point_maze/hierarchical_cl/weights


# conditioned decoder skill
python3 src/train/train_skill.py --path=src/configs/skill/point_maze/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_maze
# when the program finshed, please copy the checkpoint to ./experiments/skill/point_maze/hierarchical_cd/weights

