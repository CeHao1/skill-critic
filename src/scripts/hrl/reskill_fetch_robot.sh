
export EXP_DIR=./experiments
export DATA_DIR=./data

# ================= Table Cleanup =================
# spirl open-loop
python src/train/train_rl.py \
--path=src/configs/hrl/reskill_fetch_robot/table_cleanup/spirl --seed=0 --gpu=0 \
--prefix=nav 

# spirl closed-loop
python src/train/train_rl.py \
--path=src/configs/hrl/reskill_fetch_robot/table_cleanup/spirl_cl --seed=0 --gpu=0 \
--prefix=nav 

# skill critic, warm-start
python src/train/train_rl.py \
--path=src/configs/hrl/reskill_fetch_robot/table_cleanup/skill_critic/warm_start --seed=0 --gpu=0 \
--prefix=nav 

# copy the saved weights to the formal folder, then run the formal
python3 src/train/train_rl.py  \
--path=src/configs/hrl/reskill_fetch_robot/table_cleanup/skill_critic/formal \
--gpu=0 --seed=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=nav 

# ================= Slippery push =================
# spirl open-loop
python src/train/train_rl.py \
--path=src/configs/hrl/reskill_fetch_robot/slippery_push/spirl --seed=0 --gpu=0 \
--prefix=nav 

# spirl closed-loop
python src/train/train_rl.py \
--path=src/configs/hrl/reskill_fetch_robot/slippery_push/spirl_cl --seed=0 --gpu=0 \
--prefix=nav 

# skill critic, warm-start
python src/train/train_rl.py \
--path=src/configs/hrl/reskill_fetch_robot/slippery_push/skill_critic/warm_start --seed=0 --gpu=0 \
--prefix=nav 

# copy the saved weights to the formal folder, then run the formal
python3 src/train/train_rl.py  \
--path=src/configs/hrl/reskill_fetch_robot/slippery_push/skill_critic/formal \
--gpu=0 --seed=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=nav 
