
import numpy as np
from src.utils.general_utils import ParamDict, AttrDict
from src.utils.file_operation import *

from tqdm import tqdm
import os

# raw data from ReSkill
fname = 'reskill/dataset/fetch_block_40000/demos.npy'

# output directory
rollout_dir = os.path.join(os.environ['EXP_DIR'], '/table_cleanup/batch_0')
seqs = np.load(fname, allow_pickle=True)


for idx in tqdm(range(len(seqs))):
    seq = seqs[idx]
    action = np.array(seq.actions)
    observation = np.array(seq.obs)
    image = np.zeros((len(action), 1, 1, 3))
    done = [False for _ in image]
    done[-1] = True

    done = np.array(done)
    episode = AttrDict( observation=observation, action=action , image = image, done=done)
    save_rollout(str(idx), episode, rollout_dir)