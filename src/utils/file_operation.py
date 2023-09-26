import pickle
import os
import joblib
import h5py

import numpy as np 

def save_file(file_path, file):
    joblib.dump(file, file_path)
        
def load_file(file_path):
    file = joblib.load(file_path)
    return file

def delete_file(file_dir, file_name, data_type=''):
    make_dir(file_dir)
    name_space = os.listdir(file_dir)
    if_exist_flag = False
    for names in name_space:
        if file_name in names:
            os.remove(file_dir + '/' + names)
            print('DELETE; TYPE: {};  delete {} in {}'.format(data_type, file_name, file_dir))
            if_exist_flag = True
    if not if_exist_flag:
        print(file_name, 'doet NOT exists in ', file_dir)

def make_dir(file_dir):
     isExists=os.path.exists(file_dir)
     if not isExists:
         os.makedirs(file_dir)

def search_file(file_dir, file_name):
    make_dir(file_dir)
    file_names = os.listdir(file_dir)
    if file_name in file_names:
        return True
    else:
        return False


def save_rollout(index, episode, file_dir= './rollout/'):
    make_dir(file_dir)
    save_path = os.path.join(file_dir, "rollout_{}.h5".format(index))

    f = h5py.File(save_path, "w")
    f.create_dataset("traj_per_file", data=1)

    # store trajectory info in traj0 group
    traj_data = f.create_group("traj0")
    traj_data.create_dataset("states", data=np.array(episode.observation))
    traj_data.create_dataset("images", data=np.array(episode.image, dtype=np.uint8))
    traj_data.create_dataset("actions", data=np.array(episode.action))

    terminals = np.array(episode.done)
    if np.sum(terminals) == 0:
        terminals[-1] = True

    # build pad-mask that indicates how long sequence is
    is_terminal_idxs = np.nonzero(terminals)[0]
    pad_mask = np.zeros((len(terminals),))
    pad_mask[:is_terminal_idxs[0]] = 1.
    traj_data.create_dataset("pad_mask", data=pad_mask)

    f.close()

def load_rollout(index, file_dir= './rollout/'):
    # file_dir = './rollout/'
    make_dir(file_dir)
    save_path = file_dir + "rollout_{}.h5".format(index)

    F =  h5py.File(save_path, 'r') 
    return F