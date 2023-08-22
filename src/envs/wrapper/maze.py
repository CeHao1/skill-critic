import numpy as np

import src.envs.d4rl_pointmaze 
from src.envs.wrapper.environment import GymEnv
from src.utils.general_utils import ParamDict, AttrDict


class MazeEnv(GymEnv):
    """Shallow wrapper around gym env for maze envs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _default_hparams(self):
        default_dict = ParamDict({
        })
        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        super().reset()
        if self.TARGET_POS is not None and self.START_POS is not None:
            self._env.set_target(self.TARGET_POS)
            self._env.reset_to_location(self.START_POS)
        self._env.render(mode='rgb_array')  # these are necessary to make sure new state is rendered on first frame
        obs, _, _, _ = self._env.step(np.zeros_like(self._env.action_space.sample()))
        return self._wrap_observation(obs)

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, info     # casting reward to float64 is important for getting shape later


class ACRandMaze0S40Env(MazeEnv):
    START_POS = np.array([10., 24.])
    # TARGET_POS = np.array([18., 8.])
    # TARGET_POS = np.array([6., 16.]) # level-1 easiest one
    TARGET_POS = np.array([20., 16.]) # level-2

    VIS_RANGE = [[-3, 43], [-3, 43]]

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-randMaze0S40-ac-v0",
        })
        return super()._default_hparams().overwrite(default_dict)


class ACmMaze1(MazeEnv):  
    # v7
    # TARGET_POS = np.array([13, 9]) # right mid
    TARGET_POS = np.array([9, 13]) # good and selected, top mid
    # TARGET_POS = np.array([9, 10]) # center
    # TARGET_POS = np.array([1, 17]) # top left
    
    START_POS = np.array([1,1]) # bottom left
    # START_POS = np.array([9,10]) # middle
    
    VIS_RANGE = [[-1, 20], [-1, 21]]
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-mMaze1-v0",
        })
        return super()._default_hparams().overwrite(default_dict)

class ACmMaze2(MazeEnv):  
    # 
    # TARGET_POS = np.array([3, 10]) # selected
    TARGET_POS = np.array([5, 1]) # 
    
    START_POS = np.array([1,1]) #
    VIS_RANGE = [[-1, 19], [-1, 18]]
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-mMaze2-v0",
        })
        return super()._default_hparams().overwrite(default_dict)
    
class ACmMaze3(MazeEnv):
    
    TARGET_POS = np.array([9,8]) # 
    START_POS = np.array([1,1]) #
    
    
    # TARGET_POS = np.array([8,8]) # 
    # START_POS = np.array([2,1]) #
    
    
    VIS_RANGE = [[0, 10], [0, 9]]
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'name': "maze2d-mMaze3-v0",
        })
        return super()._default_hparams().overwrite(default_dict)