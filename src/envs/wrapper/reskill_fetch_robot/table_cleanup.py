import numpy as np

import reskill.rl.envs
from src.envs.wrapper.environment import GymEnv
from src.utils.py_utils import ParamDict, AttrDict

class TableCleanup(GymEnv):
    """Tiny wrapper around GymEnv for robot tasks."""
    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "FetchCleanUp-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        return info

    def _postprocess_info(self, info):
        return info

    def _wrap_observation(self, obs):
        out = np.concatenate((obs["observation"], obs["desired_goal"]))
        return out


        