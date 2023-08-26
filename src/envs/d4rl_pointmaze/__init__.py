from src.envs.d4rl_pointmaze.maze_layouts import rand_layout
from src.envs.d4rl_pointmaze.maze_model import MazeEnv
from src.envs.d4rl_pointmaze.maze_strings import maze_name_space
from gym.envs.registration import register



register(
    id='maze2d-randMaze0S40-ac-v0',
    entry_point='src.envs.d4rl_pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=0, size=40),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)

##################### SEMANTIC MAZE LAYOUTS #############

register(
    id='maze2d-mMaze1-v0',
    entry_point='src.envs.d4rl_pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': maze_name_space['m_maze1'],
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-mMaze2-v0',
    entry_point='src.envs.d4rl_pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': maze_name_space['m_maze2'],
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)

