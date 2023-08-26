from src.configs.rl.point_maze.prior_initialized.base_conf import *
from src.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from src.agents.specific.maze_agents import MazeActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = ACLearnedPriorAugmentedPIPolicy
configuration.agent = MazeActionPriorSACAgent
