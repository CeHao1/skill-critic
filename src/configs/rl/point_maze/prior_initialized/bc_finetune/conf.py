from src.configs.rl.point_maze.prior_initialized.base_conf import *
from src.policies.prior_policies import ACPriorInitializedPolicy
from src.agents.specific.maze_agents import MazeSACAgent

agent_config.policy = ACPriorInitializedPolicy
configuration.agent = MazeSACAgent
