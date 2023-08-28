from src.configs.hrl.point_maze.maze_spirl.skill_critic.base_conf import *

# agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
agent_config.initial_train_stage = skill_critic_stages.HYBRID

# ll_agent_config.td_schedule_params = AttrDict(p=5.)
# ll_agent_config.td_schedule_params = AttrDict(p=10.)
# ll_agent_config.td_schedule_params = AttrDict(p=20.)
# ll_agent_config.td_schedule_params = AttrDict(p=50.)
ll_agent_config.td_schedule_params = AttrDict(p=80.)



# ll_policy_params.manual_log_sigma = [-1, -1]
# ll_policy_params.manual_log_sigma = [-2, -2]
ll_policy_params.manual_log_sigma = [-3, -3]
# ll_policy_params.manual_log_sigma = [-5, -5]