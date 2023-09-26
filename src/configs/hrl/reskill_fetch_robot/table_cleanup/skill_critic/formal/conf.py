from src.configs.hrl.reskill_fetch_robot.fetch_robot.skill_critic.conf import *

# agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
agent_config.initial_train_stage = skill_critic_stages.HYBRID

# ll_agent_config.td_schedule_params = AttrDict(p=10.)
# ll_agent_config.td_schedule_params = AttrDict(p=30.)
# ll_agent_config.td_schedule_params = AttrDict(p=50.)
# ll_agent_config.td_schedule_params = AttrDict(p=80.)
ll_agent_config.td_schedule_params = AttrDict(p=100.)

ll_policy_params.manual_log_sigma = [-3] * 4