from src.configs.rl.reskil_fetch_robot.base_conf import * 

from src.envs.wrapper.reskill_fetch_robot.slippery_push import SlipperyPush
configuration.environment = SlipperyPush
