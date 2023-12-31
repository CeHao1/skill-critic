import os
import copy

from src.utils.general_utils import AttrDict
from src.agents.agent import FixedIntervalHierarchicalAgent
from src.policies.mlp_policies import SplitObsMLPPolicy
from src.policies.critic import SplitObsMLPCritic
from src.envs.wrapper.maze import ACRandMaze0S40Env
from src.samplers.sampler import ACMultiImageAugmentedHierarchicalSampler
from src.samplers.replay_buffer import UniformReplayBuffer
from src.agents.ac_agent import SACAgent
from src.models.skill_prior_mdl import ImageSkillPriorMdl
from src.configs.default_data_configs.point_maze import data_spec
from src.agents.specific.maze_agents import MazeACSkillSpaceAgent


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the maze env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': ACRandMaze0S40Env,
    'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 30,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 100000,
    'n_warmup_steps': 5e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=1e-2,
    n_input_frames=2,
    prior_input_res=data_spec.res,
    nz_vae=10,
    n_rollout_steps=10,
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    model=ImageSkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill/point_maze/hierarchical"),
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=10,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    unused_obs_size=ll_model_params.prior_input_res **2 * 3 * ll_model_params.n_input_frames,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    unused_obs_size=hl_policy_params.unused_obs_size,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=SplitObsMLPPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=hl_critic_params,
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=SACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=MazeACSkillSpaceAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=False,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)

# reduce replay capacity because we are training image-based, do not dump (too large)
from src.samplers.replay_buffer import SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = ll_model_params.prior_input_res**2*3 * 2 + \
                                                             hl_agent_config.policy_params.action_dim   # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False

