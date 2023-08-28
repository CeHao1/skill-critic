import os
import copy

from src.utils.general_utils import AttrDict
from src.agents.joint_agent import JointAgent, skill_critic_stages
from src.policies.critic import SplitObsMLPCritic, MLPCritic
from src.samplers.sampler import ACMultiImageAugmentedHierarchicalSampler
from src.samplers.replay_buffer import UniformReplayBuffer
from src.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from src.envs.wrapper.maze import ACRandMaze0S40Env
from src.agents.ll_agent import MazeLLInheritAgent
from src.policies.cd_model_policy import AC_DecoderRegu_TimeIndexedCDMdlPolicy
from src.agents.specific.maze_agents import MazeHLInheritAgent
from src.models.cond_dec_spirl_mdl import ImageTimeIndexCDSPiRLMDL
from src.configs.default_data_configs.point_maze import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'skill critic on the maze env'

configuration = {
    'seed': 42,
    'agent': JointAgent,
    'environment': ACRandMaze0S40Env,
    'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'data_dir': '.',
    # "use_update_after_sampling": True,
    'num_epochs': 500,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 5e3,
}
configuration = AttrDict(configuration)

# Replay Buffer
hl_replay_params = AttrDict(
    capacity = 1e5,
    dump_replay=False,
)

ll_replay_params = AttrDict(
    capacity = 5e5,
    dump_replay=False,
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
    clip_q_target=False,
)

######================= Low-Level ===============######
# LL Policy Model
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    nz_vae = 10,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ImageTimeIndexCDSPiRLMDL, 
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill/point_maze/hierarchical_cd"),
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    output_dim=1,
    action_input=True,
    
    discard_part = 'mid', # obs = (s+z+t) + a //remove image in the middle
    unused_obs_start = data_spec.state_dim,
    unused_obs_size=ll_model_params.prior_input_res **2 * 3 * ll_model_params.n_input_frames,

)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=AC_DecoderRegu_TimeIndexedCDMdlPolicy, 
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,
    # obs(s + z + t) + a = 4 + 10 + 10 + 2
    critic_params=ll_critic_params,
    replay_params=ll_replay_params,
))

######=============== High-Level ===============########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim, # QHL(s, z), no K
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    unused_obs_size=ll_model_params.prior_input_res **2 * 3 * ll_model_params.n_input_frames,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=ACLearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=hl_critic_params,
    replay_params=hl_replay_params,
    
    td_schedule_params=AttrDict(p=1.),
))

#####========== Joint Agent =======#######
agent_config = AttrDict(
    hl_agent=MazeHLInheritAgent, 
    hl_agent_params=hl_agent_config,
    ll_agent=MazeLLInheritAgent,  
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=False,
    update_hl=True,
    update_ll=True,
    
    initial_train_stage = skill_critic_stages.HL_TRAIN
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)
