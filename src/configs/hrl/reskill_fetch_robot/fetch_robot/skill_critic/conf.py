import os
import copy


from src.utils.general_utils import AttrDict
from src.samplers.replay_buffer import UniformReplayBuffer

from src.envs.wrapper.reskill_fetch_robot.table_cleanup import TableCleanup
from src.configs.default_data_configs.reskill_fetch_robot import data_spec
from src.samplers.sampler import HierarchicalSampler

from src.models.cond_dec_spirl_mdl import TimeIndexCDSPiRLMDL
from src.policies.cd_model_policy import DecoderRegu_TimeIndexedCDMdlPolicy
from src.policies.prior_policies import LearnedPriorAugmentedPIPolicy
from src.policies.critic import MLPCritic
from src.agents.joint_agent import JointAgent, skill_critic_stages
from src.agents.hl_agent  import HLInheritAgent
from src.agents.ll_agent  import LLInheritAgent

configuration = {
    'seed': 42,
    'agent': JointAgent,
    'environment': TableCleanup,
    'sampler': HierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 50,
    'max_rollout_len': 50,
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

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######


# model
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    nz_vae = 10,
    n_rollout_steps=10,

    cond_decode = True,
)


# policy
ll_policy_params = AttrDict(
    policy_model = TimeIndexCDSPiRLMDL,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], 
                "skill/reskill_fetch_robot/hierarchical_cd"),

    # manual_log_sigma = [-5]* data_spec.n_actions,
)
ll_policy_params.update(ll_model_params)

# critic
ll_critic_params = AttrDict(
    input_dim=data_spec.state_dim + ll_model_params.nz_vae + ll_model_params.n_rollout_steps,
    action_dim=data_spec.n_actions,
    output_dim=1,
    n_layers=5,  # number of policy network layer
    nz_mid=256,
    action_input=True,
)

# agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=DecoderRegu_TimeIndexedCDMdlPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,                
    critic_params=ll_critic_params,
))



###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=ll_model_params.nz_vae,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    nz_mid=256,
    n_layers=5,

    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=5,  # number of policy network laye
    nz_mid=256,
    action_input=True,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=LearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=MLPCritic,
    critic_params=hl_critic_params,
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=HLInheritAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=LLInheritAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,


    log_video_caption=True,

    initial_train_stage = skill_critic_stages.HL_TRAIN
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
)


# ====================

agent_config.initial_train_stage = skill_critic_stages.HL_TRAIN
# agent_config.initial_train_stage = skill_critic_stages.HYBRID

# ll_agent_config.td_schedule_params = AttrDict(p=10.)
# ll_agent_config.td_schedule_params = AttrDict(p=30.)
# ll_agent_config.td_schedule_params = AttrDict(p=50.)
# ll_agent_config.td_schedule_params = AttrDict(p=80.)
# ll_agent_config.td_schedule_params = AttrDict(p=100.)

# ll_policy_params.manual_log_sigma = [-3] * 4