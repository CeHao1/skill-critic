import os

from src.models.skill_prior_mdl import SkillPriorMdl
from src.components.logger import Logger
from src.utils.general_utils import AttrDict
from src.configs.default_data_configs.reskill_fetch_robot import data_spec
from src.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': SkillPriorMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'reskill_fetch_robot'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',

    'batch_size':128,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=5e-4,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + 1  # flat last action from seq gets cropped
