
import os

from src.configs.skill.point_maze.hierarchical_cl.conf import * 
from src.models.cond_dec_spirl_mdl import ImageTimeIndexCDsrcMDL

from src.components.logger import Logger
from src.utils.general_utils import AttrDict
from src.configs.default_data_configs.point_maze import data_spec
from src.components.evaluator import TopOfNSequenceEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))

configuration = {
    'model': ImageTimeIndexCDsrcMDL,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'point_maze'),
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',

    'batch_size':128,
}
configuration = AttrDict(configuration)

'''
model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
    cond_decode=True,
)
'''
