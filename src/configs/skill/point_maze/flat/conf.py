import os

from src.models.bc_mdl import ImageBCMdl
from src.utils.general_utils import AttrDict
from src.configs.default_data_configs.point_maze import data_spec
from src.components.evaluator import DummyEvaluator
from src.components.logger import Logger


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageBCMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'point_maze'),
    'epoch_cycles_train': 4,
    'evaluator': DummyEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    n_input_frames=2,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 1 + 1 + (model_config.n_input_frames - 1)
