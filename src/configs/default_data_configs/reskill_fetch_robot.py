from src.utils.general_utils import AttrDict
from src.demo.data_loader import GlobalSplitVideoDataset

data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    n_actions=4,
    state_dim=25,
    crop_rand_subseq=True,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
)
data_spec.max_seq_len = 280