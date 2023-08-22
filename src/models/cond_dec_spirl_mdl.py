import torch
import torch.nn as nn
import numpy as np

from src.utils.general_utils import batch_apply, ParamDict, AttrDict
from src.utils.pytorch_utils import make_one_hot
from src.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from src.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from src.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from src.modules.variational_inference import MultivariateGaussian
from src.utils.checkpoint_utils import load_by_key, freeze_modules


class CDsrcMdl(SkillPriorMdl):
    """src model with closed-loop, conditional decoder low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size= self.action_size * 2,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()

    def decode(self, z, cond_inputs, steps, inputs=None):
        # the decode only use for training, so here we use deterministic 
        
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)

        output = batch_apply(decode_inputs, self.decoder)
        output = output[..., :self.action_size] # only get the mean
        return output

    def _build_inference_net(self):
        input_size  = self._hp.action_dim 
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _run_inference(self, inputs):
        inf_input = inputs.actions
        return MultivariateGaussian(self.q(inf_input)[:, -1])

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q])
        else:
            super().load_weights_and_freeze()

    @property
    def enc_size(self):
        return self._hp.state_dim

    @property
    def action_size(self):
        return self._hp.action_dim


class TimeIndexCDsrcMDL(CDsrcMdl):
    # decoder input (s, z, idx)

    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae + self._hp.n_rollout_steps, 
                                 output_size= self.action_size * 2,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()

    def decode(self, z, cond_inputs, steps, inputs=None):
        # the decode only use for training, so here we use deterministic 
        
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs) # states

        # idx = torch.tensor(torch.arange(steps), device=self.device)
        idx = torch.tensor(np.arange(steps), device=self.device)
        one_hot = make_one_hot(idx, steps).repeat(seq_enc.shape[0], 1, 1)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1), one_hot), dim=-1)

        # print('='*20)
        # print('seq_enc', seq_enc.shape, 'z', z.shape)
        # print('seq_enc[:, :steps]', seq_enc[:, :steps].shape)
        # print('z repeat', z[:, None].repeat(1, steps, 1).shape)
        # print('decode_inputs',  decode_inputs.shape)

        # # print('idx', idx.shape)
        # print('one hot', one_hot.shape)

        output = batch_apply(decode_inputs, self.decoder)
        output = output[..., :self.action_size]
        return output

class ImageTimeIndexCDsrcMDL(TimeIndexCDsrcMDL, ImageSkillPriorMdl):
    """src model with closed-loop, conditional decoder that operates on image observations."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'prior_input_res': 32,      # input resolution of prior images
            'encoder_ngf': 8,           # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,        # number of prior input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _build_prior_net(self):
        return ImageSkillPriorMdl._build_prior_net(self)

    def _build_inference_net(self):
        self.img_encoder = nn.Sequential(ResizeSpatial(self._hp.prior_input_res),  # encodes image inputs
                                         Encoder(self._updated_encoder_params()),
                                         RemoveSpatial(),)
        return TimeIndexCDsrcMDL._build_inference_net(self)

    def _get_seq_enc(self, inputs):
        # stack input image sequence
        stacked_imgs = torch.cat([inputs.images[:, t:t+inputs.actions.shape[1]]
                                  for t in range(self._hp.n_input_frames)], dim=2)
        # encode stacked seq
        return batch_apply(stacked_imgs, self.img_encoder)

    def _learned_prior_input(self, inputs):
        return ImageSkillPriorMdl._learned_prior_input(self, inputs)

    def _regression_targets(self, inputs):
        return ImageSkillPriorMdl._regression_targets(self, inputs)

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return self.img_encoder(obs)

    @property
    def enc_size(self):
        return self._hp.nz_enc

    @property
    def prior_input_size(self):
        return self.enc_size
    