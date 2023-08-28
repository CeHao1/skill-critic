import torch
import numpy as np

from src.utils.general_utils import AttrDict, ParamDict
from src.utils.pytorch_utils import no_batchnorm_update
from src.policies.basic_policies import Policy
from src.agents.agent import BaseAgent
from src.modules.variational_inference import MultivariateGaussian, mc_kl_divergence

class CDModelPolicy(Policy):

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.update_model_params(self._hp.policy_model_params)
        super().__init__()
        self.steps_since_hl, self.last_z = np.Inf, None


    def _default_hparams(self):
        default_dict = ParamDict({
            'policy_model': None,              # policy model class
            'policy_model_params': None,       # parameters for the policy model
            'policy_model_checkpoint': None,   # checkpoint path of the policy model
            'policy_model_epoch': 'latest',    # epoch that checkpoint should be loaded for (defaults to latest)
            'load_weights': True,              # optionally allows to *not* load the weights (ie train from scratch)
            'initial_log_sigma': -50,          # initial log sigma of policy dist (since model is deterministic)
        })

        # we can set manual log sigma, 'manual_log_sigma'
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs):
        with no_batchnorm_update(self):     # BN updates harm the initialized policy
            return super().forward(obs)

    def _build_network(self):
        net = self._hp.policy_model(self._hp.policy_model_params, None)
        if self._hp.load_weights:
            BaseAgent.load_model_weights(net, self._hp.policy_model_checkpoint, self._hp.policy_model_epoch)  

        if 'manual_log_sigma' in self._hp:
            print('!!! use manual log sigma to initialize cd model policy', self._hp.manual_log_sigma)
            init_log_sigma = np.array(self._hp.manual_log_sigma, dtype=np.float32)
            assert init_log_sigma.shape[0] == self.action_dim
        else:
            init_log_sigma = self._hp.initial_log_sigma * np.ones(self.action_dim, dtype=np.float32)

        self._log_sigma = torch.tensor(init_log_sigma, device=self.device, requires_grad=False)   

        if 'min_log_sigma' in self._hp:
            self._min_log_sigma = torch.tensor(self._hp.min_log_sigma,device=self.device, requires_grad=False)

        return net

    def _compute_action_dist(self, obs):
        assert len(obs.shape) == 2
        split_obs = self._split_obs(obs)
        concatenate_obs = self._get_concatenate_obs(split_obs)

        # concatenate_obs = obs
        act = self.net.decoder(concatenate_obs)
        return self._get_constrainted_distribution(act)

    def _get_constrainted_distribution(self, act, deterministic=False):
        act_mean = act[..., : self.net.action_size]
        act_log_std = act[..., self.net.action_size :]
        log_sigma =  act_log_std + self._log_sigma[None].repeat(act.shape[0], 1)

        if 'min_log_sigma' in self._hp:
            min_log_sigma = self._min_log_sigma[None].repeat(act.shape[0], 1)
            log_sigma = torch.clamp(log_sigma, min=min_log_sigma)

        if deterministic:
            log_sigma = torch.ones_like(log_sigma) * -50

        return MultivariateGaussian(mu=act_mean, log_sigma=log_sigma)

    def _get_concatenate_obs(self, split_obs):
        return torch.cat((split_obs.cond_input, split_obs.z), dim=-1)

    def sample_rand(self, obs):
        if len(obs.shape) == 1:
            output_dict = self.forward(obs[None])
            output_dict.action = output_dict.action[0]
            return output_dict
        return self.forward(obs)    # for prior-initialized policy we run policy directly for rand sampling from prior

    def reset(self):
        self.steps_since_hl, self.last_z = np.Inf, None

    def _split_obs(self, obs):
        assert obs.shape[1] == self.net.state_dim + self.net.latent_dim
        return AttrDict(
            cond_input=obs[:, :-self.net.latent_dim],   # condition decoding on state
            z=obs[:, -self.net.latent_dim:],
        )

    @staticmethod
    def update_model_params(params):
        params.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        params.batch_size = 1  # run only single-element batches for forward pass

    @property
    def horizon(self):
        return self._hp.policy_model_params.n_rollout_steps

class TimeIndexedCDMdlPolicy(CDModelPolicy):
    def _split_obs(self, obs):
        assert obs.shape[1] == self.net.state_dim + self.net.latent_dim + self.net.n_rollout_steps
        return AttrDict(
            cond_input=obs[:, :self.net.state_dim],   # condition decoding on state
            z=obs[:, self.net.state_dim: self.net.state_dim + self.net.latent_dim],
            time_index = obs[:, self.net.state_dim + self.net.latent_dim:] # or [:, -self.net.n_rollout_steps:]
        )

    def _get_concatenate_obs(self, split_obs):
        return torch.cat((split_obs.cond_input, split_obs.z, split_obs.time_index), dim=-1)


class DecoderRegu_TimeIndexedCDMdlPolicy(TimeIndexedCDMdlPolicy):
    def __init__(self, config):
        super().__init__(config)
        # add the decoder net
        self.decoder_net = self._hp.policy_model(self._hp.policy_model_params, None)
        BaseAgent.load_model_weights(self.decoder_net, self._hp.policy_model_checkpoint, self._hp.policy_model_epoch)  

    def _default_hparams(self):
        default_dict = ParamDict({
            'num_mc_samples': 10,             # number of samples for monte-carlo KL estimate
            'max_divergence_range': 100,   # range at which prior divergence gets clipped
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs):
        policy_output = super().forward(obs)
        if not self._rollout_mode:
            raw_decoder_divergence, policy_output.prior_dist = self._compute_decoder_divergence(policy_output, obs)
            policy_output.prior_divergence = self.clamp_divergence(raw_decoder_divergence)

        return policy_output

    def clamp_divergence(self, divergence):
        return torch.clamp(divergence, -self._hp.max_divergence_range, self._hp.max_divergence_range)

    def _compute_decoder_divergence(self, policy_output, obs):
        split_obs = self._split_obs(obs)
        concatenate_obs = self._get_concatenate_obs(split_obs)
        
        with no_batchnorm_update(self.decoder_net): 
            act = self.decoder_net.decoder(concatenate_obs).detach()
            decoder_dist = self._get_constrainted_distribution(act, deterministic=False)
            return self._mc_divergence(policy_output, decoder_dist), decoder_dist

    def _mc_divergence(self, policy_output, decoder_dist):
        return mc_kl_divergence(policy_output.dist, decoder_dist, n_samples=self._hp.num_mc_samples)


class AC_DecoderRegu_TimeIndexedCDMdlPolicy(DecoderRegu_TimeIndexedCDMdlPolicy):
    def _split_obs(self, obs):
        # the obs = state + image (32x32 x 3 color x 2 time)
        dim_image = self.net.resolution**2 * 3 * 2 
        obs_dim = self.net.state_dim + dim_image
        assert obs.shape[1] == obs_dim + self.net.latent_dim + self.net.n_rollout_steps

        unflattened_obs = self.net.unflatten_obs(obs[:, :obs_dim])
        return AttrDict(
            cond_input=self.net.enc_obs(unflattened_obs.prior_obs),   # condition decoding on image
            z=obs[:, obs_dim : obs_dim + self.net.latent_dim],
            time_index = obs[:, obs_dim + self.net.latent_dim:] # or [:, -self.net.n_rollout_steps:]
        )

