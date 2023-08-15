from src.utils.general_utils import ParamDict, map_dict, AttrDict
from src.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, parse_one_hot, make_one_hot

from src.agents.ac_agent import SACAgent
from src.agents.prior_sac_agent import ActionPriorSACAgent

import matplotlib.pyplot as plt
import numpy as np
import torch
import copy



class LLInheritAgent(ActionPriorSACAgent):
    def __init__(self, config):
        ActionPriorSACAgent.__init__(self, config)
        # critic is Qa(s,z,k,a)
        # policy is PIa(a|s,z,k)
        self._update_ll_policy_flag = True
        self._update_ll_q_flag = True
        
    def fast_assign_flags(self, flags):
        self._update_ll_policy_flag = flags[0]
        self._update_ll_q_flag = flags[1]

    def update_by_hl_agent(self, hl_agent):
        self.hl_agent = hl_agent
        self.hl_critics = hl_agent.critics
        self.hl_critic_targets = hl_agent.critic_targets
        self.hl_critic_opts = hl_agent.critic_opts

    def update(self, experience_batch=None):

        # logging
        info = AttrDict(    # losses
        )

        if not (self._update_ll_policy_flag or self._update_ll_q_flag):
            return info

        if experience_batch is not None:
            self.add_experience(experience_batch)

        # sample batch and normalize
        experience_batch = self._sample_experience()
        experience_batch = self._normalize_batch(experience_batch)
        experience_batch = map2torch(experience_batch, self._hp.device)
        experience_batch = self._preprocess_experience(experience_batch)


        # (1) LL policy loss
        policy_output = self._run_policy(experience_batch.observation)
        
        # update alpha
        alpha_loss = self._update_alpha(experience_batch, policy_output)
        # info.update(AttrDict(ll_alpha_loss=alpha_loss,))

        if self._update_ll_policy_flag:
            policy_loss = self._compute_policy_loss(experience_batch, policy_output)

        # (2) Qa(s,z,k,a) loss, Qa_target
        if self._update_ll_q_flag:
            ll_q_target, q_hl_next, q_ll_next, u_next = self._compute_ll_q_target(experience_batch)
            ll_critic_loss, ll_qs = self._compute_critic_loss(experience_batch, ll_q_target)           

        # (4) update loss
        # policy
        if self._update_ll_policy_flag:
            self._perform_update(policy_loss, self.policy_opt, self.policy)
            info.update(AttrDict(
                ll_policy_loss=policy_loss,
            ))

        # ll
        if self._update_ll_q_flag:
            [self._perform_update(critic_loss, critic_opt, critic)
                    for critic_loss, critic_opt, critic in zip(ll_critic_loss, self.critic_opts, self.critics)]
            info.update(AttrDict(
                # ll_q_target=ll_q_target.mean(),
                # ll_q_1=ll_qs[0].mean(),
                # ll_q_2=ll_qs[1].mean(),
                ll_q_hl_next=q_hl_next.mean(), 
                ll_q_ll_next=q_ll_next.mean(), 
                u_next=u_next.mean(),
                qa_critic_loss_1=ll_critic_loss[0],
                qa_critic_loss_2=ll_critic_loss[1],
            ))

        # (5) soft update targets
        if self._update_ll_q_flag:
            [self._soft_update_target_network(critic_target, critic)
                    for critic_target, critic in zip(self.critic_targets, self.critics)]

        
        # if self._update_steps % 100 == 0:
        #     info.update(AttrDict(       # gradient norms
        #         policy_grad_norm=avg_grad_norm(self.policy),
        #         critic_1_grad_norm=avg_grad_norm(self.critics[0]),
        #         critic_2_grad_norm=avg_grad_norm(self.critics[1]),
        #     ))

        info.update(AttrDict(       # misc
            ll_alpha=self.alpha,
            ll_pi_KLD=policy_output.prior_divergence.mean(),
            ll_policy_entropy=policy_output.dist.entropy().mean(),
            ll_avg_sigma = policy_output.dist.sigma.mean(),
            ll_target_divergence = self._target_divergence(self.schedule_steps),
            ll_avg_reward=experience_batch.reward.mean(),
        ))
        info.update(self._aux_info(experience_batch, policy_output))
        info = map_dict(ten2ar, info)

        self._update_steps += 1

        return info

    # ================================ ll critic ================================
    def _compute_ll_q_target(self, experience_batch):
        # Qa(s,z,k,a) = r + gamma * U_next
        # U_next = [1-beta] * q_ll_next + [beta] * q_hl_next

        with torch.no_grad():
            # (0) use s to generate hl policy
            split_obs = self._split_obs(experience_batch.observation_next)
            hl_policy_output_next = self.hl_agent._run_policy(self._get_hl_obs(split_obs))
            ll_policy_output_next = self._run_policy(experience_batch.observation_next)

            # (1) V_next
            q_hl_next, q_ll_next = self._compute_next_value(experience_batch, hl_policy_output_next, ll_policy_output_next)

            # (2) U_next 
            beta_mask = self._compute_termination_mask(experience_batch.observation_next)
            u_next =  (1 - beta_mask) * q_ll_next + beta_mask * q_hl_next

            # (3) ll_q_target
            ll_q_target = experience_batch.reward * self._hp.reward_scale + (1 - experience_batch.done) * self._hp.discount_factor * u_next
            
            if self._hp.clip_q_target:
                ll_q_target = self._clip_q_target(ll_q_target)
                
            ll_q_target = ll_q_target.detach()
            check_shape(ll_q_target, [self._hp.batch_size])

        return ll_q_target, q_hl_next, q_ll_next, u_next

    def _compute_next_value(self, experience_batch, hl_policy_output_next, ll_policy_output_next): 

        split_obs = self._split_obs(experience_batch.observation_next)
        obs = self._get_hl_obs(split_obs)

        # QLL(k+1)
        q_ll_next_raw = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(ll_policy_output_next.action)).q 
                            for critic_target in self.critic_targets])
        q_ll_next = q_ll_next_raw - self.alpha * ll_policy_output_next.prior_divergence[:, None]

        # for ablation study. LL with entropy
        # q_ll_next = q_ll_next_raw - self.alpha * ll_policy_output_next.log_prob[:, None]

        # QHL(k+1)
        q_hl_next_raw = torch.min(*[critic_target(obs, self._prep_action(hl_policy_output_next.action)).q for critic_target in self.hl_critic_targets])
        q_hl_next = q_hl_next_raw - self.alpha * ll_policy_output_next.prior_divergence[:, None]

        # for ablation study. conventional Qa.
        # q_hl_next = q_ll_next


        check_shape(q_ll_next, [self._hp.batch_size, 1])
        check_shape(q_hl_next, [self._hp.batch_size, 1])
        
        return q_hl_next.squeeze(-1), q_ll_next.squeeze(-1)

    def _compute_termination_mask(self, obs):
        # [1,0,0] is the start of a new hl step, so return 1 in the mask
        split_obs  = self._split_obs(obs) # to s,z,k
        beta_onehot = split_obs.time_index

        if isinstance(beta_onehot, np.ndarray):
            beta_onehot = torch.from_numpy(beta_onehot).float()

        beta = parse_one_hot(beta_onehot)
        beta_mask = torch.where(beta==torch.tensor(0), 1, 0)
        return beta_mask

    # =============================== utils ===========================
    def _split_obs(self, obs):
        assert obs.shape[1] == self.state_dim + self.latent_dim + self.onehot_dim
        return AttrDict(
            state=obs[:, :self.state_dim],   # condition decoding on state
            z=obs[:, self.state_dim:-self.onehot_dim],
            time_index=obs[:, -self.onehot_dim:],
        )
        
    def _get_hl_obs(self, split_obs):
        return split_obs.state
    
    def _get_ll_obs(self, split_obs):
        return split_obs.state
    # ====================================== property =======================================
    @property
    def state_dim(self):
        return self._hp.policy_params.policy_model_params.state_dim

    @property
    def latent_dim(self):
        return self._hp.policy_params.policy_model_params.nz_vae

    @property
    def onehot_dim(self):
        return self._hp.policy_params.policy_model_params.n_rollout_steps

    @property
    def action_dim(self):
        return self._hp.policy_params.policy_model_params.action_dim


    # ======================== for ablation study ====================
    # def _compute_alpha_loss(self, policy_output):
    #     self._target_entropy = -2
    #     return -1 * (self.alpha * (self._target_entropy + policy_output.log_prob).detach()).mean()


    # def _compute_policy_loss(self, experience_batch, policy_output):
    #     """Computes loss for policy update."""
    #     q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
    #                                   for critic in self.critics])
    #     policy_loss = -1 * q_est + self.alpha * policy_output.log_prob[:, None]
    #     check_shape(policy_loss, [self._hp.batch_size, 1])
    #     return policy_loss.mean()
    


class MazeLLInheritAgent(LLInheritAgent):
    def _split_obs(self, obs):
        assert obs.shape[1] == self.state_dim + self.image_dim + self.latent_dim + self.onehot_dim
        return AttrDict(
            state=obs[:, :self.state_dim],   
            image=obs[:, self.state_dim: self.state_dim + self.image_dim],
            z=obs[:, self.state_dim + self.image_dim:-self.onehot_dim],
            time_index=obs[:, -self.onehot_dim:],
        )
        
    def _get_hl_obs(self, split_obs):
        return torch.cat((split_obs.state, split_obs.image), dim=-1)
    
    def _get_ll_obs(self, split_obs):
        return split_obs.state
    
    @property
    def prior_input_res(self):
        return self._hp.policy_params.policy_model_params.prior_input_res
    
    @property
    def image_dim(self):
        return self.prior_input_res**2 * 3  * self._hp.policy_params.policy_model_params.n_input_frames

    