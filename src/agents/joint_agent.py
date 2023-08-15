import imp
from src.agents.agent import HierarchicalAgent, FixedIntervalHierarchicalAgent, FixedIntervalTimeIndexedHierarchicalAgent

from src.utils.general_utils import ParamDict, AttrDict, prefix_dict, map_dict

from enum import Enum
from tqdm import tqdm

class skill_critic_stages(Enum):
    WARM_START = 0
    HL_TRAIN = 1
    LL_TRAIN = 2
    HYBRID = 3

    LL_TRAIN_PI = 4
    HL_LLVAR = 5
    FIX_LL_PI = 6

class JointInheritAgent(FixedIntervalTimeIndexedHierarchicalAgent):
    def __init__(self, config):
        super().__init__(config)
        self.set_agents()
        self._train_stage = None

        # update the trianing stage
        if self._train_stage is None:
            self._train_stage = self._hp.initial_train_stage

        self.train_stages_control(self._train_stage)

    def set_agents(self):
        self.ll_agent.update_by_hl_agent(self.hl_agent)

    def _default_hparams(self):
        default_dict = ParamDict({
            'initial_train_stage': skill_critic_stages.HYBRID,
        })
        return super()._default_hparams().overwrite(default_dict)

    def train_stages_control(self, stage=None):
        print('!! Change Skill-critic stage to ', stage)

        if stage == skill_critic_stages.WARM_START:
        # 1) warm-start stage
            # policy: HL var, LL var
            # update: HL Q, LL Q (to convergence)
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([False, True,])
            self.ll_agent.fast_assign_flags([False, True])

        elif stage == skill_critic_stages.HL_TRAIN:
        # 2) HL training stage:
            # policy: HL var, LL determine
            # update: HL Q, LL Q, HL Pi
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_on_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([False, True])

        elif stage == skill_critic_stages.LL_TRAIN:
        # 3) LL training stage:
            # policy: HL var, LL var
            # update: HL Q, LL Q, LL Pi
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([False, False])
            self.ll_agent.fast_assign_flags([True, True])

        elif stage == skill_critic_stages.HYBRID:
        # 4) hybrid stage
            # policy: all var
            # update: all
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([True, True])

        elif stage == skill_critic_stages.HL_LLVAR:
        # 5) only train LL policy, without LL variance
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([False, True])
            
        elif stage == skill_critic_stages.SC_WO_LLVAR:
        # 6) only train LL policy, without LL variance
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_on_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([True, True])

        else:
            self.hl_agent.switch_off_deterministic_action_mode()
            self.ll_agent.switch_off_deterministic_action_mode()
            self.hl_agent.fast_assign_flags([True, True])
            self.ll_agent.fast_assign_flags([True, True])


    # ====================== for update, we have some stages ====================
    def update(self, experience_batches):
        """Updates high-level and low-level agents depending on which parameters are set."""
        assert isinstance(experience_batches, AttrDict)  # update requires batches for both HL and LL
        update_outputs = AttrDict()

        # 1) add experience
        if self._hp.update_hl:
            self.hl_agent.add_experience(experience_batches.hl_batch)

        if self._hp.update_ll:
            self.ll_agent.add_experience(experience_batches.ll_batch)


        # 2) for and update HL, LL
        # for idx in tqdm(range(self._hp.update_iterations)):
        for idx in range(self._hp.update_iterations):
            
            if self._hp.update_hl:
                hl_update_outputs = self.hl_agent.update()
                update_outputs.update(hl_update_outputs)

            if self._hp.update_ll:
                ll_update_outputs = self.ll_agent.update()
                update_outputs.update(ll_update_outputs)

        return update_outputs
