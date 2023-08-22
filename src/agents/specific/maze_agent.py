import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.agents.ac_agent import SACAgent
from src.agents.prior_sac_agent import ActionPriorSACAgent
from src.agents.hl_agent  import HLInheritAgent
from src.agents.skill_space_agent import SkillSpaceAgent, ACSkillSpaceAgent

from src.envs.maze import ACRandMaze0S40Env, ACmMaze1, ACmMaze2, ACmMaze3

class MazeAgent:
    chosen_maze = ACmMaze2
    START_POS = chosen_maze.START_POS
    TARGET_POS = chosen_maze.TARGET_POS
    VIS_RANGE = chosen_maze.VIS_RANGE

    """Adds replay logging function."""
    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes maze trajectories from replay buffer (if step < replay capacity)."""
        # if step > self.replay_buffer.capacity:
        #     return   # visualization does not work if earlier samples were overridden

        # get data
        size = self.replay_buffer.size
        states = self.replay_buffer.get().observation[:size, :2]

        print('!! place 1, log maze image, step is', step)
        plot_maze_fun(states, logger, step, size)

        
class MazeSkillSpaceAgent(SkillSpaceAgent, MazeAgent):
    """Collects samples in replay buffer for visualization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = self._hp.replay(self._hp.replay_params)

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer (used during warmup)."""
        self.replay_buffer.append(experience_batch)
        return SkillSpaceAgent.add_experience(self, experience_batch)

    def update(self, experience_batch):
        self.replay_buffer.append(experience_batch)
        return SkillSpaceAgent.update(self, experience_batch)

    def visualize(self, logger, rollout_storage, step):
        # MazeAgent.visualize(self, logger, rollout_storage, step) # only plot high level
        SkillSpaceAgent.visualize(self, logger, rollout_storage, step)


class MazeACSkillSpaceAgent(MazeSkillSpaceAgent, ACSkillSpaceAgent):
    """Maze version of ACSkillSpaceAgent for obs with agent-centric prior input."""
    def _act(self, obs):
        return ACSkillSpaceAgent._act(self, obs)


class MazeSACAgent(SACAgent, MazeAgent):
    def visualize(self, logger, rollout_storage, step):
        MazeAgent.visualize(self, logger, rollout_storage, step)
        SACAgent.visualize(self, logger, rollout_storage, step)


class MazeActionPriorSACAgent(ActionPriorSACAgent, MazeAgent):
    def visualize(self, logger, rollout_storage, step):
        MazeAgent.visualize(self, logger, rollout_storage, step)
        ActionPriorSACAgent.visualize(self, logger, rollout_storage, step)


class MazeNoUpdateAgent(MazeAgent, SACAgent):
    """Only logs rollouts, does not update policy."""
    def update(self, experience_batch):
        self.replay_buffer.append(experience_batch)
        return {}


class MazeACActionPriorSACAgent(ActionPriorSACAgent, MazeAgent):
    def __init__(self, *args, **kwargs):
        ActionPriorSACAgent.__init__(self, *args, **kwargs)
        from src.components.replay_buffer import SplitObsUniformReplayBuffer
        # TODO: don't hardcode this for res 32x32
        self.vis_replay_buffer = SplitObsUniformReplayBuffer({'capacity': 1e7, 'unused_obs_size': 6144,})

    def add_experience(self, experience_batch): 
        self.vis_replay_buffer.append(experience_batch)
        super().add_experience(experience_batch)

    def update(self, experience_batch):
        self.vis_replay_buffer.append(experience_batch)
        return ActionPriorSACAgent.update(self, experience_batch)

    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)
        self._vis_hl_q(logger, step)
        ActionPriorSACAgent.visualize(self, logger, rollout_storage, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes maze trajectories from replay buffer (if step < replay capacity)."""
        # if step > self.replay_buffer.capacity:
        #     return   # visualization does not work if earlier samples were overridden

        print('!! place 2, log maze image, step is', step)
        # get data
        size = self.vis_replay_buffer.size
        states = self.vis_replay_buffer.get().observation[:size, :2]
        plot_maze_fun(states, logger, step, size)

class MazeHLSkillAgent(HLSKillAgent, MazeAgent):
    def __init__(self, *args, **kwargs):
        HLSKillAgent.__init__(self, *args, **kwargs)
        from src.components.replay_buffer import SplitObsUniformReplayBuffer
        # TODO: don't hardcode this for res 32x32
        self.vis_replay_buffer = SplitObsUniformReplayBuffer({'capacity': 1e7, 'unused_obs_size': 6144,})

    def add_experience(self, experience_batch): 
        self.vis_replay_buffer.append(experience_batch)
        super().add_experience(experience_batch)

    def update(self, experience_batch=None):
        # self.vis_replay_buffer.append(experience_batch)
        return HLSKillAgent.update(self, experience_batch)

    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)
        self._vis_hl_q(logger, step)
        HLSKillAgent.visualize(self, logger, rollout_storage, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes maze trajectories from replay buffer (if step < replay capacity)."""
        size = self.vis_replay_buffer.size
        states = self.vis_replay_buffer.get().observation[:size, :2]
        plot_maze_fun(states, logger, step, size)

class MazeHLInheritAgent(HLInheritAgent, MazeAgent):
    def __init__(self, *args, **kwargs):
        HLInheritAgent.__init__(self, *args, **kwargs)
        from src.components.replay_buffer import SplitObsUniformReplayBuffer
        self.vis_replay_buffer = SplitObsUniformReplayBuffer({'capacity': 1e7, 'unused_obs_size': 6144,})

    def add_experience(self, experience_batch): 
        self.vis_replay_buffer.append(experience_batch)
        super().add_experience(experience_batch)

    def update(self, experience_batch=None):
        # self.vis_replay_buffer.append(experience_batch)
        return HLInheritAgent.update(self, experience_batch)

    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)
        self._vis_hl_q(logger, step)
        HLInheritAgent.visualize(self, logger, rollout_storage, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes maze trajectories from replay buffer (if step < replay capacity)."""
        size = self.vis_replay_buffer.size
        states = self.vis_replay_buffer.get().observation[:size, :2]
        plot_maze_fun(states, logger, step, size, prefix='')



def plot_maze_fun(states, logger, step, size, prefix=''):
    fig = plt.figure(figsize=(8,8))
    plt.scatter(states[:, 0], states[:, 1], s=5, c=np.arange(size), cmap='Blues')
    plt.plot(MazeAgent.START_POS[0], MazeAgent.START_POS[1], 'go', markeredgecolor='k')
    plt.plot(MazeAgent.TARGET_POS[0], MazeAgent.TARGET_POS[1], 'mo', markeredgecolor='k')
    plt.axis("equal")
    plt.title(prefix + 'replay, step ' + str(step) + ' size ' + str(size))
    plt.xlim(MazeAgent.VIS_RANGE[0])
    plt.ylim(MazeAgent.VIS_RANGE[1])
    logger.log_plot(fig, prefix + "replay_vis", step)
    plt.close(fig)
    
    # plot density
    # remove the point that close to the start
    dist = states[:, :2] - MazeAgent.START_POS
    dist_to_start = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
    show_index = np.where(dist_to_start > 1.0)[0]
    
    fig = plt.figure(figsize=(10,8))
    sns.histplot(x=states[show_index, 0], y=states[show_index, 1], cmap='Blues', cbar=True,
                 bins=50, pthresh=0.01)
    plt.plot(MazeAgent.START_POS[0], MazeAgent.START_POS[1], 'go', markeredgecolor='k')
    plt.plot(MazeAgent.TARGET_POS[0], MazeAgent.TARGET_POS[1], 'mo', markeredgecolor='k')
    plt.axis("equal")
    plt.title(prefix + 'density, step ' + str(step) + ' size ' + str(size))
    plt.xlim(MazeAgent.VIS_RANGE[0])
    plt.ylim(MazeAgent.VIS_RANGE[1])
    logger.log_plot(fig, prefix + "density_vis", step)
    plt.close(fig)
    
    # recent
    size= min(size, int(1e4))
    fig = plt.figure(figsize=(8,8))
    plt.scatter(states[-size:, 0], states[-size:, 1], s=5, c=np.arange(size), cmap='Blues')
    plt.plot(MazeAgent.START_POS[0], MazeAgent.START_POS[1], 'go', markeredgecolor='k')
    plt.plot(MazeAgent.TARGET_POS[0], MazeAgent.TARGET_POS[1], 'mo', markeredgecolor='k')
    plt.axis("equal")
    plt.title(prefix + 'replay, recent 10k, step ' + str(step) + ' size ' + str(size))
    plt.xlim(MazeAgent.VIS_RANGE[0])
    plt.ylim(MazeAgent.VIS_RANGE[1])
    logger.log_plot(fig, prefix + "replay_vis, , recent 10k, step", step)
    plt.close(fig)

def plot_maze_value(q, states, logger, step, size, fig_name='vis'):
    fig = plt.figure(figsize=(10,8))
    plt.scatter(states[:, 0], states[:, 1], s=5, c=q, cmap='Oranges')
    plt.plot(MazeAgent.START_POS[0], MazeAgent.START_POS[1], 'go', markeredgecolor='k')
    plt.plot(MazeAgent.TARGET_POS[0], MazeAgent.TARGET_POS[1], 'mo', markeredgecolor='k')
    plt.axis("equal")
    plt.title(fig_name + ' step ' + str(step) + ' size ' + str(size))
    plt.xlim(MazeAgent.VIS_RANGE[0])
    plt.ylim(MazeAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, fig_name, step)
    plt.close(fig)