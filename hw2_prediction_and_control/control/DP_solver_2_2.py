import random
import numpy as np
from collections import deque
from gridworld import GridWorld

from typing import List, Optional

# Type hinting
State, Action, Reward = int, int, float

class SummaryWriter:
    enable = True

    def __init__(self, project: str, name: str, config: Optional[dict] = None):
        if not self.enable:
            return

        self.run = wandb.init(project=project, name=name, config=config)

    def finish(self):
        if not self.enable:
            return

        self.run.finish()

    def log(self, **kwargs):
        if not self.enable:
            return

        wandb.log(kwargs)

if SummaryWriter.enable:
    import wandb

class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def sample_action(self, state: State) -> Action:
        """Sample action from Q(s,a) using epsilon-greedy

        Args:
            state (State): state

        Returns:
            Action: action
        """
        actions = range(self.action_space)
        return random.choice(actions) if random.random() < self.epsilon else self.policy_index[state]

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index

    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(
            self, state_trace: List[State], action_trace: List[Action], reward_trace: List[Reward]
        ) -> float:
        """Evaluate the policy and update the values after one episode

        Args:
            state_trace (List[State]): state trace
            action_trace (List[Action]): action trace
            reward_trace (List[Reward]): reward trace

        Returns:
            float: average absolute estimation loss
        """
        # TODO: Evaluate state value for each Q(s,a)

        G, trajectory = 0, [(s, a, r) for s, a, r in zip(state_trace, action_trace, reward_trace)][::-1]

        loss = []
        for s, a, r in trajectory:
            G = self.discount_factor * G + r
            estimation_loss = (G - self.q_values[s, a])
            self.q_values[s, a] += self.lr * estimation_loss
            loss.append(abs(estimation_loss))

        return sum(loss) / len(loss)

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy

        return self.get_policy_index()

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []

        # Collect average non-discounted reward over last 10 episodes
        episodes_reward : deque[float] = deque(maxlen=10)

        # Collect average estimated loss over last 10 episodes
        losses : deque[float]          = deque(maxlen=10)

        # Construct summary writer
        writer = SummaryWriter(project="rl", name="MC", config={
            "discount_factor": self.discount_factor,
            "learning_rate": self.lr,
            "epsilon": self.epsilon,
        })

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            # Sample action from epsilon-greedy policy
            action = self.sample_action(current_state)
            next_state, reward, done = self.grid_world.step(action)

            # Add A_{t}, R_{t+1}, S_{t+1} into trace
            state_trace.append(next_state)
            action_trace.append(action)
            reward_trace.append(reward)

            # Update current state
            current_state = next_state

            # if the trajectory is done, then evaluate the policy and improve it
            # also reset the trajectory
            if not done:
                continue

            iter_episode += 1
            estimation_loss = self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()

            # Log the reward over last 10 episodes
            non_discounted_average_reward = sum(reward_trace) / len(reward_trace)
            episodes_reward.append(non_discounted_average_reward)
            writer.log(reward=sum(episodes_reward) / len(episodes_reward))

            # Log the estimation loss over last 10 episodes
            losses.append(estimation_loss)
            writer.log(loss=sum(losses) / len(losses))

            # Reset the trajectory
            state_trace  = [current_state]
            action_trace = []
            reward_trace = []

        writer.finish()


class SARSA(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> float:
        """Evaluate the policy and update the values after one step

        Returns:
            float: average absolute estimation loss
        """
        # TODO: Evaluate Q value after one step and improve the policy

        next_value = (self.q_values[s2, a2] if not is_done else 0)
        temporal_difference_error = \
            r + self.discount_factor * next_value - self.q_values[s, a]

        self.q_values[s, a] += self.lr * temporal_difference_error
        self.policy_index[s] = self.q_values[s].argmax()

        return abs(temporal_difference_error)

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = current_state
        prev_a = self.sample_action(prev_s)
        prev_r = None
        is_done = False
        reward_trace = []
        estimation_loss = []

        # Collect average non-discounted reward over last 10 episodes
        episodes_reward : deque[float] = deque(maxlen=10)

        # Collect average estimated loss over last 10 episodes
        losses : deque[float]          = deque(maxlen=10)

        # Construct summary writer
        writer  = SummaryWriter(project="rl", name="SARSA", config={
            "discount_factor": self.discount_factor,
            "learning_rate": self.lr,
            "epsilon": self.epsilon,
        })

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            # Sample action from epsilon-greedy policy
            current_state, reward, done = self.grid_world.step(prev_a)
            reward_trace.append(reward)

            # Choose next action from epsilon-greedy policy
            action = self.sample_action(current_state)

            # Update Q(s,a) after one step
            loss = self.policy_eval_improve(prev_s, prev_a, reward, current_state, action, done)
            estimation_loss.append(loss)

            # Update current state
            prev_s, prev_a = current_state, action

            # if the trajectory is done, then reset the trajectory
            if not done:
                continue

            iter_episode += 1

            # Log the reward and estimation loss
            non_discounted_average_reward = sum(reward_trace) / len(reward_trace)
            episodes_reward.append(non_discounted_average_reward)

            losses.append(sum(estimation_loss) / len(estimation_loss))

            writer.log(reward=sum(episodes_reward) / len(episodes_reward))
            writer.log(loss=sum(losses) / len(losses))

            # Reset the trajectory
            reward_trace = []
            estimation_loss = []

        writer.finish()

class Q_Learning(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> List:
        # TODO: sample a batch of index of transitions from the buffer
        n = min(self.sample_batch_size, len(self.buffer))
        return [self.buffer[i] for i in random.sample(range(len(self.buffer)), n)]

    def policy_eval_improve(self, s, a, r, s2, is_done) -> float:
        """Evaluate the policy and update the values after one step

        Returns:
            float: average absolute estimation loss
        """
        #TODO: Evaluate Q value after one step and improve the policy
        # NOTE: take max over all actions here
        next_value = self.q_values[s2].max() if not is_done else 0
        temporal_difference_error = \
            r + self.discount_factor * next_value - self.q_values[s, a]

        self.q_values[s, a] += self.lr * temporal_difference_error
        self.policy_index[s] = self.q_values[s].argmax()

        return abs(temporal_difference_error)

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = current_state
        prev_a = None
        prev_r = None
        is_done = False
        samples = []
        reward_trace = []
        estimation_loss = []
        transition_count = 0

        # Collect average non-discounted reward over last 10 episodes
        episodes_reward : deque[float] = deque(maxlen=10)

        # Collect average estimated loss over last 10 episodes
        losses : deque[float]          = deque(maxlen=10)

        # Construct summary writer
        writer  = SummaryWriter(project="rl", name="Q-Learning", config={
            "discount_factor": self.discount_factor,
            "learning_rate": self.lr,
            "epsilon": self.epsilon,
            "buffer_size": self.buffer.maxlen,
            "update_frequency": self.update_frequency,
            "sample_batch_size": self.sample_batch_size,
        })

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            # Sample action from epsilon-greedy policy
            prev_a = self.sample_action(prev_s)
            current_state, reward, done = self.grid_world.step(prev_a)

            # Store the transition into buffer
            self.add_buffer(prev_s, prev_a, reward, current_state, done)
            transition_count += 1

            # Store the reward and estimation loss
            next_value = self.q_values[current_state].max() if not done else 0
            temporal_difference_error = \
                reward + self.discount_factor * next_value - self.q_values[prev_s, prev_a]

            reward_trace.append(reward)
            estimation_loss.append(abs(temporal_difference_error))

            # Uniformly sample a batch of transitions from buffer
            if transition_count % self.update_frequency == 0:
                samples = self.sample_batch()

                # Update Q(s,a) after one step
                for s, a, r, s2, d in samples:
                    self.policy_eval_improve(s, a, r, s2, d)

            # Update current state
            prev_s = current_state

            # if the trajectory is done, then reset the trajectory
            if not done:
                continue

            iter_episode += 1

            # Log the reward and estimation loss
            non_discounted_average_reward = sum(reward_trace) / len(reward_trace)
            episodes_reward.append(non_discounted_average_reward)

            losses.append(sum(estimation_loss) / len(estimation_loss))

            writer.log(reward=sum(episodes_reward) / len(episodes_reward))
            writer.log(loss=sum(losses) / len(losses))

            reward_trace = []
            estimation_loss = []

        writer.finish()