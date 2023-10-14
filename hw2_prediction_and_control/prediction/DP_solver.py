import numpy as np
import json
from collections import defaultdict

from typing import List, Union, Tuple
from gridworld import GridWorld

import pysnooper

# Type hinting
State, Reward = int, float

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
        self.values       = np.zeros(self.state_space)

    def get_all_state_values(self) -> np.array:
        return self.values


class MonteCarloPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float):
        """Constructor for MonteCarloPrediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)

        # Initialize the sampled returns table
        self.returns = defaultdict(list)

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        episodes: List[Tuple[State, Reward]] = []
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()

            # Push S_{t}, R_{t+1} into episodes
            episodes.append((current_state, reward))
            current_state = next_state

            # Collect an episode until done-flag is True.
            if not done:
                continue

            # Then, do the first-visit MC update for each state in the episode.
            G, states = 0, [s for s, _ in episodes]
            for s, r in episodes[::-1]:
                G = self.discount_factor * G + r

                states.pop()
                if s not in states:
                    self.returns[s].append(G)
                    self.values[s] = np.mean(self.returns[s])

            # After updated the state-value table V(s), reset the episode.
            episodes = []


class TDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.grid_world.check():
            next_state, reward, done = self.grid_world.step()

            current_value = self.values[current_state]
            next_value    = (self.values[next_state] if not done else 0)

            # Update the state-value table V(s)
            self.values[current_state] += \
                self.lr * (reward + self.discount_factor * next_value - current_value)

            current_state = next_state


class NstepTDPrediction(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, num_step: int):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world, discount_factor)
        self.lr     = learning_rate
        self.n      = num_step

    # @pysnooper.snoop()
    def run(self) -> None:
        """Run the algorithm until self.grid_world.check() == False"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        episodes: List[Tuple[State, Reward]] = []
        t, T = 0, float('inf')
        while self.grid_world.check():
            if t < T:
                next_state, reward, done = self.grid_world.step()

                # Push S_{t}, R_{t+1} into buffer
                episodes.append((current_state, reward))
                current_state = next_state

                # if terminal state, set T = t + 1. Otherwise, set T = inf
                T = t + 1 if done else T

            # Do the TD(lambda) update when tau is non-negative
            if (tau := t - self.n + 1) >= 0:
                G = sum([(self.discount_factor ** i) * r for i, (_, r) in enumerate(episodes[tau:])])

                if tau + self.n < T:
                    G += (self.discount_factor ** self.n) * self.values[next_state]

                # Update the state-value table V(s)
                self.values[episodes[tau][0]] += self.lr * (G - self.values[episodes[tau][0]])

            # Update the time step
            t += 1

            # If the episode is done, reset the episode.
            if tau == T - 1:
                episodes.clear()
                t, T = 0, float('inf')

