import heapq
import os
import pickle

import numpy as np

from gridworld import GridWorld


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
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)
        self.Q = self.get_q_value

    def _dump(self, dirname: str) -> None:
        """Dump the values and policy

        Args:
            dirname (str): directory name
        """
        with open(os.path.join(dirname, f"values.pkl"), "wb") as f:
            pickle.dump(self.values, f)

        with open(os.path.join(dirname, f"policy.pkl"), "wb") as f:
            pickle.dump(self.policy, f)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done_flag = self.grid_world.step(state, action)
        return reward + self.discount_factor * self.values[next_state] * (1 - done_flag)


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        return self.values[state]

    def evaluate(self) -> bool:
        """Evaluate the policy and update the values for one iteration

        Returns:
            bool: True if the values have not converged, False otherwise
        """
        # TODO: Implement the policy evaluation step
        v = np.zeros_like(self.values)

        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                p = self.policy[s, a]
                v[s] += p * self.Q(s, a)

        delta, self.values = np.max(np.abs(v - self.values)), v
        return delta >= self.threshold

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        while self.evaluate():
            pass


class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        return self.Q(state, self.policy[state])
        # return np.max([self.Q(state, a) for a in range(self.grid_world.get_action_space())])

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        delta, v = self.threshold, np.zeros_like(self.values)

        while delta >= self.threshold:
            for s in range(self.grid_world.get_state_space()):
                v[s] = self.get_state_value(s)

            delta, self.values = np.max(np.abs(v - self.values)), v

    def policy_improvement(self) -> bool:
        """Improve the policy based on the evaluated values

        Returns:
            bool: True if the policy has not converged, False otherwise
        """
        # TODO: Implement the policy improvement step
        policy = np.zeros_like(self.policy)
        for s in range(self.grid_world.get_state_space()):
            policy[s] = np.argmax([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

        stable, self.policy = np.all(policy == self.policy), policy
        return not stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        self.policy_evaluation()
        while self.policy_improvement():
            self.policy_evaluation()


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        return self.values[state]

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        v = np.zeros_like(self.values)

        for s in range(self.grid_world.get_state_space()):
            v[s] = np.max([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

        delta, self.values = np.max(np.abs(v - self.values)), v
        return delta >= self.threshold

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy = np.zeros_like(self.policy)
        for s in range(self.grid_world.get_state_space()):
            policy[s] = np.argmax([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

        self.policy = policy

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while self.policy_evaluation():
            pass

        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run_inplace_dynamic_programming(self) -> None:
        def policy_evaluation():
            """Evaluate the policy and update the values"""
            v = self.values.copy()
            for s in range(self.grid_world.get_state_space()):
                self.values[s] = np.max([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

            delta = np.max(np.abs(self.values - v))
            return delta >= self.threshold

        def policy_improvement():
            """Improve the policy based on the evaluated values"""
            policy = np.zeros_like(self.policy)
            for s in range(self.grid_world.get_state_space()):
                policy[s] = np.argmax([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

            self.policy = policy

        while policy_evaluation():
            pass

        policy_improvement()

    def run_prioritized_sweeping(self) -> None:
        raise NotImplementedError

    def run_real_time_dynamic_programming(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        PRIORITIZED_SWEEPING = 'prioritized_sweeping'
        INPLACE_DYNAMIC_PROGRAMMING = 'inplace_dynamic_programming'
        REALTIME_DYNAMIC_PROGRAMMING = 'real_time_dynamic_programming'

        f = {
            PRIORITIZED_SWEEPING: self.run_prioritized_sweeping,
            INPLACE_DYNAMIC_PROGRAMMING: self.run_inplace_dynamic_programming,
            REALTIME_DYNAMIC_PROGRAMMING: self.run_real_time_dynamic_programming,
        }

        learner = f[INPLACE_DYNAMIC_PROGRAMMING]
        return learner()
