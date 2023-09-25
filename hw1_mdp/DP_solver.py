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
        return reward + self.discount_factor * self.values[next_state]


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
        return self.values[state]

    def evaluate(self) -> bool:
        """Evaluate the policy and update the values for one iteration

        Returns:
            bool: True if the values have not converged, False otherwise
        """
        v = np.zeros_like(self.values)

        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                p = self.policy[state, action]

                next_state, reward, done_flag = self.grid_world.step(state, action)
                v[state] += p * (reward + self.discount_factor * self.get_state_value(next_state) * (1 - done_flag))

        delta, self.values = np.max(np.abs(v - self.values)), v
        return delta >= self.threshold

    def run(self) -> None:
        """Run the algorithm until convergence."""
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
        return self.values[state]

    def policy_evaluation(self) -> bool:
        """Evaluate the policy and update the values"""
        delta, v = self.threshold, np.zeros_like(self.values)

        while delta >= self.threshold:
            for state in range(self.grid_world.get_state_space()):
                action = self.policy[state]

                next_state, reward, done_flag = self.grid_world.step(state, action)
                v[state] = reward + self.discount_factor * self.get_state_value(next_state) * (1 - done_flag)

            delta, self.values = np.max(np.abs(v - self.values)), v

    def policy_improvement(self) -> bool:
        """Improve the policy based on the evaluated values

        Returns:
            bool: True if the policy has not converged, False otherwise
        """
        policy = np.zeros_like(self.policy)
        for state in range(self.grid_world.get_state_space()):
            policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

        is_stable, self.policy = np.all(policy == self.policy), policy
        return not is_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        run = True

        while run:
            self.policy_evaluation()
            run = self.policy_improvement()


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
        return self.values[state]

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        v = np.zeros_like(self.values)

        for state in range(self.grid_world.get_state_space()):
            action = self.policy[state]

            next_state, reward, done_flag = self.grid_world.step(state, action)
            v[state] = reward + self.discount_factor * self.get_state_value(next_state) * (1 - done_flag)

        delta, self.values = np.max(np.abs(v - self.values)), v
        return delta >= self.threshold

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        policy = np.zeros_like(self.policy)
        for state in range(self.grid_world.get_state_space()):
            policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

        self.policy = policy

    def run(self) -> None:
        """Run the algorithm until convergence"""
        run = True

        while run:
            run = self.policy_evaluation()
            self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        raise NotImplementedError
