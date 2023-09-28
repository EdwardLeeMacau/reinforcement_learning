import heapq
from collections import defaultdict
from enum import IntEnum
# import os
# import pickle

import numpy as np
# import pysnooper

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

    # def _dump(self, dirname: str) -> None:
    #     """Dump the values and policy

    #     Args:
    #         dirname (str): directory name
    #     """
    #     with open(os.path.join(dirname, f"values.pkl"), "wb") as f:
    #         pickle.dump(self.values, f)

    #     with open(os.path.join(dirname, f"policy.pkl"), "wb") as f:
    #         pickle.dump(self.policy, f)

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
        return np.max([self.Q(state, a) for a in range(self.grid_world.get_action_space())])

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        v = np.zeros_like(self.values)

        for s in range(self.grid_world.get_state_space()):
            v[s] = self.get_state_value(s)

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


class InplaceDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for InplaceDynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        v = self.values.copy()
        for s in range(self.grid_world.get_state_space()):
            self.values[s] = np.max([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

        delta = np.max(np.abs(self.values - v))
        return delta >= self.threshold

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        policy = np.zeros_like(self.policy)
        for s in range(self.grid_world.get_state_space()):
            policy[s] = np.argmax([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

        self.policy = policy

    def run(self):
        while self.policy_evaluation():
            pass

        self.policy_improvement()

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.key_index = {}  # key to index mapping
        self.count = 0

    def __str__(self):
        return str(self.heap)

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        _, _, item = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for idx, (p, c, i) in enumerate(self.heap):
            if i == item:
                # item already in, so has either lower or higher priority
                # if already in with smaller priority, don't do anything
                if p <= priority:
                    break
                # if already in with larger priority, update the priority and restore min-heap property
                del self.heap[idx]
                self.heap.append((priority, c, i))
                heapq.heapify(self.heap)
                break
            else:
                # item is not in, so just add to priority queue
                self.push(item, priority)

# Reference:
# https://github.com/kamenbliznashki/sutton_barto/blob/master/ch08_prioritized_sweeping.py
class PrioritizedSweeping(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PrioritizedSweeping

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        dim = (grid_world.get_state_space(), grid_world.get_action_space())

        self.n_planning_steps = 10              # number of planning steps
        self.alpha = 0.1                        # learning rate
        # self.pq = PriorityQueue()               # priority queue

        self._model = dict()                    # model(s, a) = (s', r)
        self._predecessors = defaultdict(set)   # predecessors(s) = {(s_predecessor, a_predecessor), ...}
        self._Q = np.zeros(dim)                 # store estimated Q(s, a)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        return np.max(self._Q[state])

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        v = self.values.copy()
        for s in range(self.grid_world.get_state_space()):
            self.values[s] = np.max([self.Q(s, a) for a in range(self.grid_world.get_action_space())])

        delta = np.max(np.abs(self.values - v))
        return delta >= self.threshold

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        policy = np.zeros_like(self.policy)
        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                next_state = self._model[(s, a)][0]
                values = self.values[next_state]

            # pick the action that maximizes the value
            policy[s] = next_state[np.argmax(values)]

        self.policy = policy

    def get_q_value(self, state: int, action: int) -> float:
        """Instead, query the Q-value from the iteratively updated Q-table."""
        return self._Q[state, action]

    def get_action(self, state: int) -> int:
        return np.argmax(self._Q[state])

    def get_predecessors(self, state: int) -> set:
        return self._predecessors[state]

    def init_model(self) -> None:
        for s in range(self.grid_world.get_state_space()):
            for a in range(self.grid_world.get_action_space()):
                next_state, reward, done_flag = self.grid_world.step(s, a)

                self._model[(s, a)] = (s if done_flag else next_state, reward)
                # if not done_flag:
                self._predecessors[next_state].add((s, a))

                # v = reward + self.discount_factor * np.max(self._Q[next_state]) * (1 - done_flag)
                # if (err := np.abs(v - self._Q[s, a])) > self.threshold:
                #     self.pq.push((s, a), -err)

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update the Q-value for a state and action"""
        # Update the model and predecessors for the current state and action
        self.init_model()

        # self._model[state, action] = (next_state, reward)
        # self._predecessors[next_state].add((state, action))

        #
        pq = PriorityQueue()

        # Update the Q-value for the current state and action
        v = reward + self.discount_factor * np.max(self._Q[next_state])
        if (err := np.abs(v - self._Q[state, action])) > self.threshold:
            pq.push((state, action), -err)

        # Update the Q-value for the predecessors of the current state
        for _ in range(self.n_planning_steps):
            if pq.is_empty():
                break

            state, action = pq.pop()
            next_state, reward = self._model[(state, action)]
            self._Q[state, action] += self.alpha * (reward + self.discount_factor * np.max(self._Q[next_state]) - self._Q[state, action])

            for s_pred, a_pred in self.get_predecessors(state):
                r, _ = self._model[(s_pred, a_pred)]

                v = r + self.discount_factor * np.max(self._Q[state])
                if (err := np.abs(v - self._Q[s_pred, a_pred])) > self.threshold:
                    pq.push((s_pred, a_pred), -err)


    def run(self) -> None:
        """Run the algorithm until convergence"""
        # Construct the model (with the information of predecessors),
        # self.init_model()

        # Iterate throught episodes
        state, done_flag = 0, 0
        while not done_flag:
            # Based on the previous observation, select an action
            action = self.get_action(state)

            # Interact with the environment
            next_state, reward, done_flag = self.grid_world.step(state, action)

            # Update the Q-table
            self.update(state, action, reward, next_state)
            state = next_state

        print(self._Q)

        # Update values and policy
        self.policy_evaluation()
        self.policy_improvement()

class Programming(IntEnum):
    PRIORITIZED_SWEEPING = 0
    INPLACE_DYNAMIC_PROGRAMMING = 1
    # REALTIME_DYNAMIC_PROGRAMMING = 2

class AsyncDynamicProgramming:
    ctor = [PrioritizedSweeping, InplaceDynamicProgramming]
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        T = Programming.INPLACE_DYNAMIC_PROGRAMMING
        self.programming = self.ctor[T](grid_world, discount_factor)

    def __getattr__(self, name):
        return getattr(self.programming, name)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        return self.programming.run()
