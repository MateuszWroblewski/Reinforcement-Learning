from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2023.constants import EX1_CONSTANTS as CONSTANTS
from rl2023.exercise1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)
        delta=np.inf
        ### I PUT MY CODE HERE - beginning ###
        while (delta>theta):
            #print("V= ",V)
            delta=0
            for state_index in range(self.state_dim):
                old_value_of_current_state=V[state_index]
                V[state_index]=max([np.sum([self.mdp.P[state_index, action, new_state_index]*(self.mdp.R[state_index, action, new_state_index]+self.gamma*V[new_state_index]) for new_state_index in range(self.state_dim)]) for action in range(self.action_dim)])
                delta=max(delta, abs(old_value_of_current_state-V[state_index]))
                #print("delta= ",delta)

        ### I PUT MY CODE HERE - end ###
        #raise NotImplementedError("Needed for Q1")
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        ### I PUT MY CODE HERE - beginning ###
        for state_index in range(self.state_dim):
            action_values=[np.sum([self.mdp.P[state_index, action, new_state_index]*(self.mdp.R[state_index, action, new_state_index]+self.gamma*V[new_state_index]) for new_state_index in range(self.state_dim)]) for action in range(self.action_dim)]
            best_action_value=max(action_values)
            optimal_actions=np.argwhere(action_values == best_action_value).flatten().tolist()
            for index_of_an_optimal_action in optimal_actions:
                policy[state_index, index_of_an_optimal_action]=1/(len(optimal_actions))
        #raise NotImplementedError("Needed for Q1")
        #print("akcje to: \n", self.mdp.actions)
        #print("stany to: \n", self.mdp.states)
        #print("policy to \n", policy)
        ### I PUT MY CODE HERE - end ###
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)
        ### I PUT MY CODE HERE - beginning ###
        delta=np.inf
        while delta>self.theta:
            delta=0
            for state_index in range(self.state_dim):
                old_value_of_current_state=V[state_index]
                V[state_index]=sum([policy[state_index, action]*sum([self.mdp.P[state_index, action, new_state]*(self.mdp.R[state_index,action,new_state]+self.gamma*V[new_state]) for new_state in range(self.state_dim)]) for action in range(self.action_dim)])
                delta=max(delta, abs(old_value_of_current_state-V[state_index]))
        #raise NotImplementedError("Needed for Q1")
        ### I PUT MY CODE HERE - end ###
        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        policy = np.zeros([self.state_dim, self.action_dim])
        V = np.zeros([self.state_dim])
        ### I PUT MY CODE HERE - beginning ###
        policy_stable=0
        while policy_stable==0:
            policy_stable=1
            for state_index in range(self.state_dim):
                old_optimal_action=np.argmax(policy[state_index, :])
                action_values=[np.sum([self.mdp.P[state_index, action, new_state_index]*(self.mdp.R[state_index, action, new_state_index]+self.gamma*V[new_state_index]) for new_state_index in range(self.state_dim)]) for action in range(self.action_dim)]
                optimal_action=np.argmax(action_values)
                if optimal_action!=old_optimal_action:
                    policy_stable=0
                    policy[state_index, optimal_action]=1
                    policy[state_index, old_optimal_action]=0
            self._policy_eval(policy)
            #print("V= ",V)
            #print("akcje to: \n", self.mdp.actions)
            #print("stany to: \n", self.mdp.states)
            #print("policy to \n", policy)
            


        #raise NotImplementedError("Needed for Q1")
        ### I PUT MY CODE HERE - end ###
        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
Transition("X:0Y:0", "right", "X:1Y:0", 0.7,-1),
Transition("X:0Y:0", "right", "X:0Y:0", 0.3,-1),
Transition("X:0Y:0", "left", "X:4Y:0", 0.6,-1),
Transition("X:0Y:0", "left", "X:0Y:0", 0.4,-1),
Transition("X:0Y:0", "up", "X:0Y:1", 0.5,-1),
Transition("X:0Y:0", "up", "X:0Y:0", 0.5,-1),
Transition("X:0Y:0", "down", "X:0Y:6", 0.4,-1),
Transition("X:0Y:0", "down", "X:0Y:0", 0.6,-1),
Transition("X:0Y:1", "right", "X:1Y:1", 0.7,-1),
Transition("X:0Y:1", "right", "X:0Y:1", 0.3,-1),
Transition("X:0Y:1", "left", "X:4Y:1", 0.6,-1),
Transition("X:0Y:1", "left", "X:0Y:1", 0.4,-1),
Transition("X:0Y:1", "up", "X:0Y:2", 0.5,-1),
Transition("X:0Y:1", "up", "X:0Y:1", 0.5,-1),
Transition("X:0Y:1", "down", "X:0Y:0", 0.4,-1),
Transition("X:0Y:1", "down", "X:0Y:1", 0.6,-1),
Transition("X:0Y:2", "right", "X:1Y:2", 0.7,-1),
Transition("X:0Y:2", "right", "X:0Y:2", 0.3,-1),
Transition("X:0Y:2", "left", "X:4Y:2", 0.6,-1),
Transition("X:0Y:2", "left", "X:0Y:2", 0.4,-1),
Transition("X:0Y:2", "up", "X:0Y:3", 0.5,-1),
Transition("X:0Y:2", "up", "X:0Y:2", 0.5,-1),
Transition("X:0Y:2", "down", "X:0Y:1", 0.4,-1),
Transition("X:0Y:2", "down", "X:0Y:2", 0.6,-1),
Transition("X:0Y:3", "right", "X:1Y:3", 0.7,-1),
Transition("X:0Y:3", "right", "X:0Y:3", 0.3,-1),
Transition("X:0Y:3", "left", "X:4Y:3", 0.6,-1),
Transition("X:0Y:3", "left", "X:0Y:3", 0.4,-1),
Transition("X:0Y:3", "up", "X:0Y:4", 0.5,-1),
Transition("X:0Y:3", "up", "X:0Y:3", 0.5,-1),
Transition("X:0Y:3", "down", "X:0Y:2", 0.4,-1),
Transition("X:0Y:3", "down", "X:0Y:3", 0.6,-1),
Transition("X:0Y:4", "right", "X:1Y:4", 0.7,-1),
Transition("X:0Y:4", "right", "X:0Y:4", 0.3,-1),
Transition("X:0Y:4", "left", "X:4Y:4", 0.6,-1),
Transition("X:0Y:4", "left", "X:0Y:4", 0.4,-1),
Transition("X:0Y:4", "up", "X:0Y:5", 0.5,-1),
Transition("X:0Y:4", "up", "X:0Y:4", 0.5,-1),
Transition("X:0Y:4", "down", "X:0Y:3", 0.4,-1),
Transition("X:0Y:4", "down", "X:0Y:4", 0.6,-1),
Transition("X:0Y:5", "right", "X:1Y:0", 0.7,-1),
Transition("X:0Y:5", "right", "X:0Y:5", 0.3,-1),
Transition("X:0Y:5", "left", "X:4Y:0", 0.6,-1),
Transition("X:0Y:5", "left", "X:0Y:5", 0.4,-1),
Transition("X:0Y:5", "up", "X:0Y:6", 0.5,-1),
Transition("X:0Y:5", "up", "X:0Y:5", 0.5,-1),
Transition("X:0Y:5", "down", "X:0Y:4", 0.4,-1),
Transition("X:0Y:5", "down", "X:0Y:5", 0.6,-1),
Transition("X:0Y:6", "right", "X:1Y:1", 0.7,-1),
Transition("X:0Y:6", "right", "X:0Y:6", 0.3,-1),
Transition("X:0Y:6", "left", "X:4Y:1", 0.6,-1),
Transition("X:0Y:6", "left", "X:0Y:6", 0.4,-1),
Transition("X:0Y:6", "up", "X:0Y:0", 0.5,-1),
Transition("X:0Y:6", "up", "X:0Y:6", 0.5,-1),
Transition("X:0Y:6", "down", "X:0Y:5", 0.4,-1),
Transition("X:0Y:6", "down", "X:0Y:6", 0.6,-1),
Transition("X:1Y:0", "right", "X:2Y:0", 0.7,-1),
Transition("X:1Y:0", "right", "X:1Y:0", 0.3,-1),
Transition("X:1Y:0", "left", "X:0Y:0", 0.6,-1),
Transition("X:1Y:0", "left", "X:1Y:0", 0.4,-1),
Transition("X:1Y:0", "up", "X:1Y:1", 0.5,-1),
Transition("X:1Y:0", "up", "X:1Y:0", 0.5,-1),
Transition("X:1Y:0", "down", "X:1Y:6", 0.4,-1),
Transition("X:1Y:0", "down", "X:1Y:0", 0.6,-1),
Transition("X:1Y:1", "right", "X:2Y:1", 0.7,-1),
Transition("X:1Y:1", "right", "X:1Y:1", 0.3,-1),
Transition("X:1Y:1", "left", "X:0Y:1", 0.6,-1),
Transition("X:1Y:1", "left", "X:1Y:1", 0.4,-1),
Transition("X:1Y:1", "up", "X:1Y:2", 0.5,-1),
Transition("X:1Y:1", "up", "X:1Y:1", 0.5,-1),
Transition("X:1Y:1", "down", "X:1Y:0", 0.4,-1),
Transition("X:1Y:1", "down", "X:1Y:1", 0.6,-1),
Transition("X:1Y:2", "right", "X:2Y:2", 0.7,-1),
Transition("X:1Y:2", "right", "X:1Y:2", 0.3,-1),
Transition("X:1Y:2", "left", "X:0Y:2", 0.6,-1),
Transition("X:1Y:2", "left", "X:1Y:2", 0.4,-1),
Transition("X:1Y:2", "up", "X:1Y:3", 0.5,-1),
Transition("X:1Y:2", "up", "X:1Y:2", 0.5,-1),
Transition("X:1Y:2", "down", "X:1Y:1", 0.4,-1),
Transition("X:1Y:2", "down", "X:1Y:2", 0.6,-1),
Transition("X:1Y:3", "right", "X:2Y:3", 0.7,-1),
Transition("X:1Y:3", "right", "X:1Y:3", 0.3,-1),
Transition("X:1Y:3", "left", "X:0Y:3", 0.6,-1),
Transition("X:1Y:3", "left", "X:1Y:3", 0.4,-1),
Transition("X:1Y:3", "up", "X:1Y:4", 0.5,-1),
Transition("X:1Y:3", "up", "X:1Y:3", 0.5,-1),
Transition("X:1Y:3", "down", "X:1Y:2", 0.4,-1),
Transition("X:1Y:3", "down", "X:1Y:3", 0.6,-1),
Transition("X:1Y:4", "right", "X:2Y:4", 0.7,-1),
Transition("X:1Y:4", "right", "X:1Y:4", 0.3,-1),
Transition("X:1Y:4", "left", "X:0Y:4", 0.6,-1),
Transition("X:1Y:4", "left", "X:1Y:4", 0.4,-1),
Transition("X:1Y:4", "up", "X:1Y:5", 0.5,-1),
Transition("X:1Y:4", "up", "X:1Y:4", 0.5,-1),
Transition("X:1Y:4", "down", "X:1Y:3", 0.4,-1),
Transition("X:1Y:4", "down", "X:1Y:4", 0.6,-1),
Transition("X:1Y:5", "right", "X:2Y:0", 0.7,-1),
Transition("X:1Y:5", "right", "X:1Y:5", 0.3,-1),
Transition("X:1Y:5", "left", "X:0Y:0", 0.6,-1),
Transition("X:1Y:5", "left", "X:1Y:5", 0.4,-1),
Transition("X:1Y:5", "up", "X:1Y:6", 0.5,-1),
Transition("X:1Y:5", "up", "X:1Y:5", 0.5,-1),
Transition("X:1Y:5", "down", "X:1Y:4", 0.4,-1),
Transition("X:1Y:5", "down", "X:1Y:5", 0.6,-1),
Transition("X:1Y:6", "right", "X:2Y:1", 0.7,-1),
Transition("X:1Y:6", "right", "X:1Y:6", 0.3,-1),
Transition("X:1Y:6", "left", "X:0Y:1", 0.6,-1),
Transition("X:1Y:6", "left", "X:1Y:6", 0.4,-1),
Transition("X:1Y:6", "up", "X:1Y:0", 0.5,-1),
Transition("X:1Y:6", "up", "X:1Y:6", 0.5,-1),
Transition("X:1Y:6", "down", "X:1Y:5", 0.4,-1),
Transition("X:1Y:6", "down", "X:1Y:6", 0.6,-1),
Transition("X:2Y:0", "right", "X:3Y:0", 0.7,-1),
Transition("X:2Y:0", "right", "X:2Y:0", 0.3,-1),
Transition("X:2Y:0", "left", "X:1Y:0", 0.6,-1),
Transition("X:2Y:0", "left", "X:2Y:0", 0.4,-1),
Transition("X:2Y:0", "up", "X:2Y:1", 0.5,-1),
Transition("X:2Y:0", "up", "X:2Y:0", 0.5,-1),
Transition("X:2Y:0", "down", "X:2Y:6", 0.4,-1),
Transition("X:2Y:0", "down", "X:2Y:0", 0.6,-1),
Transition("X:2Y:1", "right", "X:3Y:1", 0.7,-1),
Transition("X:2Y:1", "right", "X:2Y:1", 0.3,-1),
Transition("X:2Y:1", "left", "X:1Y:1", 0.6,-1),
Transition("X:2Y:1", "left", "X:2Y:1", 0.4,-1),
Transition("X:2Y:1", "up", "X:2Y:2", 0.5,-1),
Transition("X:2Y:1", "up", "X:2Y:1", 0.5,-1),
Transition("X:2Y:1", "down", "X:2Y:0", 0.4,-1),
Transition("X:2Y:1", "down", "X:2Y:1", 0.6,-1),
Transition("X:2Y:2", "right", "X:3Y:2", 0.7,-1),
Transition("X:2Y:2", "right", "X:2Y:2", 0.3,-1),
Transition("X:2Y:2", "left", "X:1Y:2", 0.6,-1),
Transition("X:2Y:2", "left", "X:2Y:2", 0.4,-1),
Transition("X:2Y:2", "up", "X:2Y:3", 0.5,-1),
Transition("X:2Y:2", "up", "X:2Y:2", 0.5,-1),
Transition("X:2Y:2", "down", "X:2Y:1", 0.4,-1),
Transition("X:2Y:2", "down", "X:2Y:2", 0.6,-1),
Transition("X:2Y:3", "right", "X:3Y:3", 0.7,22),
Transition("X:2Y:3", "right", "X:2Y:3", 0.3,-1),
Transition("X:2Y:3", "left", "X:1Y:3", 0.6,-1),
Transition("X:2Y:3", "left", "X:2Y:3", 0.4,-1),
Transition("X:2Y:3", "up", "X:2Y:4", 0.5,-1),
Transition("X:2Y:3", "up", "X:2Y:3", 0.5,-1),
Transition("X:2Y:3", "down", "X:2Y:2", 0.4,-1),
Transition("X:2Y:3", "down", "X:2Y:3", 0.6,-1),
Transition("X:2Y:4", "right", "X:3Y:4", 0.7,-1),
Transition("X:2Y:4", "right", "X:2Y:4", 0.3,-1),
Transition("X:2Y:4", "left", "X:1Y:4", 0.6,-1),
Transition("X:2Y:4", "left", "X:2Y:4", 0.4,-1),
Transition("X:2Y:4", "up", "X:2Y:5", 0.5,-1),
Transition("X:2Y:4", "up", "X:2Y:4", 0.5,-1),
Transition("X:2Y:4", "down", "X:2Y:3", 0.4,-1),
Transition("X:2Y:4", "down", "X:2Y:4", 0.6,-1),
Transition("X:2Y:5", "right", "X:3Y:0", 0.7,-1),
Transition("X:2Y:5", "right", "X:2Y:5", 0.3,-1),
Transition("X:2Y:5", "left", "X:1Y:0", 0.6,-1),
Transition("X:2Y:5", "left", "X:2Y:5", 0.4,-1),
Transition("X:2Y:5", "up", "X:2Y:6", 0.5,-1),
Transition("X:2Y:5", "up", "X:2Y:5", 0.5,-1),
Transition("X:2Y:5", "down", "X:2Y:4", 0.4,-1),
Transition("X:2Y:5", "down", "X:2Y:5", 0.6,-1),
Transition("X:2Y:6", "right", "X:3Y:1", 0.7,-1),
Transition("X:2Y:6", "right", "X:2Y:6", 0.3,-1),
Transition("X:2Y:6", "left", "X:1Y:1", 0.6,-1),
Transition("X:2Y:6", "left", "X:2Y:6", 0.4,-1),
Transition("X:2Y:6", "up", "X:2Y:0", 0.5,-1),
Transition("X:2Y:6", "up", "X:2Y:6", 0.5,-1),
Transition("X:2Y:6", "down", "X:2Y:5", 0.4,-1),
Transition("X:2Y:6", "down", "X:2Y:6", 0.6,-1),
Transition("X:3Y:0", "right", "X:4Y:0", 0.7,-1),
Transition("X:3Y:0", "right", "X:3Y:0", 0.3,-1),
Transition("X:3Y:0", "left", "X:2Y:0", 0.6,-1),
Transition("X:3Y:0", "left", "X:3Y:0", 0.4,-1),
Transition("X:3Y:0", "up", "X:3Y:1", 0.5,-1),
Transition("X:3Y:0", "up", "X:3Y:0", 0.5,-1),
Transition("X:3Y:0", "down", "X:3Y:6", 0.4,-1),
Transition("X:3Y:0", "down", "X:3Y:0", 0.6,-1),
Transition("X:3Y:1", "right", "X:4Y:1", 0.7,-1),
Transition("X:3Y:1", "right", "X:3Y:1", 0.3,-1),
Transition("X:3Y:1", "left", "X:2Y:1", 0.6,-1),
Transition("X:3Y:1", "left", "X:3Y:1", 0.4,-1),
Transition("X:3Y:1", "up", "X:3Y:2", 0.5,-1),
Transition("X:3Y:1", "up", "X:3Y:1", 0.5,-1),
Transition("X:3Y:1", "down", "X:3Y:0", 0.4,-1),
Transition("X:3Y:1", "down", "X:3Y:1", 0.6,-1),
Transition("X:3Y:2", "right", "X:4Y:2", 0.7,-1),
Transition("X:3Y:2", "right", "X:3Y:2", 0.3,-1),
Transition("X:3Y:2", "left", "X:2Y:2", 0.6,-1),
Transition("X:3Y:2", "left", "X:3Y:2", 0.4,-1),
Transition("X:3Y:2", "up", "X:3Y:3", 0.5,22),
Transition("X:3Y:2", "up", "X:3Y:2", 0.5,-1),
Transition("X:3Y:2", "down", "X:3Y:1", 0.4,-1),
Transition("X:3Y:2", "down", "X:3Y:2", 0.6,-1),
Transition("X:3Y:3", "right", "X:3Y:3", 1,0),
Transition("X:3Y:3", "left", "X:3Y:3", 1,0),
Transition("X:3Y:3", "up", "X:3Y:3", 1,0),
Transition("X:3Y:3", "down", "X:3Y:3", 1,0),
Transition("X:3Y:4", "right", "X:4Y:4", 0.7,-1),
Transition("X:3Y:4", "right", "X:3Y:4", 0.3,-1),
Transition("X:3Y:4", "left", "X:2Y:4", 0.6,-1),
Transition("X:3Y:4", "left", "X:3Y:4", 0.4,-1),
Transition("X:3Y:4", "up", "X:3Y:5", 0.5,-1),
Transition("X:3Y:4", "up", "X:3Y:4", 0.5,-1),
Transition("X:3Y:4", "down", "X:3Y:3", 0.4,22),
Transition("X:3Y:4", "down", "X:3Y:4", 0.6,-1),
Transition("X:3Y:5", "right", "X:4Y:0", 0.7,-1),
Transition("X:3Y:5", "right", "X:3Y:5", 0.3,-1),
Transition("X:3Y:5", "left", "X:2Y:0", 0.6,-1),
Transition("X:3Y:5", "left", "X:3Y:5", 0.4,-1),
Transition("X:3Y:5", "up", "X:3Y:6", 0.5,-1),
Transition("X:3Y:5", "up", "X:3Y:5", 0.5,-1),
Transition("X:3Y:5", "down", "X:3Y:4", 0.4,-1),
Transition("X:3Y:5", "down", "X:3Y:5", 0.6,-1),
Transition("X:3Y:6", "right", "X:4Y:1", 0.7,-1),
Transition("X:3Y:6", "right", "X:3Y:6", 0.3,-1),
Transition("X:3Y:6", "left", "X:2Y:1", 0.6,-1),
Transition("X:3Y:6", "left", "X:3Y:6", 0.4,-1),
Transition("X:3Y:6", "up", "X:3Y:0", 0.5,-1),
Transition("X:3Y:6", "up", "X:3Y:6", 0.5,-1),
Transition("X:3Y:6", "down", "X:3Y:5", 0.4,-1),
Transition("X:3Y:6", "down", "X:3Y:6", 0.6,-1),
Transition("X:4Y:0", "right", "X:0Y:0", 0.7,-1),
Transition("X:4Y:0", "right", "X:4Y:0", 0.3,-1),
Transition("X:4Y:0", "left", "X:3Y:0", 0.6,-1),
Transition("X:4Y:0", "left", "X:4Y:0", 0.4,-1),
Transition("X:4Y:0", "up", "X:4Y:1", 0.5,-1),
Transition("X:4Y:0", "up", "X:4Y:0", 0.5,-1),
Transition("X:4Y:0", "down", "X:4Y:6", 0.4,-1),
Transition("X:4Y:0", "down", "X:4Y:0", 0.6,-1),
Transition("X:4Y:1", "right", "X:0Y:1", 0.7,-1),
Transition("X:4Y:1", "right", "X:4Y:1", 0.3,-1),
Transition("X:4Y:1", "left", "X:3Y:1", 0.6,-1),
Transition("X:4Y:1", "left", "X:4Y:1", 0.4,-1),
Transition("X:4Y:1", "up", "X:4Y:2", 0.5,-1),
Transition("X:4Y:1", "up", "X:4Y:1", 0.5,-1),
Transition("X:4Y:1", "down", "X:4Y:0", 0.4,-1),
Transition("X:4Y:1", "down", "X:4Y:1", 0.6,-1),
Transition("X:4Y:2", "right", "X:0Y:2", 0.7,-1),
Transition("X:4Y:2", "right", "X:4Y:2", 0.3,-1),
Transition("X:4Y:2", "left", "X:3Y:2", 0.6,-1),
Transition("X:4Y:2", "left", "X:4Y:2", 0.4,-1),
Transition("X:4Y:2", "up", "X:4Y:3", 0.5,-1),
Transition("X:4Y:2", "up", "X:4Y:2", 0.5,-1),
Transition("X:4Y:2", "down", "X:4Y:1", 0.4,-1),
Transition("X:4Y:2", "down", "X:4Y:2", 0.6,-1),
Transition("X:4Y:3", "right", "X:0Y:3", 0.7,-1),
Transition("X:4Y:3", "right", "X:4Y:3", 0.3,-1),
Transition("X:4Y:3", "left", "X:3Y:3", 0.6,22),
Transition("X:4Y:3", "left", "X:4Y:3", 0.4,-1),
Transition("X:4Y:3", "up", "X:4Y:4", 0.5,-1),
Transition("X:4Y:3", "up", "X:4Y:3", 0.5,-1),
Transition("X:4Y:3", "down", "X:4Y:2", 0.4,-1),
Transition("X:4Y:3", "down", "X:4Y:3", 0.6,-1),
Transition("X:4Y:4", "right", "X:0Y:4", 0.7,-1),
Transition("X:4Y:4", "right", "X:4Y:4", 0.3,-1),
Transition("X:4Y:4", "left", "X:3Y:4", 0.6,-1),
Transition("X:4Y:4", "left", "X:4Y:4", 0.4,-1),
Transition("X:4Y:4", "up", "X:4Y:5", 0.5,-1),
Transition("X:4Y:4", "up", "X:4Y:4", 0.5,-1),
Transition("X:4Y:4", "down", "X:4Y:3", 0.4,-1),
Transition("X:4Y:4", "down", "X:4Y:4", 0.6,-1),
Transition("X:4Y:5", "right", "X:0Y:0", 0.7,-1),
Transition("X:4Y:5", "right", "X:4Y:5", 0.3,-1),
Transition("X:4Y:5", "left", "X:3Y:0", 0.6,-1),
Transition("X:4Y:5", "left", "X:4Y:5", 0.4,-1),
Transition("X:4Y:5", "up", "X:4Y:6", 0.5,-1),
Transition("X:4Y:5", "up", "X:4Y:5", 0.5,-1),
Transition("X:4Y:5", "down", "X:4Y:4", 0.4,-1),
Transition("X:4Y:5", "down", "X:4Y:5", 0.6,-1),
Transition("X:4Y:6", "right", "X:0Y:1", 0.7,-1),
Transition("X:4Y:6", "right", "X:4Y:6", 0.3,-1),
Transition("X:4Y:6", "left", "X:3Y:1", 0.6,-1),
Transition("X:4Y:6", "left", "X:4Y:6", 0.4,-1),
Transition("X:4Y:6", "up", "X:4Y:0", 0.5,-1),
Transition("X:4Y:6", "up", "X:4Y:6", 0.5,-1),
Transition("X:4Y:6", "down", "X:4Y:5", 0.4,-1),
Transition("X:4Y:6", "down", "X:4Y:6", 0.6,-1),


)

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)
