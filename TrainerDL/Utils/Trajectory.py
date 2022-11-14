
from abc import ABC, abstractmethod
import random

class Trajectory(ABC):

	""" Abstract Class defining an API to access trajectories """

	def __init__(self):
		pass


	@abstractmethod
	def getLength(self):
		""" Length of the trajectory corresponds to #States.
		Because the terminal state does not have a action assiociated with it, it is assumed that #Actions = #States - 1 """
		pass # To implement by children



	@abstractmethod
	def getState(self, index):
		pass # To implement by children



	@abstractmethod
	def getAction(self, index):
		pass # To implement by children



	def getStateTransition(self, index):
		""" A state-transition, as opposed to stateAction-transition specifies how the state changed under the effect of an action.
		State-Transition_t := [State_t, Action_t, State_t+1] """
		if (index < 0 or index >= (self.getLength() - 1)):
			raise IndexError(f"Simple transition cannot be accessed outside []")
		return [self.getState(index), self.getAction(index), self.getState(index + 1)]

	def getTransition(self, index):
		""" State-transitions are the default """
		return self.getStateTransition(index)



	def getStateActionTransition(self, index):
		""" A state-action-transition, as opposed to state-transition specifies how the state-action pair evolved in the trajectory.
		State-Action-Transition_t := [State_t, Action_t, State_t+1, Action_t+1] """
		if (index < 0 or index >= (self.getLength() - 2)):
			raise IndexError(f"Simple transition cannot be accessed outside []")
		return [self.getState(index), self.getAction(index), self.getState(index + 1), self.getAction(index + 1)]



	def iterate(self, randomized=False, includeNextAction=False):
		""" Simplify iteration trough the trajectory """

		n_indices = self.getLength() - 1 # There are 1 less transition than the overall length of the trajectory
		if (includeNextAction) : n_indices -= 1 # The terminal state does not have an action assiciated with it : There are 1 less state-ACTION transition

		indices = range(n_indices)

		if (randomized):
			random.shuffle(indices)

		for index in indices :
			if (includeNextAction) : yield self.getStateActionTransition(index)
			else : yield self.getTransition(index)



	def __call__(self, index):
		""" Sugar coating to simplify transition access """
		return self.getStateTransition(index)



	def print(self) :
		for index, (state, action, nextState) in enumerate(self.iterate(randomized=False)) :
			print(f"Trajectory({index}) : State={state}\t\tAction={action}\t\tNextState={nextState}")
