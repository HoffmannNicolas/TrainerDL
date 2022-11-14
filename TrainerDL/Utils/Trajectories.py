

class Trajectories():

    """ Deals with a set of trajectories """

    def __init__(self, trajectories):
        self.trajectories = trajectories



    def iterate(self, randomized=False, includeNextAction=False) :
        """ Iterate through all trajectories """

        if (randomized) :
            pass # TODO : Use length of all trajectory and their total to sample random from them, in proportion of their amount of transitions.

        for trajectory in self.trajectories:
            for transition in trajectory.iterate(includeNextAction=includeNextAction):
                yield transition
