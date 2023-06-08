# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def iterationStep(self):
        values = util.Counter()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                values[state] = -sys.maxsize
                for action in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, action)
                    if values[state] < q:
                        values[state] = q
                if values[state] == -sys.maxsize:
                    values[state] = 0

        return values

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            self.values = self.iterationStep()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sumValues = 0.0
        for nextState, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, nextState)
            sumValues += p * (r + self.discount * self.values[nextState])
        return sumValues
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action = None
        value = -sys.maxsize
        for currentAction in self.mdp.getPossibleActions(state):
            currentValue = self.computeQValueFromValues(state, currentAction)
            if currentValue > value:
                action = currentAction
                value = currentValue
        return action
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def iterationStep(self, state):
        if not self.mdp.isTerminal(state):
            value = -sys.maxsize
            for action in self.mdp.getPossibleActions(state):
                q = self.computeQValueFromValues(state, action)
                if value < q:
                    value = q
            if value == -sys.maxsize:
                value = 0
            return value
        else:
            return 0

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        statesSize = len(states)
        for i in range(self.iterations):
            state = states[i % statesSize]
            self.values[state] = self.iterationStep(state)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getPredecessors(self):
        predecessors = dict()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, p in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState not in predecessors.keys():
                        predecessors[nextState] = set()
                    predecessors[nextState].add(state)

        return predecessors

    def computeDiff(self, state):
        maxQ = -sys.maxsize
        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if maxQ < q:
                maxQ = q
        if maxQ == -sys.maxsize:
            maxQ = 0

        return abs(self.values[state] - maxQ)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = self.getPredecessors()
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = self.computeDiff(state)
                queue.push(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                return

            state = queue.pop()

            if not self.mdp.isTerminal(state):
                value = -sys.maxsize
                for action in self.mdp.getPossibleActions(state):
                    q = self.computeQValueFromValues(state, action)
                    if value < q:
                        value = q
                if value == -sys.maxsize:
                    value = 0
                self.values[state] = value
            else:
                self.values[state] = value

            for predecessor in predecessors[state]:
                diff = self.computeDiff(predecessor)
                if diff > self.theta:
                    queue.update(predecessor, -diff)


