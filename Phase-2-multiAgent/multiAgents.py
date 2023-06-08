# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        i = 0
        edible = list()

        if successorGameState.isWin():
            return float('inf')

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            gDistance = manhattanDistance(newPos, ghostPos)
            if newScaredTimes[i] <= gDistance:
                if gDistance < 2:
                    return -float('inf')
            else:
                edible.append(ghostPos)
            i += 1

        min_distance = float('inf')
        edible.extend(newFood.asList())

        for food in edible:
            distance = manhattanDistance(newPos, food)
            if distance < min_distance:
                min_distance = distance

        return successorGameState.getScore() - currentGameState.getScore() + 1 / min_distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def max(self, gameState, agentIndex, depth):
        maximum = -float("inf")
        nextAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.minimax(successor, agentIndex + 1, depth)
            if maximum < v['utility']:
                maximum = v['utility']
                nextAction = action
        return {
            'utility': maximum,
            'action': nextAction
        }

    def min(self, gameState, agentIndex, depth):
        minimum = float("inf")
        nextAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.minimax(successor, agentIndex + 1, depth)
            if minimum > v['utility']:
                minimum = v['utility']
                nextAction = action
        return {
            'utility': minimum,
            'action': nextAction
        }

    def minimax(self, gameState, agentIndex=0, depth=0):

        agentIndex = agentIndex % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return {
                'utility': self.evaluationFunction(gameState),
                'action': None
            }

        if agentIndex == 0:
            if depth < self.depth:
                return self.max(gameState, agentIndex, depth + 1)
            else:
                return {
                    'utility': self.evaluationFunction(gameState),
                    'action': None
                }
        else:
            return self.min(gameState, agentIndex, depth)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState)['action']
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        maximum = -float("inf")
        nextAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.minimax(successor, agentIndex + 1, depth, alpha, beta)
            if maximum < v['utility']:
                maximum = v['utility']
                nextAction = action
            if maximum > beta:
                return {
                    'utility': maximum,
                    'action': nextAction
                }
            if maximum > alpha:
                alpha = maximum
        return {
            'utility': maximum,
            'action': nextAction
        }

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        minimum = float("inf")
        nextAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.minimax(successor, agentIndex + 1, depth, alpha, beta)
            if minimum > v['utility']:
                minimum = v['utility']
                nextAction = action
            if minimum < alpha:
                return {
                    'utility': minimum,
                    'action': nextAction
                }
            if minimum < beta:
                beta = minimum
        return {
            'utility': minimum,
            'action': nextAction
        }

    def minimax(self, gameState, agentIndex=0, depth=0, alpha=float("-inf"), beta=float("inf")):

        agentIndex = agentIndex % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return {
                'utility': self.evaluationFunction(gameState),
                'action': None
            }

        if agentIndex == 0:
            if depth < self.depth:
                return self.maxValue(gameState, agentIndex, depth + 1, alpha, beta)
            else:
                return {
                    'utility': self.evaluationFunction(gameState),
                    'action': None
                }
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState)['action']
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max(self, gameState, agentIndex, depth):
        maximum = -float("inf")
        nextAction = None
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = self.expectimax(successor, agentIndex + 1, depth)
            if maximum < v['utility']:
                maximum = v['utility']
                nextAction = action
        return {
            'utility': maximum,
            'action': nextAction
        }

    def exp(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        result = 0
        p = 1.0 / len(legalActions)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            result += p * self.expectimax(successor, agentIndex + 1, depth)['utility']
        return {
            'utility': result,
            'action': None
        }

    def expectimax(self, gameState, agentIndex=0, depth=0):

        agentIndex = agentIndex % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return {
                'utility': self.evaluationFunction(gameState),
                'action': None
            }

        if agentIndex == 0:
            if depth < self.depth:
                return self.max(gameState, agentIndex, depth + 1)
            else:
                return {
                    'utility': self.evaluationFunction(gameState),
                    'action': None
                }
        else:
            return self.exp(gameState, agentIndex, depth)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState)['action']
        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    capsules = currentGameState.getCapsules()

    value = 0
    fDistances = list()
    for food in foods.asList():
        fDistances.append(manhattanDistance(pacmanPosition, food))

    if len(fDistances) > 0:
        value -= 1 / 100 * (max(fDistances) + min(fDistances))

    i = 0
    minGhostDistance = float('inf')
    for ghostPos in ghostPositions:
        distance = manhattanDistance(pacmanPosition, ghostPos)
        if scaredTimers[i] > distance and distance < minGhostDistance:
            minGhostDistance = distance
        elif distance > 3:
            value += 3
        elif distance == 1:
            return -float('inf')
        else:
            value += distance
    i += 1

    if minGhostDistance != float('inf'):
        value += 1.0 / minGhostDistance
    return value + currentGameState.getScore() - 20 * len(capsules)


    # capsules = currentGameState.getCapsules()
    #
    # avg = lambda x: float(sum(x)) / len(x)
    # score = currentGameState.getScore()
    #
    # foodDistances = [1.0 / util.manhattanDistance(pacmanPosition, food) for food in foods]
    # capsuleDistances = [1.0 / util.manhattanDistance(pacmanPosition, capsule) for capsule in capsules]
    # ghostDistances = [util.manhattanDistance(pacmanPosition, ghost) for ghost in ghostPositions]
    #
    #
    #
    # foodScore = 10 * avg(foodDistances) if foodDistances else float("inf")
    #
    # if capsuleDistances:
    #     capsuleScore = 10 * avg(capsuleDistances) * (1 if sum(scaredTimers) == 0 else -1)
    # else:
    #     capsuleScore = 0
    #
    # if ghostDistances:
    #     ghostScore = min(ghostDistances) / 2.0 * (0.7 if sum(scaredTimers) > 0 else -1.5)
    # else:
    #     ghostScore = 0
    #
    # return score + foodScore + capsuleScore + 0.5 * ghostScore

    # total_score = 0
    # closestFoodManDistance = float('inf')
    # closestGhostManDistance = float('inf')
    #
    # for food in foods.asList():
    #     if manhattanDistance(food, pacmanPosition) < closestFoodManDistance:
    #         closestFoodManDistance = manhattanDistance(food, pacmanPosition)
    #
    #
    #
    # for ghost_pos in ghostPositions:
    #     if manhattanDistance(ghost_pos, pacmanPosition) < closestGhostManDistance:
    #         closestGhostManDistance = manhattanDistance(ghost_pos, pacmanPosition)
    #
    # scaredGhostNumber = 0
    # for remained_time in scaredTimers:
    #     if remained_time > 0:
    #         scaredGhostNumber += 1
    #
    # total_score -= (100 * len(currentGameState.getCapsules()))
    # total_score -= (10 * len(foods.asList()))
    # if len(foods.asList()) != 0:
    #     total_score -= (5 * closestFoodManDistance)
    # total_score -= scaredGhostNumber
    # if closestGhostManDistance < 2:
    #     if closestGhostManDistance == 0:
    #         total_score -= (10000 * (closestGhostManDistance + 1))
    #     else:
    #         total_score -= (10000 * closestGhostManDistance)
    #
    # return total_score


# Abbreviation
better = betterEvaluationFunction
