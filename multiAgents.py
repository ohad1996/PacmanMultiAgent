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
        # Not useful if pacman isn't moving!
        if action is 'Stop':
            return float('-inf')

        # If pacman is nearby a ghost
        if min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]) <= 1:
            # in other cases pacman needs to run away!
            return float('-inf')

        # Incase pacman stepped on a food
        if currentGameState.getNumFood() is not newFood.count():
            return float('inf')

        # Finding the closest food!
        minFoodDistance = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        return -minFoodDistance


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
        pacmanIndex = 0
        minimaxAns, action = self.minimaxFunc(gameState, currDepth=self.depth, agentIndex=pacmanIndex)
        return action

    def minimaxFunc(self, gameState, currDepth, agentIndex):
        pacmanIndex = 0
        # the function will stop if we reached the deepest depth -
        # when its the minimax calculation of evaluationFunction!
        if currDepth is 0 or not gameState.getLegalActions(agentIndex) \
                or gameState.isWin() \
                or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # generate all the sons states of the current state
        StateSons = [(gameState.generateSuccessor(agentIndex, action), action) for action in
                     gameState.getLegalActions(agentIndex)]

        nextAgent = agentIndex + 1
        # if the nextAgent is pacman - we will need to finish the current turn and all the
        # ghosts, and the depth should be decreased.
        if nextAgent is pacmanIndex or nextAgent is gameState.getNumAgents():
            nextAgent = pacmanIndex
            currDepth -= 1

        # recursive calculation for all the minimax situations!
        allMinimaxes = [(self.minimaxFunc(state, currDepth, nextAgent)[0], action) for state, action in StateSons]

        # if it's pacman turn (max player)
        if agentIndex is pacmanIndex:
            if len(allMinimaxes) is 0:
                return float('-inf')
            return max(allMinimaxes)

        # if it's a ghost turn (min player)
        else:
            if len(allMinimaxes) is 0:
                return float('inf')
            return min(allMinimaxes)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacmanIndex = 0
        action, score = self.alphaBetaFunc(self.depth, pacmanIndex, float('-inf'), float('inf'), gameState)
        return action

    def alphaBetaFunc(self, currDepth, agentIndex, alpha, beta, gameState):
        pacmanIndex = 0
        # Incase all agents have finished playing their turn in a move -
        # starting pacman's move again and increase current depth
        if agentIndex is gameState.getNumAgents():
            agentIndex = pacmanIndex
            currDepth -= 1

        # We will stop when we reached to the last depth
        # or we have no moves left
        # or we are at winning state
        # or at losing state.
        if currDepth is 0 or\
                not gameState.getLegalActions(agentIndex) \
                or gameState.isWin() \
                or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # initialize the bestScore&bestAction that pacman or the ghost did!
        bestScore = bestAction = 0

        # incase its pacman turn (max)
        if agentIndex is pacmanIndex:
            for action in gameState.getLegalActions(agentIndex):

                # continues the new game state that generated by pacman's
                # action and the current alpha and beta values
                nextGameState = gameState.generateSuccessor(agentIndex, action)

                # recursively continues to the ghosts agents by increasing
                # the agent index and to seeking out
                # to the best score - the maximum score
                curScore = self.alphaBetaFunc(currDepth, agentIndex + 1, alpha, beta, nextGameState)[1]

                # checks which ghost gave the best score!
                if bestScore is 0 or curScore > bestScore:
                    bestScore = curScore
                    bestAction = action

                # alpha is the biggest score of all the legal actions!
                alpha = max([alpha, curScore])

                # incase alpha is bigger than beta - prune the tree
                if alpha > beta:
                    break

        # Incase its ghost turn (min)
        else:
            for action in gameState.getLegalActions(agentIndex):  # For each legal action of ghost agent

                # generate the new game states sons for the agent and from there -
                # checks recursively which son game the best score - the minimum score!
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                curScore = self.alphaBetaFunc(currDepth, agentIndex + 1, alpha, beta, nextGameState)[1]

                # updating the best score&action that found
                if bestScore is 0 or curScore < bestScore:
                    bestScore = curScore
                    bestAction = action

                # beta is the smallest score of all the legal actions!
                beta = min([beta, curScore])

                # incase beta is less than alpha - prune the tree
                if alpha > beta:
                    break

        # if we couldn't get any bestScore -
        # we will need to return the score of the current gameState in evaluationFunction
        if bestScore is 0:
            return None, self.evaluationFunction(gameState)

        # return the bestAction and bestScore of the current agent!
        return bestAction, bestScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacmanIndex = 0
        action, score = self.expectiMaxFunc(self.depth, pacmanIndex, gameState)
        return action

    def expectiMaxFunc(self, currDepth, agentIndex, gameState):
        pacmanIndex = 0

        # incase all agents have finished playing their turn in a move -
        # starting pacman's move again and increase current depth
        if agentIndex is gameState.getNumAgents():
            agentIndex = pacmanIndex
            currDepth -= 1

        # We will stop when we reached to the last depth
        # or we have no moves left
        # or we are at winning state
        # or at losing state.
        if currDepth is 0 or not gameState.getLegalActions(agentIndex) \
                or gameState.isWin() \
                or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        bestScore = bestAction = 0

        # incase its pacman turn (max)
        if agentIndex is pacmanIndex:

            for action in gameState.getLegalActions(agentIndex):

                # gets the expectimax score of successor
                nextGameState = gameState.generateSuccessor(agentIndex, action)

                # continues to the next ghost agent (or maby pacman..)
                # by increasing the agent index
                # and gaining their scores
                # in this current depth
                score = self.expectiMaxFunc(currDepth, agentIndex + 1, nextGameState)[1]

                # Updating the best score and action
                if bestScore < score:
                    bestScore = score
                    bestAction = action

        # incase its ghost turn (min)
        else:
            curGhostActions = gameState.getLegalActions(agentIndex)

            # initialize the chance of turning for the current ghost
            ActionChance = float(len(curGhostActions))

            for action in curGhostActions:

                # continues to the next ghost agent (or maby pacman..) for
                # getting all the expectiMax values from all other agents
                # by increasing the agent's index and gaining their scores
                # in this current depth
                nextGameState = gameState.generateSuccessor(agentIndex, action)
                score = self.expectiMaxFunc(currDepth, agentIndex + 1, nextGameState)[1]

                # calculating the average of the ghosts scores that splits in the same
                # possibility of choice
                bestScore += score / ActionChance
                bestAction = action

        # return the bestAction and bestScore of the current agent!
        return bestAction, bestScore


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    for make pacman to be a ghost-hunting - pacman needs the capsules so we needed to lower the
    the score any time he chooses to do not use the capsule!

    for make pacman to be a food-gobbling - we needed to show him that if he turn to a food
    direction - he will have more for his score but still be warn from the non scared ghosts!

    and that what will make pacman unstoppable! :)
    """
    "*** YOUR CODE HERE ***"

    capsules = currentGameState.getCapsules()
    newFood = currentGameState.getFood().asList()
    pacmanPosition = currentGameState.getPacmanPosition()

    # the less capsules - the more pacman used them & the more scared ghosts!
    if len(capsules) is not 0:
        capsules = -10
    else:
        capsules = 0

    minFoodDist = [manhattanDistance(pacmanPosition, food) for food in newFood]
    if len(minFoodDist) is not 0:
        minFoodDist = min(minFoodDist)
    else:
        minFoodDist = 1
    # for to increase the score - we would like to make pacman
    # understand that food will increase the score the less distance
    # between pacman to the closest food!
    minFoodDistChance = 1.0 / minFoodDist

    # the more pacman is closed to a ghost - the less the score it will be!
    totalGhostsDistances = 0
    for ghostPosition in currentGameState.getGhostPositions():
        curGhostDistance = manhattanDistance(pacmanPosition, ghostPosition)
        totalGhostsDistances += curGhostDistance

    # the total score
    return currentGameState.getScore() - totalGhostsDistances + minFoodDistChance + capsules


# Abbreviation
better = betterEvaluationFunction
