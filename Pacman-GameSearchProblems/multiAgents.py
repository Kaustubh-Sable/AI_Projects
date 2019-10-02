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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        currentFoodState = currentGameState.getFood()

        # variable to score an evaluation of the current state of ReflexAgent
        preference = 0

        # determining preference of the action based on food.
        for i in range(0, len(currentFoodState.data)):
            for j in range(0, len(currentFoodState.data[i])):
                if currentFoodState[i][j] is True:
                    dist = util.manhattanDistance(newPos,(i,j))
                    if dist == 0:
                        preference = preference + 100
                    else:
                        preference = preference + 21/(dist)

        # determining preference of the action based on capsules.
        power_pellet = currentGameState.getCapsules()
        for pellet in power_pellet:
            dist = util.manhattanDistance(newPos, pellet)
            if dist==0:
                preference = preference + 1000
            else:
                preference = preference + 100/dist

        # determining preference of the action based on ghosts.
        ghost_weight = 0
        for ghost in newGhostStates:
            d = util.manhattanDistance(newPos, ghost.getPosition())
            if d<=1:
                if ghost.scaredTimer>0:
                    # eat pacman if already eaten a pallet
                    ghost_weight = 99999
                else:
                    preference = preference - 320

        if ghost_weight!=0:
            return ghost_weight
        else:
            return preference

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"

        agent_index = 0

        # Collect legal moves and successor states
        depth = 1

        # function call to get next action and score for particular agent
        action, score = self.getActionScore('', gameState, depth, agent_index)

        return action

    def getActionScore(self, action, gameState, depth, agent_index):
        """
            Recursive function to get next action and score based on the current game state, depth and type of agent.
        """
        if agent_index == gameState.getNumAgents():
            if depth >= self.depth:
                # terminating condition : leaf has been reached
                return action, self.evaluationFunction(gameState)
            depth = depth + 1
            # set agent as max (pacman) when depth is increased.
            agent_index = 0

        # Collect legal moves and successor states
        legal_actions = gameState.getLegalActions(agent_index)

        if len(legal_actions) == 0:
            # leaf level reached
            return action, self.evaluationFunction(gameState)

        scores = []

        # evaluating each possible action
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_index, action)
            ac, score = self.getActionScore(action, successor_state, depth, agent_index+1)
            scores.append((action, score))

        bestScore = 0
        if agent_index==0:
            # if agent is max, take the maximum score and return the corresponding action with it.
            if len(scores)==0:
                return 'center', -9999
            bestScore = -9999
            best_action = ''
            for action,score in scores:
                if score > bestScore:
                    bestScore = score
                    best_action = action
        else:
            # if agent is min, take the minimum score and return the corresponding action with it.
            if len(scores)==0:
                return 'center', 9999
            bestScore = 9999
            best_action = ''
            for action, score in scores:
                if score < bestScore:
                    bestScore = score
                    best_action = action

        return best_action, bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        agent_index = 0
        depth = 1

        # function call to get next action and score for particular agent
        action, score = self.getActionScore('', gameState, depth, agent_index)

        return action

    def getActionScore(self, action, gameState, depth, agent_index):
        """
            Recursive function to get next action and score based on the current game state, depth and type of agent.
        """
        if agent_index == gameState.getNumAgents():
            if depth >= self.depth:
                # terminating condition : leaf has been reached
                return action, self.evaluationFunction(gameState)
            depth = depth + 1
            # set agent as max (pacman) when depth is increased.
            agent_index = 0

        # Collect legal moves and successor states
        legal_actions = gameState.getLegalActions(agent_index)

        if len(legal_actions) == 0:
            # leaf level reached
            return action, self.evaluationFunction(gameState)

        scores = []

        # evaluating each possible action
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(agent_index, action)
            ac, score = self.getActionScore(action, successor_state, depth, agent_index + 1)
            scores.append((action, score))

        bestScore = 0
        if agent_index == 0:
            # if agent is max, take the maximum score and return the corresponding action with it.
            if len(scores) == 0:
                return 'center', -9999
            bestScore = -9999
            best_action = ''
            for action, score in scores:
                if score > bestScore:
                    bestScore = score
                    best_action = action
        else:
            # if agent is a chance node, take the average score and return the corresponding action with it.
            if len(scores) == 0:
                return 'center', 9999
            avgScore = 0
            for action, score in scores:
                avgScore += score
            bestScore = avgScore/len(scores)
            best_action = ''

        return best_action, bestScore

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
