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


from sklearn.linear_model import GammaRegressor
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
    # def ghost_distance(newGhostStates, newPos) :

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # ghost_distance(newGhostStates, newPos)
        
        minGhost = float("inf")
        for state in newGhostStates :
           minGhost = min(minGhost, manhattanDistance(newPos, state.getPosition()))
           nearGhost = state
        
        minFood = float("inf")
        for foodPosition in newFood :
            minFood = min(minFood, manhattanDistance(newPos, foodPosition))
        
        minPower = float("inf")
        for powerPos in successorGameState.getCapsules() : 
            minPower = min(minPower, manhattanDistance(newPos, powerPos))

        evaluation_Score = successorGameState.getScore() 
        minFood = min(minFood, minPower)
        

        if len(newFood) > 0 or len(successorGameState.getCapsules()) > 0 :
            evaluation_Score -= (minFood * 1)
 
        if len(newGhostStates) > 0 :
            if nearGhost.scaredTimer == 0 :
                evaluation_Score += (minGhost * 0.1)
            else :
                evaluation_Score -= (minGhost)
        
        evaluation_Score -= len(newFood) * 10
        flag = 0 
        for i in newScaredTimes :
            if i > 0 :
                flag = 1 
        if flag == 0 :
            evaluation_Score -= len(successorGameState.getCapsules()) * 50 
        return evaluation_Score

        
        # return successorGameState.getScore()


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
        #  res = [score, action]
        res = self.get_val(gameState, 0, 0)
        return res[1]
    
    # i = index
    def get_val(self, gameState, i, depth) :
        # returns value as pair of score and action based on three cases :
        # 1) reach Terminal state
        # 2) Max-agent
        # 3) Min_agent

        # handling Terminal States 
        actions = gameState.getLegalActions(i)
        if len(actions) == 0 or depth == self.depth :
            scores = gameState.getScore()
            return scores, ""
        
        # if i = 0 pacman is a Max-agent 
        if i == 0 :
            return self.max_val(gameState, i, depth)
        
        # if i > 0 pacman is a Min-agent 
        else :
            return self.min_val(gameState, i, depth)
    
    def max_val(self, gameState, i, depth) :
        # returns max value for Max-agent
        max_val = float("-inf")
        max_action = ""

        for move in gameState.getLegalActions(i) :
            successor = gameState.generateSuccessor(i, move)
            succ_index = i + 1
            succ_depth = depth

            # Update the successor agent's index and depth if it's pacman
            num = gameState.getNumAgents() 
            if succ_index == num :
                succ_index = 0
                succ_depth += 1
            
            current_val = self.get_val(successor, succ_index, succ_depth)[0]

            if current_val > max_val :
                max_val = current_val
                max_action = move

        return max_val, max_action

    def min_val(self, gameState, i, depth):
        
        # Returns the min value for Min-agent
         
        min_val = float("inf")
        min_action = ""

        for move in gameState.getLegalActions(i):
            successor = gameState.generateSuccessor(i, move)
            succ_index = i + 1
            succ_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if succ_index == gameState.getNumAgents():
                succ_index = 0
                succ_depth += 1

            current_val = self.get_val(successor, succ_index, succ_depth)[0]

            if current_val < min_val:
                min_val = current_val
                min_action = move

        return min_val, min_action

        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        num = gameState.getNumAgents()
        Lose = gameState.isLose() 
        Win = gameState.isWin()
        if depth is self.depth * num \
                or Lose or Win :
                eval_func = self.evaluationFunction(gameState)
                return eval_func
        if agentIndex is 0:
            return self.maxval(gameState, agentIndex, depth, alpha, beta)[1]
        else:
            return self.minval(gameState, agentIndex, depth, alpha, beta)[1]

    def maxval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max",-float("inf"))
        legal_actions = gameState.getLegalActions(agentIndex)
        for action in legal_actions :
            generate_successor = gameState.generateSuccessor(agentIndex,action)
            num = gameState.getNumAgents()
            succAction = (action,self.alphabeta(generate_successor,
                                      (depth + 1)% num ,depth+1, alpha, beta))
            bestAction = max(bestAction,succAction,key=lambda x:x[1])

            # Prunning
            if bestAction[1] > beta: return bestAction
            else: alpha = max(alpha,bestAction[1])

        return bestAction

    def minval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("min",float("inf"))
        legal_actions = gameState.getLegalActions(agentIndex)
        for action in legal_actions :
            generate_successor = gameState.generateSuccessor(agentIndex,action)
            num = gameState.getNumAgents()
            succAction = (action,self.alphabeta(generate_successor,
                                      (depth + 1)% num,depth+1, alpha, beta))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])

            # Prunning
            if bestAction[1] < alpha: return bestAction
            else: beta = min(beta, bestAction[1])

        return bestAction
        # util.raiseNotDefined()


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
        def maxValue(depth, currentState):
            pacmanLegalActions = currentState.getLegalActions(0)
            pacmanWinState = currentState.isWin() 
            pacmanLoseState = currentState.isLose()
            if (depth > self.depth) or pacmanWinState or pacmanLoseState :
                pacman_eval_func = self.evaluationFunction(currentState)
                return pacman_eval_func

            max_Value = -1e10

            for pacmanAction in pacmanLegalActions:
                pacmanSuccessor = currentState.generateSuccessor(0, pacmanAction)
                max_Value = max(max_Value, chanceValue(depth, pacmanSuccessor, 1))
            return max_Value

        def chanceValue(depth,currentState,agentIndex):
            ghostLegalActions = currentState.getLegalActions(agentIndex)
            ghostWinState = currentState.isWin()
            ghostLoseState = currentState.isLose()
            if ghostWinState or ghostLoseState :
                ghost_eval_func = self.evaluationFunction(currentState)
                return ghost_eval_func
            
            gostChanceValue = 0
            agentCount = currentState.getNumAgents()
            if agentIndex <= agentCount - 2:
                for action in ghostLegalActions:
                    gostSuccessor = currentState.generateSuccessor(agentIndex,action)
                    gostChanceValue += chanceValue(depth, gostSuccessor, agentIndex + 1)
                    
            else:
                for action in ghostLegalActions:
                    gostSuccessor = currentState.generateSuccessor(agentIndex,action)
                    gostChanceValue += maxValue(depth + 1, gostSuccessor) / len(ghostLegalActions)
            
            return gostChanceValue

        bestValue = -1e10
        bestAction = None
        for pacmanAction in gameState.getLegalActions(0):
            pacmanSuccessorState = gameState.generateSuccessor(0, pacmanAction)
            gostValue = chanceValue(1, pacmanSuccessorState, 1)
            if gostValue > bestValue:
                bestValue = gostValue
                bestAction = pacmanAction
        return bestAction
        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    pacmanPos = currentGameState.getPacmanPosition()
    foodLeft = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsulesLeft = currentGameState.getCapsules()
    
    if len(capsulesLeft) > 0:
        capsuleDistances = []
        for capsule in capsulesLeft:
            capsuleDistances.append(manhattanDistance(capsule,pacmanPos))
        score = score - min(capsuleDistances)
    
    if len(foodLeft) > 0:
        foodDistances = []
        for food in foodLeft:
            foodDistances.append(manhattanDistance(food,pacmanPos))
        score = score - min(foodDistances)
    
    if currentGameState.hasFood(pacmanPos[0],pacmanPos[1]): score = score + 50
    if currentGameState.isLose(): score = score - 9999
    if currentGameState.isWin(): score = score + 9999 
    for capsule in capsulesLeft:
        if pacmanPos == capsule : score = score + 100


    for ghost in ghostStates: 
        if ghost.scaredTimer > 0:
            score = score + manhattanDistance(ghost.getPosition(),pacmanPos)
        else:
            score = score - manhattanDistance(ghost.getPosition(),pacmanPos)

    score = score - len(foodLeft)

    return score
    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
