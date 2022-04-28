# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from asyncio import coroutines
from logging import exception
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    start = problem.getStartState()
    c = problem.getStartState()
    # for storing explored nodes 
    explored = []
    # first we explore start node so we append it to explored list 
    explored.append(start)
    # frontier is a stack 
    frontier = util.Stack()
    # this tuple contains the start node and list of actions which is empty 
    stateTuple = (start , [])
    # push the stateTuple to frontier 
    frontier.push(stateTuple)
    # till frontier isn't empty and we didn't reach the goal state we explore 
    if problem.isGoalState(start) :
        return []
    while not frontier.isEmpty() and not problem.isGoalState(c) :
        # frontier contains stateTuple here we pop it to two variables the first is state and the second is the action 
        state, actions = frontier.pop()
        # here the state is explore so we append it to explored list 
        explored.append(state)
        # initializing successors 
        successors = problem.getSuccessors(state)
        # so if you print start node successor there is a tuple and the first item is coordinates because of that we initialized coordinates 
        for successor in successors :
            coordinates = successor[0]
            # here we make graph search so we don't need to check all the nodes 
            if not coordinates in explored :
                c = successor[0]
                direction = successor[1]
                path = actions + [direction]
                frontier.push((coordinates, path))
    return path

    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()
    explored = []
    start = problem.getStartState()
    # check if player is in goal state at first or not
    if problem.isGoalState(start):
        return []
    else:
        frontier.push((start,[]))

        while(True):

            # check that we have any state to go
            if frontier.isEmpty():
                return []
            else:
                # returns us our current state
                currentState,path = frontier.pop()
                explored.append(currentState)

                # Check that if there is goal or not
                if problem.isGoalState(currentState):
                    return path
                else:
                    successors = problem.getSuccessors(currentState)

                    # new states will add here
                    if successors:
                        for successor in successors:
                            i = 0
                            for exploredItem in explored:
                                if (successor[0] == exploredItem):
                                    i = 1
                            for state in frontier.list:
                                if (successor[0] == state[0]):
                                    i = 1
                            if (i == 1):
                                continue
                            newPath = path + [successor[1]]
                            frontier.push((successor[0],newPath))
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start = problem.getStartState()
    frontier = util.PriorityQueue()
    explored = []
    if problem.isGoalState(start) :
        return []
    
    frontier.push((start, [], 0),0)
    while not frontier.isEmpty():
        state = frontier.pop()
        if problem.isGoalState(state[0]): 
            return state[1]
        if state[0] not in explored :
            explored.append(state[0])
            successors = problem.getSuccessors(state[0])
            for successor in successors : 
                if successor[0] not in explored :
                    path = state[1] + [successor[1]]
                    frontier.push((successor[0],path), problem.getCostOfActions(path))
    return path


    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
     
    start = problem.getStartState()
    # frontier: (position,direction,cost) 
    frontier = util.PriorityQueue()
    explored = []

    if problem.isGoalState(start):
        return []

    frontier.push((start, [], 0), 0)

    while True:
        cur_state = frontier.pop()

        if problem.isGoalState(cur_state[0]):
            return cur_state[1]

        if cur_state[0] not in explored:
            explored.append(cur_state[0])
            successors = problem.getSuccessors(cur_state[0])
            for successor in successors:
                if successor[0] not in explored:
                    cost = cur_state[2] + successor[2]
                    totalCost = cost + heuristic(successor[0], problem)
                    frontier.push((successor[0], cur_state[1] + [successor[1]], cost), totalCost)
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
