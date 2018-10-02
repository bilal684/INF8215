# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani, Mohammed Esseddik Ben Yahia, Xiangyi Zhang
"""

import numpy as np
import time
import copy
import heapq

def find_cycle(pi, vertex, visited=None):
    """
    """
    if visited is None:
        visited = []
    if vertex in visited:
        visited += [vertex]
        return visited
    if pi[vertex] is not None:
        visited += [vertex]
        return find_cycle(pi, pi[vertex][1], visited)
    else:
        return []

def getCycle(theCycle):
    if len(theCycle) > 0:
        for i in range(len(theCycle) - 2, -1, -1):
            if theCycle[i] == theCycle[-1]:
                return theCycle[i:]


def MST(not_visited, graph):
    """
    """
    pi = constructPi(not_visited, graph)
    #for v in range(0, len(graph)):

    
def constructPi(not_visited, graph):
    pi = []
    for i in range(0, len(graph)):
        pi[i] = None
    for i in range(0, len(not_visited)):
        currentMin = (999999, 0)
        for j in range(0, len(graph)):
            if(graph[j][i] > 0):
                if currentMin[0] > graph[j][i]:
                    currentMin = (graph[j][i], j)
        pi[i] = currentMin
    return pi


def minimum_spanning_arborescence(sol):
    """
    Returns the cost to reach the vertices in the unvisited list 
    """
    
    

def A_star(graph, places):
    """
    Performs the A* algorithm
    """

    # blank solution
    root = Solution(graph=graph, places=places)
    # search tree T
    T = []
    heapq.heapify(T)
    heapq.heappush(T, root)
    bestSol = None
    while len(T) > 0:
        bestSol = heapq.heappop(T)
        if len(bestSol.not_visited) == 0:
            break
        for i in range(0, len(bestSol.not_visited) - 1):
            newSol = copy.deepcopy(bestSol)
            newSol.add(i)
            heapq.heappush(T, newSol)
        if len(bestSol.not_visited) == 1:
            newSol = copy.deepcopy(bestSol)
            newSol.add(0)
            heapq.heappush(T, newSol)
    return bestSol




class Solution:
    childsCount = 0
       
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.graph = graph 
        self.visited = [places[0]] # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])
        self.h = 0 # Estimated cost
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))
        if len(self.not_visited) > 0:
            self.h = minimum_spanning_arborescence(self)
        else:
            self.h = 0

    def addForDijkstra(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))

    def __lt__(self, other):
        if self.g + self.h == other.g + other.h:
            if len (self.visited) > len(other.visited):
                return True
            else:
                return False
        return self.g + self.h < other.g + other.h
    

        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')



pi = [(0, 1), (0, 4), (0, 1), (0,2), (0,3), (0,3), (0,5)]
P = find_cycle(pi, 6)
print(getCycle(P))

#1, 4, 3, 2
#graph = read_graph()
##test 3  --------------  OPT. SOL. = 26
#start_time = time.time()
#places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
##astar_sol = A_star(graph=graph, places=places)
#print(astar_sol.g)
#print(astar_sol.visited)
#print("--- %s seconds ---" % (time.time() - start_time))
