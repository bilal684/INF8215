# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani, Mohammed Esseddik Ben Yahia, Xiangyi Zhang
"""

import numpy as np
import time
import copy
import heapq
import sys

class Vertex:
    visitedVertexes = []
    def __init__(self):
        self.id = None
        self.currentCost = sys.maxsize
        self.parent = None

    def __lt__(self, other):
        return self.currentCost < other.currentCost


def fastest_path_estimation(sol):
    """
    Returns the time spent on the fastest path between 
    the current vertex c and the ending vertex pm
    """
    c = sol.visited[-1]
    pm = sol.not_visited[-1]

    graph = read_graph()
    heap = []
    heapq.heapify(heap)
    currentVertex = Vertex()
    currentVertex.id = c
    currentVertex.currentCost = 0
    heapq.heappush(heap, currentVertex)
    while len(heap) > 0:
        currentVertex = heapq.heappop(heap)
        if (currentVertex.id == pm): #We're done.
            break
        for i in range(0, len(graph[currentVertex.id])):
            if(graph[currentVertex.id, i] > 0 and i not in Vertex.visitedVertexes):
                adjacentVertex = Vertex()
                adjacentVertex.id = i
                newCost = currentVertex.currentCost + graph[currentVertex.id, i]
                if(newCost < adjacentVertex.currentCost):
                    adjacentVertex.currentCost = newCost
                    adjacentVertex.parent = currentVertex
                    heapq.heappush(heap, adjacentVertex)
        Vertex.visitedVertexes.append(currentVertex.id)
    return currentVertex.currentCost
    #T = []
    #heapq.heapify(T)
    #root = copy.deepcopy(sol)
    #root.g = 0
    #heapq.heappush(T, (root.g, root))
    #bestSolution = None
    #while T:
    #    currentSolution = heapq.heappop(T)[1]
    #    i = 0
    #    #TODO POSER LA QUESTION AU CHARGER est-ce que je peux return ici directement ou je dois absolument m'assurer que le heap soit vide??
    #    if(len(currentSolution.not_visited) == 0):
    #        return currentSolution.g
    #    #if len(currentSolution.not_visited)  == 0:
        #    if bestSolution is None:
        #        bestSolution = currentSolution
        #    elif bestSolution.g > currentSolution.g:
        #        bestSolution = currentSolution
    #    while i < len(currentSolution.not_visited) - 1:
    #        newSolution = copy.deepcopy(currentSolution)
    #        newSolution.add(i)
    #        heapq.heappush(T, (newSolution.g, newSolution))
    #        i+=1
    #    if len(currentSolution.not_visited) == 1:
    #        newSolution = copy.deepcopy(currentSolution)
    #        newSolution.add(0)
    #        heapq.heappush(T, (newSolution.g, newSolution))
    #return bestSolution.g

    
    #currentCost = sol.g



class Solution:
    childsCount = 0
       
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.h = 0 # Cout restant vers la solution finale.
        self.graph = graph 
        self.visited = [places[0]] # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])
        #self.not_visited = copy.deepcopy(places[1:]) # list of attractions not yet visited
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))

    def __lt__(self, other):
        return ((self.g + self.h) <= (other.g + other.h))
    

        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

graph = read_graph()
 
#test 1  --------------  OPT. SOL. = 27
start_time = time.time()

places=[0, 5, 13, 16, 6, 9, 4]
sol = Solution(places, graph)
fastestPath = fastest_path_estimation(sol)
#sol = bfs(graph=graph, places=places)
print(fastestPath)
print("--- %s seconds ---" % (time.time() - start_time))
