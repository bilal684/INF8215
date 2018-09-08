# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani
"""

import numpy as np

import copy

from queue import Queue

def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    solution = Solution(places, graph)
    root = Node(solution)
    buildTree(graph, places, node)
    
    
    
def buildTree(graph, places, node):
    newSolution = node.solution #Get parent solution
    for i range(0, newSolution.not_visited.len - 1):
        newSolution.add(i)
        newNode
        

class Node:
    def __init__(self, solution):
        self.solution = solution
        self.childs = []
    
    def addChild(self, Node):
        self.childs.append(Node)

class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.graph = graph 
        self.visited = [places[0]] # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:]) # list of attractions not yet visited
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))
        
        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

graph = read_graph()
