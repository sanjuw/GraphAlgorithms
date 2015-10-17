# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:40:43 2015

@author: Sanjeevni
"""

# Note: Need to add location of other files (graph.py, graph_functions.py)
# to sys.path as shown below for this code to run.
import sys
# print(sys.path)
sys.path.append("C:\\Users\\Sanjeevni\\code\\python_projects\\course")
from wanchoo_graph import *
from wanchoo_graph_functions import *
#--------------------
# Problem 1   
#--------------------
# Create Random Graph with 6 nodes and 10 edges
randomG = createRandomGraph(6,10)

# Randomly assign location to all nodes in randomG
# Then print to see output
G_nodelist = randomG.getNodes().values()
setRandLocation(randomG)
for v in G_nodelist:
    print(v.location)

# Set distance between nodes as edge weight
setDistance(randomG)

# Part c
distDict = {}
for v in G_nodelist:
    distDict[v.nodeid] =[(x,y) for x,y in zip(v.edgeList.keys(), v.edgeList.values())]
print(distDict)

# -----------------
# Problem 2
# -----------------
# Test
p2_graph = createRandomGraph(10,8)
connected_components(p2_graph) 
   
nodelist = p2_graph.getNodes()
for n in nodelist:
    print("Component: ", getattr(nodelist[n], 'connected_component'))

# Timing
x= list(range(0,1001,100))
runtimes = plotruntime(connected_components, 10, x)

#-------------------    
# Problem 3
#-------------------
# Test
g_test1 = createRandomGraph(10,7)
setRandLocation(g_test1)
plotGraph(g_test1)

connected_components(g_test1)
plotGraph(g_test1, True)

#--------------------
# Problem 4
#--------------------
# Test
test1 = createRandConnectedGraph(10,15)
setRandLocation(test1)

test1_MST = MST(test1, format="graph")
plotGraph(test1, showMST=True)

# Timing
x = list(range(1000,5000,500))
plotruntime(MST, 5, x)