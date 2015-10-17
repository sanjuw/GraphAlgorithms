# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 07:28:18 2015

@author: Sanjeevni
"""
import numpy as np
import random
from wanchoo_graph import Graph

# Create random graph with v vertices and e edges
def createRandomGraph(v,e):
    randAdjList = [] # Keep track of edges alredy called. Should ultimately contain unique set of e edges
    i = 0

    # Add check for if num edges are possible    
    max_possible_e = (v*(v-1))/2  
    if(e>max_possible_e):
        raise Exception("Error! Number of edges cannot exceed %d for %d edges" % (max_possible_e,v))
    
    while(i<e):
        # Generate a random pair of node ids (can't pick the same node twice)
        randNodePair = np.random.choice(range(1,v+1), 2, replace=False).tolist()
        randNodePairRev =  list(randNodePair)
        randNodePairRev.reverse()
        
        # Since undirected graph, check that neither a->b nor b->a has been picked
        if randNodePair not in randAdjList and randNodePairRev not in randAdjList:
            randAdjList.append(randNodePair)
            i+=1
    
    # Create random graph using randAdjList created above    
    rGraph = Graph()
    for i in range(1,v):
        rGraph.addNode(i)
    rGraph.createGraph(randAdjList)
    return rGraph
    
# Part b

# Assign random location coordinates (x,y) to each node of a graph
def setRandLocation(graph):
    vertices = graph.getNodes().values()
    for vertex in vertices:
        rloc = (random.random(), random.random())
        vertex.location = rloc
        
# Set edge weights to distance   
def setDistance(graph):
    vertexDict = graph.getNodes() # Dictionary with key: node id, value: node
    vertices = graph.getNodes().values() # List of nodes
    
    for sourceVert in vertices:
        edges = graph.getEdges(sourceVert.nodeid)
        for destVert in edges:
            euclideanDist = calc_distance(vertexDict[destVert].location, sourceVert.location)
            edges[destVert] = euclideanDist

# Calculate euclidean distance between coordinate tuples of form (x,y)
from math import sqrt
def calc_distance(coord_1, coord_2):
    return sqrt(sum((x-y)**2 for x,y in zip(coord_1, coord_2)))
        
# Problem 2
def connected_components(graph):
    all_vertices = graph.getNodes().keys()
    marked = []
    connections = []
    comp_count = 0
    
    # Loop through all the vertices...
    for vertex in all_vertices:
        # if vertex not yet visited...
        if not vertex in marked:
            # BFS returns 1 set of connected components            
            component = bfs(graph, vertex, []) # Breadth-First Search
            # Keep track of each set of connected components
            connections.append(component)
            # Add visited nodes to marked to keep track
            marked.extend(component)
            comp_count+=1 # Component Count
            setComponentNumber(graph, component, comp_count)
    return connections
        
# Depth-First Search
def dfs(graph, v, visited):
    visited.append(v)
    edges = graph.getEdges(v)
    for neighbor in edges:
        if not neighbor in visited:
            dfs(graph, neighbor, visited)
    return visited
   
# Breadth-First Search   
def bfs(graph, v, visited):
    queue = []
    queue.append(v)
    visited.append(v)
    
    while queue:
        v = queue.pop(0)
        edges = graph.getEdges(v)
        for neighbor in edges:
            if not neighbor in visited:
                queue.append(neighbor)
                visited.append(neighbor)            
    return visited
    
def setComponentNumber(graph, comp, count):
    all_nodes = graph.getNodes()    
    for nodeId in comp:
        all_nodes[nodeId].connected_component = count

# Timing
import time

# Calculate the runtimes 
def plotruntime(func, reps, x_arr, singleComponent=False):
    x_y_arr = {}
    for it in range(1,reps):
        for x in x_arr:
            if(singleComponent==True):
                graph = createRandConnectedGraph(x,3*x)
            else:
                graph = createRandomGraph(x,3*x)
            print('x = ', x)
            print("Nodes: %d, vertices: %d" % (x, 3*x))         
            timeStamp = time.process_time() # Start Time
            func(graph) # run p function
            timeLapse = time.process_time() - timeStamp
            print('timeLapse = ', timeLapse)
            
            if it==1: # Add first element, append rest 
                x_y_arr[x] = [timeLapse]
            else:
                x_y_arr[x].append(timeLapse)
       
    # Average runtimes for each x        
    for k in x_y_arr:
        x_y_arr[k] = np.mean(x_y_arr[k])

    # Plot using matplotlib.pyplot
    plt.xlabel('n')
    plt.ylabel('time (in seconds)')
    plt.title('Run times for different n\'s ')
    plt.plot(list(x_y_arr.keys()), list(x_y_arr.values()), 'ro')
    plt.show()
    return x_y_arr    
        
# Problem 3
import matplotlib.pyplot as plt

def plotGraph(graph, showComponents = False, showMST = False):
    all_nodes = graph.getNodes().values()
    nodelist = graph.getNodes()    
    
    location_dict = {}
    component_dict = {}

    # If showMST is true, create MST graph
    if showMST == True:
        adjList_MST = MST(graph, format="adjacency list")
    
    for node in all_nodes:       
        # Create node:location dictionary
        location_dict[node.nodeid] = nodelist[node.nodeid].location
        if(showComponents == True):
            component_dict[node.nodeid] = node.connected_component
        
        # Connect each pair of edges
        x1,y1 = node.location
        for neighbor in graph.getEdges(node.nodeid):
            x2,y2 = nodelist[neighbor].location
            if(showMST==False):
                plt.plot([x1, x2], [y1, y2], linestyle='-', color='k', linewidth=2)
            else:
                if[node.nodeid, neighbor] in adjList_MST or [neighbor, node.nodeid] in adjList_MST:
                    plt.plot([x1, x2], [y1, y2], linestyle='-', color='r', linewidth=2)
                else:
                    plt.plot([x1, x2], [y1, y2], linestyle='-', color='k', linewidth=2)

    # Add annotation to the plot
    x,y = zip(*location_dict.values())
    
    # If showComponents is true, color code such that 
    # nodes in the same component are the same color
    if(showComponents == True):
        plt.scatter(x,y, s=150, c=list(component_dict.values()))
    else: 
        plt.scatter(x,y, s = 150)
    for label, x, y in zip(list(location_dict.keys()), x, y):
        plt.annotate(label, xy=(x,y), xytext = (-5, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))

    # Add labels    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graph')
    plt.show()
    
# Minimum Spanning Tree
def MST(graph, format="adjacency list"):
    parent = {}
    
    # Add all nodes in graph to list of unvisited nodes
    unvisited = graph.getNodeList()
    # Maintain weight dictionary. Initialize all node weights to infinity
    weight_dict = dict({id, float("inf")} for id in graph.getNodeList())
    
    # Initialize all nodes to not visited
    for node in graph.getNodeList():
        graph.getNodes()[node].visited = False

    # Pick a random node as start node
    startNode = int(np.random.choice(list(weight_dict.keys()), 1))
    weight_dict[startNode] = 0 # Set weight of start node to 0
    parent[startNode] = None # Start node has no parent node

    g_nodes = graph.getNodes()        
        
    while(len(unvisited) != 0): 
        curr = getMinEdge(weight_dict, unvisited)
        g_nodes[curr].visited = True
        unvisited.remove(curr)
               
        for edge in graph.getEdges(curr):  
            if g_nodes[edge].visited == False:
                if weight_dict[curr] + getWeight(graph, curr, edge) < weight_dict[edge]:
                    weight_dict[edge] = weight_dict[curr] + getWeight(graph, curr, edge)
                    parent[edge] = curr
                    
    # Store edges of MST in adjacency list format
    # Useful format for creating a graph out of it in the future
    adjList_MST = [[parent[key], key] for key in parent if parent[key] != None]
    
    # Choose return format
    # Either graph or adjacency list
    if format=="graph":
        MinSpanTree = Graph()
        MinSpanTree.createGraph(adjList_MST)
        g_nodes = graph.getNodes()
        for node in MinSpanTree.getNodes().values():
            node.location = g_nodes[node.nodeid].location
    else:
        MinSpanTree = adjList_MST
        
    return MinSpanTree               

# Return weight of edge between vertices v1 and v2 from graph                       
def getWeight(graph, v1, v2):
    edges = graph.getEdges(v1)
    return edges[v2]

# Return the node ID of the node with minimum key value (weight) 
# amongst nodes not yet visited (provided in uv_list)
def getMinEdge(w_dict, uv_list):
    w_uv_dict = dict((key, w_dict[key]) for key in uv_list)
    minWeight = min(w_uv_dict, key = w_uv_dict.get) 
    return minWeight

def createRandConnectedGraph(v, e):
    toConnect = []
    randGraph = createRandomGraph(v,e)
    components = connected_components(randGraph)
    for comp in components:
        toConnect.append(comp[0])   
        
    for i in range(len(toConnect)-1):
        randGraph.addEdge(toConnect[i], toConnect[i+1], 1)
        
    return randGraph
