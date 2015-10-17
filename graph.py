# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:56:16 2015

@author: Sanjeevni
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:51:58 2015

@author: Sanjeevni
"""

class Node:      
    def __init__(self, nodeid, edgeList=None):
        self._nodeid = nodeid
        if(edgeList==None):
            self._edgeList = {}
        else:
            self._edgeList = edgeList
    
    def addEdge(self, dest, weight=1):
        self.edgeList[dest] = weight
    
    def __str__(self):
          return str(self.nodeid) + ' -> ' + str([x for x in self.edgeList])
  
    @property
    def nodeid(self):
        return self._nodeid
        
    @property 
    def edgeList(self):
        return self._edgeList
        
    @nodeid.setter
    def nodeid(self,value):
        self._nodeid = value
    
    @edgeList.setter
    def edgeList(self,value):
        self._edgeList = value

class Graph:
    def __init__(self):
        self.nodeList = {}
      
    def createGraph(self, adjList):       
        for node in adjList:
            if(len(node) == 2):
                weight = 1
            else:
                weight = node[2]
            self.addEdge(node[0], node[1], weight)
            
    def hasNode(self, nodeId):
        return nodeId in self.nodeList
     
    
    def addEdge(self, sourceId, destId, weight=1):
        if(not self.hasNode(sourceId)):
            self.addNode(sourceId)
        if(not self.hasNode(destId)):
            self.addNode(destId)
        
        self.nodeList[sourceId].addEdge(destId, weight)
        self.nodeList[destId].addEdge(sourceId, weight)
    
    def addNode(self, nodeId):
        newNode = Node(nodeId)
        self.nodeList[nodeId] = newNode
#        self.printNodeList()
        return newNode
    
    def getEdges(self, nodeId):
        if(self.hasNode(nodeId)):
            currNode = self.nodeList[nodeId]
            return currNode.edgeList

    def getNodeList(self):
        o_nodeList = [node for node in self.nodeList.keys()]   
        return o_nodeList
        
    def getNodes(self):
        return self.nodeList
        
    def __iter__(self):
        return iter(self.nodeList.values())
        
    def printNodeList(self):
        print("NodeList: ")
        for node in self.nodeList:   
            print(node)
        print("###########")
        
    def printGraph(self):
        "Edge List for current graph..."
        for node in self.nodeList:
            print(self.nodeList[node])
    
