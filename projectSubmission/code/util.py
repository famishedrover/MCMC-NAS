
# util.py for some helper graph functions (can be merged with graphGeneration.py)

import networkx as nx

def convertToDAG(G):
	'''
	Initializes 
	'''
	dG = nx.DiGraph()
	dG.add_edges_from(G.edges)
	return dG

def combineGraphs(lG):
	'''
	Input : [G_1,G_2...] list of graphs to be combined
	Output : G, combined Graph G on common vertices 
	Requirement : 
	must have one input and one output as A->B->C...
	All these graphs MUST be DAGs (not checking anywhere)



	Example : 
	G1 : A->1a, A->2a, 1a->B, 2a->B
	G2 : B->1b....            	->C


	'''


	dG = nx.DiGraph()

	for iG in lG :
		top = list(nx.algorithms.dag.topological_sort(lG[0]))
		dG.add_edges_from(iG.edges)

	return dG

