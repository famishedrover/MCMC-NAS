'''
Heart of the code : 
samples a random graph (or may not be random yet to verify) via MCMC strategy (not MH)

MCMC Chain  :
1. Start with some graph G 
2. Operators : 
	
	AddEdge 
	RemoveEdge 
	
	AddNode 
	RemoveNode 
	
	Stay in the same graph
	
	Swap edges 

3. Run the chain for specified times <- yet to analyze the mixing time of this 
'''


import networkx as nx
from graphPlot import plotUndirected, plotDirected

from util import convertToDAG, combineGraphs, convertGraphToPytorchNetwork

failedAttemptsSwap = 0
totalAttemptsSwap = 0

import random 
random.seed(31415)


opcount = {'total':0,'same' : 0, 'swap' :0, 'addE' :0, 'removeE':0, 'addN':0, 'removeN':0}

def addInputOutput(G_dir, inp='I', outp='O') :
	# add one input and one output node to the graph G_dir

	allNodes = G_dir.nodes 

	succ = []
	pred = []
	for n in allNodes:
		_s = list(G_dir.successors(n))
		_p = list(G_dir.predecessors(n))

		# print _s
		# print type(_s) 
		# print list(_s)
		if len(_s) == 0 :
			succ.append(n)
		if len(_p) == 0 :
			pred.append(n)

	# return G_dir

	for ss in succ :
		G_dir.add_edge(ss,outp)
	for pp in pred :
		G_dir.add_edge(inp,pp)

	return G_dir 

	# print succ
	# print pred




def switchable(samp) :
	# not used 
	t1,t2 = samp 
	a,b = t1
	c,d = t2

	if (a!=b and b!=c and c!=d) :
		return True 
	return False

def switch(G,samp):
	pass 

def singleSwitch(G):

	# opcount = {'total':0,'same' : 0, 'swap' :0, 'addE' :0, 'removeE':0, 'addN':0, 'removeN':0} <-bookkeeping
	global opcount
	global failedAttemptsSwap, totalAttemptsSwap
	# input G_k | output G_{k+1}
	r = random.random()

	opcount['total'] +=1 

	if r>=0.9 :
		opcount['same'] += 1
		return G

	elif r>=0.5 :
		totalAttemptsSwap += 1
		# pick 2 edges (a,b) & (c,d) in G
		# if switchable then great 
		# else send back G

		# samp = random.sample(G.edges, 2)
		# if (switchable(samp)) :
		# 	switch(G, samp)

		# No need to code, its there oin networkx
		try : 
			
			G = nx.algorithms.swap.double_edge_swap(G,nswap=1,max_tries=2)
		except : 

			failedAttemptsSwap+=1 

		opcount['swap'] += 1


	elif r>=0.3 :
		# add edge / pick random nodes - if no edge then add - if edge : remove

		u,v = random.sample(G.nodes,2)

		# prevent swapping original edges <- not required
		# if ((u==1 and v==2) or (u==2 and v==3) or (u==3 and v==4) or (u==4 and v==5) or (u==5 and v==6)) :
		# 	return G 

		# plotUndirected(G)
		# print (u,v)
		# print G.has_edge(u,v)

		if (G.has_edge(u,v)) :
			# delete
			opcount['removeE'] += 1
			G.remove_edge(u,v)
		else:
			# add
			opcount['addE'] += 1
			G.add_edge(u,v)

	else :
		# add/delete a node 
		r2 = random.random()
		if r2>0.5 :
			# remove 

			if (len(G.nodes)<5) :
				print 'NODES LESS THAN 5'
				return G 
			u = random.sample(G.nodes,1)

			# prevent removing original nodes
			# if u[0] in [1,2,3,4,5,6] :
			# 	return G

			G.remove_node(u[0])
			opcount['removeN'] += 1
		else :
			# nodes are numbered as 1,2,3... 
			if (len(G.nodes)>200) :
				print 'NODES GREATER THAN 200'
				return G

			# print type(G.nodes)
			newnode = sorted(list(G.nodes))[-1]+1
			# print newnode
			# print 'Before',G.nodes
			G.add_node(newnode)
			# print 'After', G.nodes
			opcount['addN'] += 1

	return G


def uniformGraphGenerator(G, K):
	# -------------------- MARKOV CHAIN ------------------------
	# input : G_0 , Output : G_k as one sample , run chain K times
	graphs = []

	for t in range(K):
		G = singleSwitch(G)
		graphs.append(G)

	return graphs 



def relabelling(G,label='') :
	islabel = False 
	if label != '' :
		islabel= True
	print 'TOPSORT'
	ts = list(nx.algorithms.dag.topological_sort(G))
	ix=0
	mapping = {'I':'I', 'O':'O', 'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E', 'F':'F'}


	for i in ts[1:-1] :
		ix+=1 
		nmap = ix

		if islabel is True : 
			nmap = str(ix)+label
		mapping[i] = nmap

	G = nx.relabel_nodes(G, mapping)

	return G


def topsort(G):
	return nx.algorithms.dag.topological_sort(G)


	# G = nx.relabel.convert_node_labels_to_integers(G_dir)

def getSemiNetworkArch(k,label,inp,outp):	
	# Initialize the graph (Subgraph)
	# label looks like 1a, 2a, 3a for a is the component 
	# inp and outp are input and output labels for this subgraph

	G = nx.Graph()
	G.add_edges_from(
	    [(1,2),(2,3),(3,4),(4,5)])

	# Run Chain
	graphs = uniformGraphGenerator(G,k)
	# Plot final output & some stats
	# plotUndirected(graphs[-1])	
	print 'Failed failedAttempts', failedAttemptsSwap
	print 'Total Attempts Swap ', totalAttemptsSwap

	# Get DAG & plot
	print 'Converting to DAG'
	# G_dir = G.to_directed()
	G_dir = convertToDAG(G)
	# plotDirected(G_dir)

	

	

	# Add Input/Output Nodes 
	G_dir = addInputOutput(G_dir, inp, outp)
	
	G_dir = relabelling(G_dir, label)

	# plotDirected(G_dir)

	return G_dir




def getFullArch(components=3, k=100):
	# creates subgraphs as components C1, C2.. and connects them with labels
	# I is the global graphs's input O is the output 
	# A,B,C.. are the intermediate connecting points

	assert components < 27 , "Support for Max 26 components, else alphabets gets wierd"

	start = 'A'


	C = []
	comp = getSemiNetworkArch(k,start.lower(),'In',start)
	C.append(comp)
	# plotDirected(comp)


	for ic in range(components-2) :
		nextStart = chr(ord(start)+1)
		comp = getSemiNetworkArch(k,nextStart.lower(),start,nextStart )
		# plotDirected(comp)
		C.append(comp)
		start = nextStart

	C.append(getSemiNetworkArch(k,'ou',start,'Ou'))

	# C1 = getSemiNetworkArch(k,'a','I','A')
	# C2 = getSemiNetworkArch(k,'b','A','B')

	# # plotDirected(C1)
	# # plotDirected(C2)
	# C3 = getSemiNetworkArch(k,'c','B','C')
	# C4 = getSemiNetworkArch(k,'d','C','D')
	# C5 = getSemiNetworkArch(k,'e','D','O')
	# # G = combineGraphs([C1,C2])
	# G = combineGraphs([C1,C2,C3,C4,C5])

	G = combineGraphs(C)
	# plotDirected(G)
	return G

# getFullArch(4,100)
