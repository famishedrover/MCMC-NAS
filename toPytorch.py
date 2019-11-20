from graphGeneration import getFullArch, topsort
from graphPlot import plotUndirected, plotDirected

from neuralnet import unit , runNetwork



# extra imports as backup
import torch
import torch.nn as nn
import torch.nn.functional as F


# To convert the graph to pytorch version : 
# 1. Get topsort of the graph from networkx 
# 2. Assign Layer to the node in the graph according to the node 
# 	e.g. some internal node is a conv layer etc... 
# 	Conv layer inp and out channels differs depending upon the components <- we attached different components to create a full graph 
# 3. Create a ModuleList for this new graph copy and write the forward function for pytorch which is essentially 
# 	traverse the topsort sequentially and any element i requires outputs of parent(i) as input 

# ------------------WRITE NETWORKX -> PYTORCH NODE CONVERSION SPECIFIC TO PROBELEM STATEMENT---------------------------
# Try for ImageNet 
def giveLayerImageNet(G, node):
	pass


# FOR MNIST <- have seperate giveLayers accroding to image input
# The order is by design is such that all 'a' component come first then 'b' so on
def giveLayer(G, node) :

	if node == 'Ou' :
		G.node[node]['layer'] = unit(8,1)


	if node == 'In' :
		G.node[node]['layer'] = unit(1,8)
	if 'a' in node :
		if node in list(G.successors('In')) :
			G.node[node]['layer'] = unit(8,8)    # start of component
		elif node in list(G.predecessors('A')) :
			G.node[node]['layer'] = unit(8,16)   # end of component
		else :
			G.node[node]['layer'] = unit(8,8)   # continuation of component

	if node == 'A' :
		G.node[node]['layer'] = unit(16,16,pool=True)
	if 'b' in node :
		if node in list(G.successors('A')) :
			G.node[node]['layer'] = unit(16,32)    # start of component
		elif node in list(G.predecessors('B')) :
			G.node[node]['layer'] = unit(32,16)   # end of component
		else :
			G.node[node]['layer'] = unit(32,32)   # continuation of component


	if node == 'B' :
		G.node[node]['layer'] = unit(16,8,pool=True)
	if 'ou' in node :
		if node in list(G.successors('B')) :
			G.node[node]['layer'] = unit(8,8)    # start of component
		elif node in list(G.predecessors('Ou')) :
			G.node[node]['layer'] = unit(8,8)   # end of component
		else :
			G.node[node]['layer'] = unit(8,8)   # continuation of component

	if node == 'Ou' :
		G.node[node]['layer'] = unit(8,8)   # final out will be like (batch,8,x,y)

# list(G_dir.successors(n))


def attachLayerDependingUponNode(G, order):
	# dict of (k,v) k=node from networkx, v is actual layer like conv etc..

	# For MNIST 
	# giveLayer = giveLayerMNIST
	
	for node in order : 
		giveLayer(G, node)

	return G



# --------------------------------- SAMPLE RUN-------------------------------------------------------------

# G = getFullArch(3, 300)
# plotDirected(G)

# graphOrder = list(topsort(G))
# # The order is by design is such that all 'a' component come first then 'b' so on

# G = attachLayerDependingUponNode(G,graphOrder)

# print G.nodes.data()


# ---------------------------------DYNAMIC NEURAL NETWORK GEN FROM NETWORKX GRAPH-----------------------------
'''
Main NN module which takes in the attachedLayer networkx Graph and creates the ModuleList Pytorch Network
'''
class Net(nn.Module):
    def __init__(self, G):
        super(Net, self).__init__()
        self.G = G # this is graph with layers attached
        self.graphOrder = list(topsort(G)) #save time in topsorting everytime when required, use this <-DO NOT CHANGE THIS ORDER!!! as nodeInNN is orderdependent
        self.nodesInNN = nn.ModuleList()

        for nod in self.graphOrder :
        	# print nod
        	self.nodesInNN.append(G.node[nod]['layer'])

        self.fc = nn.Linear(8*7*7, 10)  # 3 maxpools cause the final image to be 1,8,7,7


    def forward(self, x):
    	result = {}
    	for ix, node in enumerate(self.graphOrder) :
    		# print node
    		# find pred and get results from pred 
    		# then add those pred 
    		# then supply in the curr node
    		pred = list(self.G.predecessors(node))
    		if len(pred) == 0 : # when node == 'In'
    			result[node] = self.nodesInNN[ix](x)
    		else : 
    			# get results for each pred and add 
    			# tmp = result[pred[0]]
    			# for pNode in pred[1:] : 
    			# 	tmp +=  result[pNode]
    			result[node] = self.nodesInNN[ix](*[result[pNode] for pNode in pred])

    	x = torch.flatten(result['Ou'],1)
    	output = self.fc(x)
    	output = F.log_softmax(output, dim=1)

        return output


def testMNIST(Net,G):
	'''
	To test whether the created Net is fine (dimension wise) or not on MNIST input dimen
	'''
	x = torch.zeros((1,1,28,28))
	model = Net(G)
	print model(x).shape




# ---------------------------------RANDOM HIT/MISS CODE-------------------------------------------------------------

# nx.readwrite.nx_yaml.write_yaml(G,"model.yaml")
# runNetwork(model)





# nnModelDict = attachLayerDependingUponNode(G, graphOrder)
# making graphOrder as list rather than the generator object is the only useful thing I could find to do with topsort

# Working with networkx graphs sample <- assiging data to nodes
# print graphOrder
# print graphOrder[0]
# G.nodes[graphOrder[0]]['layer'] = 1
# print G.nodes[graphOrder[0]]['layer']











