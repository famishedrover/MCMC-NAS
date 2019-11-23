import networkx as nx
from graphPlot import plotUndirected, plotDirected, viz
import random 
from graphGeneration import getFullArch, topsort
from neuralnet import runNetwork
from toPytorch import Net, attachLayerDependingUponNode, testMNIST

from torch.utils.tensorboard import SummaryWriter # for viz 
import os
import torch

from torchsummary import summary 



components = 2			#3,4...
k = 50 				#300,100...
epochs = 10    			#10,20...
batch_size = 256 		
test_batch_size = 256
lr = 0.1
gamma = 0.7
seed=314



# Sample Architecture
G = getFullArch(components = components, k=k)
# G = getFullArch(components = components, k=k, algo='ER', numNodes = 10, p= 0.6)
# G = getFullArch(components = components, k=k, algo='BA', numNodes = 10, m= 3)
# G = getFullArch(components = components, k=k, algo='WS', numNodes = 10, kring=4, p= 0.4)
# Other params 
# algo = 'ER', numNodes , p <- prob. of edge 
# algo = 'BA', numNodes, m <- num of edges to attach from new to existing node
# algo = 'WS', numNodes, kring, p <- kring- k neighbors in ring topology, p is prob of rewiring


plotDirected(G)

exit()

# Begin Bookkeeping
tb = SummaryWriter()
os.mkdir(tb.log_dir+"/model")
os.mkdir(tb.log_dir+"/nxgraph")


# plotDirected(G)
graphOrder = list(topsort(G))
# The order is by design is such that all 'a' component come first then 'b' so on

plt = plotDirected(G, back=True, darkTheme=False)
fig = plt.gcf()
fig.savefig(tb.log_dir+"/nxgraph/"+"graph_light.png", dpi=300)
# plt.show()

del plt, fig 

plt = plotDirected(G, back=True, darkTheme=True)
fig = plt.gcf()
fig.savefig(tb.log_dir+"/nxgraph/"+"graph_dark.png", dpi=300)

# plt.show()



# fig = plotDirected(G,back=True).gcf() 
tb.add_figure('NxGraph',fig)
hdict = {"Components":components,
				"Chain Run (k)":k,
				"Epochs":epochs,
				"train_size":batch_size,
				"test_batch_size" : test_batch_size,
				"lr":lr,
				"gamma":gamma,
				"seed NN":seed,
				"seed graphGeneration":31415}
tb.add_hparams(hdict, {})  # for some reason metric-dict isint working...
# plt.show()




# Atttach NN Layers componentwise
G = attachLayerDependingUponNode(G,graphOrder)
# print G.nodes.data()

testMNIST(Net,G)


model = Net(G)

# summary(model,(1,28,28))

# Bookkeeping stuff
nx.readwrite.nx_yaml.write_yaml(G,tb.log_dir+"/nxgraph/"+"model.yaml")
torch.save(model, tb.log_dir+"/model/firstmnist_.pb")
tb.add_figure('NxGraph',plotDirected(G,back=True).gcf())
dot = viz((1,1,28,28), model)
dot.format = 'png'
dot.render(tb.log_dir+"/model/fullGraph.png")


runNetwork(model, tb=tb, epochs = epochs, batch_size = batch_size, test_batch_size = test_batch_size, lr = lr, gamma = gamma, seed=seed)

tb.close()














