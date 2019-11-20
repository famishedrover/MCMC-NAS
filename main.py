import networkx as nx
from graphPlot import plotUndirected, plotDirected
# from networkx.algorithms.swap import double_edge_swap
import random 
from graphGeneration import getFullArch, topsort
from neuralnet import runNetwork
from toPytorch import Net, attachLayerDependingUponNode, testMNIST

from torch.utils.tensorboard import SummaryWriter # for viz 
import os
import torch


# Sample Architecture
G = getFullArch(components =3, k=100)  #300,100
plotDirected(G)
graphOrder = list(topsort(G))
# The order is by design is such that all 'a' component come first then 'b' so on


# Atttach NN Layers componentwise
G = attachLayerDependingUponNode(G,graphOrder)
print G.nodes.data()

testMNIST(Net,G)

model = Net(G)


tb = SummaryWriter()
os.mkdir(tb.log_dir+"/model")
os.mkdir(tb.log_dir+"/nxgraph")
nx.readwrite.nx_yaml.write_yaml(G,tb.log_dir+"/nxgraph/"+"model.yaml")
torch.save(model, tb.log_dir+"/model/firstmnist_.pb")


runNetwork(model, tb=tb)

tb.close()














