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



def runmanytimes(components ,		
				k ,
				epochs		,	
				batch_size 	,	
				test_batch_size ,
				lr ,
				gamma ,
				seed,
				kwargs={}):

	print components, k, epochs, batch_size, test_batch_size, lr, gamma, seed
	print kwargs

	# Sample Architecture
	# G = getFullArch(components = components, k=k, algo='ER', numNodes = 10, p= 0.6)
	# G = getFullArch(components = components, k=k, algo='BA', numNodes = 10, m= 3)
	G = getFullArch(components = components, k=k, kwargs=kwargs)
	# Other params 
	# algo = 'ER', numNodes , p <- prob. of edge 
	# algo = 'BA', numNodes, m <- num of edges to attach from new to existing node
	# algo = 'WS', numNodes, kring, p <- kring- k neighbors in ring topology, p is prob of rewiring



	# Begin Bookkeeping
	try : 
		tb = SummaryWriter(comment="_"+kwargs['algo'])
	except : 
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
	hdict.update(kwargs) #add other kwargs as hyperparam

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



# components = 3 			#3,4...
# k = 100   				#300,100...
# epochs = 10    			#10,20...
# batch_size = 256 		
# test_batch_size = 256
# lr = 0.1
# gamma = 0.7
# seed=314


runs = [ 

		(3,100,30,512,512,0.1,0.7,314),
		(3,500,30,512,512,0.1,0.7,314),
		(3,1000,30,512,512,0.1,0.7,314),

		(3,100,30,512,512,0.1,0.7,3142),
		(3,500,30,512,512,0.1,0.7,3142),
		(3,1000,30,512,512,0.1,0.7,3142),


		(3,100,30,512,512,0.1,0.7,31422),
		(3,500,30,512,512,0.1,0.7,31422),
		(3,1000,30,512,512,0.1,0.7,31422),

		(3,100,30,512,512,0.1,0.7,31423),
		(3,500,30,512,512,0.1,0.7,31423),
		(3,1000,30,512,512,0.1,0.7,31423), 

		(3,100,30,512,512,0.1,0.7,31424),
		(3,500,30,512,512,0.1,0.7,31424),
		(3,1000,30,512,512,0.1,0.7,31424),

		(3,2000,30,512,512,0.1,0.7,31422),
		(3,2500,30,512,512,0.1,0.7,31422),
		(3,3000,30,512,512,0.1,0.7,31422)

		]



algoRun = [
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':20,'p':0.2}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':20,'p':0.4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':20,'p':0.6}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':20,'p':0.8}),

		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':10,'p':0.2}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':10,'p':0.4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':10,'p':0.6}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'ER','numNodes':10,'p':0.8}),

		((3,1,30,512,512,0.1,0.7,3142),{'algo':'ER','numNodes':5,'p':0.2}),
		((3,1,30,512,512,0.1,0.7,3142),{'algo':'ER','numNodes':5,'p':0.4}),
		((3,1,30,512,512,0.1,0.7,3142),{'algo':'ER','numNodes':5,'p':0.6}),
		((3,1,30,512,512,0.1,0.7,3142),{'algo':'ER', 'numNodes':5,'p':0.8}),





		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':20,'m':4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':20,'m':5}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':20,'m':7}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':20,'m':7}),

		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':10,'m':4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':10,'m':5}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':10,'m':6}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'BA', 'numNodes':10,'m':7}),

		((3,1,30,512,512,0.1,0.7,3142),{'algo':'BA', 'numNodes':5,'m':2}),
		((3,1,30,512,512,0.1,0.7,3142),{'algo':'BA', 'numNodes':5,'m':3}),
		((3,1,30,512,512,0.1,0.7,3142),{'algo':'BA', 'numNodes':5,'m':4}),
		((3,1,30,512,512,0.1,0.7,3142),{'algo':'BA', 'numNodes':10,'m':5}),





		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':20,'kring':4, 'p':0.2}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':20,'kring':5, 'p':0.4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':20,'kring':7, 'p':0.6}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':20,'kring':7, 'p':0.8}),

		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':10,'kring':4, 'p':0.2}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':10,'kring':5, 'p':0.4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':10,'kring':7, 'p':0.6}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':10,'kring':7, 'p':0.8}),

		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':5,'kring':4, 'p':0.2}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':5,'kring':5, 'p':0.4}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':5,'kring':7, 'p':0.6}),
		((3,1,30,512,512,0.1,0.7,314),{'algo':'WS', 'numNodes':5,'kring':7, 'p':0.8}),

		]






# for MCChain
# <-already done

# for curRun in runs : 
# 	print '-'*20, 'RUN :', curRun
# 	runmanytimes(*curRun)


import sys 

restart = 0 

try : 
	restart = sys.argv[1]
except : 
	pass

restart = int(restart)

for ix, curRun in enumerate(algoRun) :
	if ix<restart : 
		print ix
		continue 
	print 'Num :',ix,'-'*20, 'RUN :', curRun
	runmanytimes(*curRun[0], kwargs = curRun[1])








