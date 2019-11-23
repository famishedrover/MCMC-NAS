import os
import networkx as nx

from graphPlot import plotUndirected , plotDirected 

def getAllModels(rootfolder='./runs'):
	modelPaths = {}

	for root, dirs, files in os.walk(rootfolder):
		# get all the models as networkx graph
		if 'nxgraph' in dirs :

			algo = ''

			if 'ER' in root :
				algo = 'ER'
			elif 'BA' in root : 
				algo = 'BA'
			elif 'WS' in root :
				algo = 'WS'
			else :
				pass # for MC chain
				# raise 'Model Algorithm could not be determined.'


			modelName  = root.split('/')[-1]

			# print 'Root',root
			# print 'dirs',dirs
			# print 'Algo',algo
			# print 'modelName',modelName

			modelPath = '/'.join([root,'nxgraph','model.yaml'])


			# print 'modelPath',modelPath
			modelPaths[modelName] = modelPath
	return modelPaths



def readAllModels(folder):
	print folder
	allMpaths = getAllModels(folder)

	actualModels = {}
	for name,pa in allMpaths.iteritems() :
		print name,
		try : 
			gTmp = nx.read_yaml(pa)
			actualModels[name] = gTmp
		except : 
			print ' FAILED',
		print ''

	return actualModels


def getDensity(actualModels):
	for name, G in actualModels.iteritems():
		print name, nx.density(G)
	



# allModels = readAllModels()














