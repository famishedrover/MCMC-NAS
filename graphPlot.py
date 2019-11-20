import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from networkx.drawing.nx_agraph import write_dot, graphviz_layout



def plotUndirected(G):
	'''
	To plot Undirected Graphs (repeated edges in the edge list is fine)
	Input : Undirected Graph G 
	Output : None 
	Interface Output : matplotlib plot 
	'''

	pos = nx.spring_layout(G)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_nodes(G, pos,cmap=plt.get_cmap('jet'), node_size=500)
	nx.draw_networkx_edges(G, pos, edge_color='white')
	plt.axis('off')
	plt.show()


def plotDirected(G):
	'''
	To plot Directed Graphs 
	Input : Directed Graph G 
	Output : None 
	Interface Output : matplotlib plot 
	'''

	# pos = nx.spring_layout(G)
	pos = graphviz_layout(G, prog='dot')

	val_map = {'In':80, 'Ou':80}
	for i in range(ord('A'),ord('Z')):
		val_map[chr(i)] = 100 

	values = [ val_map.get(node, 1) for node in G.nodes()]

	nx.draw_networkx_labels(G, pos, font_color='white')
	nx.draw_networkx_nodes(G, pos, node_color= values, cmap=plt.get_cmap('jet'), node_size=500)
	nx.draw_networkx_edges(G, pos, edgelist= G.edges,edge_color='white', arrows=True)
	plt.axis('off')
	plt.show()



# G = nx.Graph()
# G.add_edges_from(
#     [('A','B'),('B','C'),('C','D'),('D','E'),('E','F')])

# print G.nodes
# plotUndirected(G)	

# import random 

# print G.edges
# print random.sample(G.edges, 2)
# print random.sample(G.edges, 2)
# print random.sample(G.edges, 2)
# print random.sample(G.edges, 2)
# print random.sample(G.edges, 2)
# print random.sample(G.edges, 2)
# print random.sample(G.edges, 2)

# ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     # ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')


def sample():
	'''
	From Stackoverflow : 
	Used as a referece to plot graphs 

	Flaws : 
	This one uses spring layout - which is useless for our case 

	It provides all basic method implementations on a DiGraph like toying with arrows, colors etc.
	'''
	G = nx.DiGraph()
	G.add_edges_from(
	    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
	     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

	val_map = {'A': 1.0,
	           'D': 0.5714285714285714,
	           'H': 0.0}

	values = [val_map.get(node, 0.25) for node in G.nodes()]

	# Specify the edges you want here
	red_edges = [('A', 'C'), ('E', 'C')]
	edge_colours = ['black' if not edge in red_edges else 'red'
	                for edge in G.edges()]
	black_edges = [edge for edge in G.edges() if edge not in red_edges]

	# Need to create a layout when doing
	# separate calls to draw nodes and edges
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
	                       node_color = values, node_size = 500)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
	nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
	plt.show()



