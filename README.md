# MCMC Project Code & Report

"Randomly Wired Networks are on the rise, have we been creating wrong Networks all along?" - Mudit Verma, Arizona State University 

## Code Files
1. main.py : Generates a graph and trains it according to default params
2. manyruns.py : Can generate and run many graphs for different algorithms in a single go. Used this to automate training for Google Colab
3. toPytorch.py : Converts a given networkx DAG to pytorch model. Has mapper function for MNIST.
4. graphGeneratoion.py : Contains functions to generate graph via different algorithms, MC, ER, BA, WS. Also has methods to generate sub graphs and combine them.
5. graphPlot.py : Contains functions to plot/save, directed/undirected graphs
6. neuralnet.py : Contains helper functions to train a given neural network
7. info.py : used to read saved models and obtain graph metrics.
8. util.py : Used for helper functions to combine component graphs.
9. VisualizationPlots.ipynb : Has code for generating relevant plots for graph metrics.
10. googleColab.ipynb : Notebook used to run on Google Colab 

## Code Folders 
1. runs/ : Stores all model info, graph plots, trained model weights, tensorboard events.
2. scalars/ : Stores accuracy information for each of the generation network according to algorithm.


## Plots 
![Testing Accuracy MC](/images/MC.png)
![Testing Accuracy ER/WS/BA](/images/Others.png)
![Graph Density](/images/density.png)
![Graph Density Histograph](/images/varHist.png)
