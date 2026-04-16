# CS540-AI

## Probabilities
Outcomes: possible results of an experiment   
Rolling a die: {1, 2, 3, 4, 5, 6}  
Events: subsets of outcomes we're interested in (Always including empty set)  
$$P({1,2,3})=0.5$$  
$$P(\emptyset)=0$$, $$P(\Omega)=1$$.  

## Machine Learning
Feature Vectors  

## Clustering
k-means  
graph based  
spectral & Laplacian, Normalized Laplacian 
t-SNE: map vectored data into 2-dim vectors based on probability of two pts to be neighbors.   
## Kernel Density
"Smooth out a histogram" - useful to predict distribution P from some samples $$x_i$$. 
## Gradient Descent
Optimize parameters to optimize a model (function) to predict certain outcome given input  
Attempts to reduce loss function using iterative process  
### Test Set
Useful for detecting overfitting: when the data does better on the training set than the test set  
MSE: mean squared error (useful for regression: minimize this on the test set!)   

# MIDTERM 2
## Game Theory
Pure Nash EQ: Assuming the opponent's move stays fixed, we cannot change our payoff if we change our current strategy; and the opponent has no reason to switch strategies if we switch. 
Mixed Nash EQ: The indifference principle - make the opponent indifferent towards changing their action; we probabilistically choose actions based on expected payoff. 


## Reinforcement Learning
Find a map between states and actions that maximizes rewards. 
