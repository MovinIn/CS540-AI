Bayes Rule
$$P(A|B) = \frac{P(B|A)*P(A)}{P(B)}$$

Remember: matrix multiplication: top left corner makes the box. As in: $$m \times n * n \times k = m \times k$$   
Remember that in matrix multiplication, the first is the transformation; the next is a set of vectors. Thus, we read left to right for the first and up and down for the right. 

### PCA
This is a dimensionality reduction, overfitting reduction technique. It maps the $$m$$ dimension vectors into $$n$$ dimension vectors st. $$m>>n$$. 
How it works  
1. Compute the covariance matrix. Shows how different variables are related to one another.
2. Compute the eigenvectors and eigenvalues of the covariance matrix (this shows the greatest variance of the data; meaning, it shows the relationship between variables).
3. We use the eigenvectors that have the largest eigenvalues to create change in basis matrix, so that only the important relationships are looked at.

### NLP
Goal: determine the next word in the sentence. In other words, determine: 

$$
P(w_i|w_{i-1},w_{i-2},\dots,w_1)
$$

Markov assumptions: we only need to know recent history. In other words: 

$$
P(w_i|w_{i-1},w_{i-2},\dots,w_1) = P(w_i|w_{i-1},w_{i-2},\dots,w_{i-k})
$$

Unigram Model: k=0  
Bigram Model: k=1  
n-gram Model: k=n-1

Perplexity: measure of uncertainty. The higher the number, the more uncertain. 
$$PP(W) = P(w_1,w_2,\dots,w_n)^{-1/n}$$. As probability of the sentence increases, the perplexity decreases.  

Bag of Words  
Term Frequency $$TF_{ij}$$: How many times a term i appears in a document j.   
Inversed Term Frequency $$IDF_{ij}$$: How rare is a term in a set of documents. 
$$TF-IDF_{ij}$$: Determines the importance of a word. If high, word $$i$$ is frequent in document $$j$$, but rare in others. If low, word $$i$$ is either rare in document $$j$$ or common across all documents.   
$$TF-IDF_{ij} = TF_{ij} \times IDF_{ij} = TF{ij}*log(\frac{N}{df_i})$$

Representing words  
Word embeddings: Describes the word with its relation to other words in the document. In an n-length vector v (the word itself), v[i] represents the relationship between word $$i$$ and the word itself.   

Supervised Learning
We are given labels to training datapoints. Goal: determine the label of a test datapoint. 
1. Classification: Think discrete variable. 
2. Regression: Think continuous variable.

Unsupervised Learning
We are NOT given labels to datapoints. Goal: find patterns within the data. 
### Clustering
1. Hierarchical
    1. Agglomerative Clustering: start each point in different clusters, then progressively merge clusters.
    2. Merging: merge closest clusters. First, define distance between two clusters $$d(a, b)$$ as: 
          1. Single-Linkage: By the distance between the closest two points from each cluster.
          2. Complete-Linkage: By the distance between the two farthest points from each cluster.
          3. Average-Linkage: By the average distance of all points from each cluster (the normalized summation of the distance of every unique pairing). 
3. Partitional
    1. k-means: center based algorithm. Given a specified input of needing $$k$$ clusters, partition the dataset into $$k$$ groups.
          1. Randomly assign k cluster centers (aka. centroids).
          2. Assign each datapoint to a specified centroid (minimum distance from centroid).
          3. Move each centroid to the center of its assigned points.
          4. Repeat steps 2 and 3 until convergence.
          5. k-means will find a local optimum.
    2. Graph based clustering: Partition a vertex set $$V$$ into $$V_1$$ and $$V_2$$. Goal: reduce cost of cut. Given a partitioning $$P={A,B}$$, define cost as: 
          1. Weight of cut: reduce $$\sum_{i\in A,j\in B} w_ij$$.
          2. Weight of cut and normalizing with balance: reduce $$Ncut(A_1,\dots,A_k) = \frac{1}{2} \sum_{i=1,j \in B}^{k} \frac{W(A_i,B_j)}{vol(A)}$$ where $$vol(A) = \sum_{i \in A}{} degree(i)$$.
          3. Laplacian L: $$L = D - A$$, where $$D$$ is the degree matrix, and $$A$$ is the adjacency matrix. Choose the smallest eigenvectors (this will make sure points close to clusters will stay close; other points will drift away).
          4. Similarly, T-SNE is a dimensionality reduction, where close points will stay close, and other points can drift however far they want. Different from PCA in that t-SNE is Local, while PCA is global.
    3. Bayesian
### Gradient Descent
Gradient descent reduces the loss function of a model by changing its parameters by a particular learning rate.   
The `features` of a model are the inputs, while the `parameters` of the model are the variables used to compute output.   

### k-nearest neighbors
A classification algorithm that takes the k nearest neighbors of a certain point. The point will be classified to the group with the majority. 
### Naive Bayes
The assumption when calculating probability: 
$$P(X_1,X_2,\dots,X_k|y) = P(X_1|y)*P(X_2|y) * \dots * P(X_k|y)$$. The actual probability should be computed using chain rule. 
