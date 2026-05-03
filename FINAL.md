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
### Neural Networks
Activation Functions
1. ReLu: max(0, x)
2. Sigmoid: A continuous fcn that sends $$x \in R \rightarrow \[0,1\]$$.
3. tanh: A continuous fcn that sends $$x \in R \rightarrow \[-1,1\]$$.
4. Softmax: useful with multiple outputs. Turns outputs f into probabilities: $$softmax(f) = \frac{e^{f_y(x)}}{\sum_{k=1}{K} e^{f_k(x)}}$$

Gradient Descent in training models    
Let $$\eta > 0$$ be the learning rate. We update parameters to get to a local optima using:

$$
w_t = w_{t-1} - \eta \frac{dL}{dw_{t-1}} = w_{t-1}-\eta \frac{1}{|D|} \sum_{x,y \in D}{} \frac{dL(x,y)}{dw_{t-1}}
$$

Types of Gradient Descent
1. Gradient Descent: Uses entire dataset in the loss fcn. 
2. Stochastic Gradient Descent: Uses a singular training point in the loss fcn. 
3. Minibatch-stochastic gradient descent: Uses $$B \subset D$$ in the loss fcn.

Problems with Gradient Descent
1. Gradient Vanishing: The gradient of the loss fcn: $$\frac{dL}{dW^t} \rightarrow 0$$
2. Gradient Explosion: The gradient of the loss fcn: $$\frac{dL}{dW^t} \rightarrow \inf$$
To stabalize training, it suffices to prove that the gradients are in a reasonable range (ex. in $$[1e-6,1e3]$$). We can do this in the following ways:
1. Use ReLu fcn
2. Add logorithms vs multiplying numbers
3. Normalization

Squared Norm Regularization    
We constrain the weights by modifying the loss function: 

$$
minL(w, b) + \frac{\delta}{2} {||w||}^2
$$

This makes it so the loss is lower bounded by the magnitude of the weights, which would solve exploding gradients and overfitting (the model adding large weights to map one-to-one for the training data). 

Dropout    
We can do dropout by removing one of the nodes of the hidden layer and replacing its value with $$x \in R$$ at a probability of p. 

Convolution
Requires a $$m \times n$$ matrix $$M$$ and a kernel $$K$$. Slides the kernel left to right and up to down, using dot product to produce a transformed matrix.     
Padding: Adds rows and columns around $$M$$ (sort of like a thick border).     
Stride: Number of rows and columns per slide. The dimensions $$m'$$ and $$n'$$ of the transformed matrix given are as follows: 

$$
m' = \lfloor \frac{2*(padding)+m-(Kernel Height)}{stride} + 1 \rfloor \qquad
n' = \lfloor \frac{2*(padding)+n-(Kernel Width)}{stride} + 1\rfloor
$$

In convolution, the number of input channels does not affect the number of output channels; rather, the number of kernels do. For each kernel we sum up its output matrices into one output channel. However, this is different from pooling: there is no summing of output matrices; rather, the input channels determine the output channels.

Pooling
Very similar to convolution, but instead of doing dot product in the window, we take the max or average instead, depending on if we use Max Pooling or Average Pooling.     

Convolutional Neural Networks (CNNs)    

Residual Blocks    
The skip block. Allows the layer to do "nothing". Tries to mimic the identity function $$f(x)=x$$ (or in other words, not to change the input at all).     
A residual block is $$Output = f(x)+x$$. For a neural net to skip the layer, all it must do is set the weights of $$f(x)$$ to nearly zero.     

Recurrent Neural Networks (RNNs)    
Includes cycles in the computational graph allowing information to persist (or in other words, memory).     

High Level Process Breakdown
1. At time step $$t$$, receive input token $$x_t$$.
2. Receive previous hidden state $$z_{t-1}$$
3. Use $$x_t$$ and $$z_{t-1}$$ to compute state $$z_t$$.
4. Use state $$z_t$$ to predict output $$y_t$$.
5. Repeat for all time steps.

In other words: 

$$
\hat{y_t} = g_y(Uz_t) \qquad z_t = g_z(Vz_{t-1}+Wx+b)
$$

where $$g_y$$ is the activation function, $$x$$ is the input matrix, $$U, W, V$$ are the weight matrices, and $$b$$ is the bias. 

Long Short-term Memory (LSTM)    
Goal: remember important information and forget unimportant information over long input sequences.     
Gates
All 3 gates represent negotiations between the cell state and the previous state. The cell state $$c$$ is long term memory, while the hidden state $$h$$ is short term memory. 
1. Forget Gate $$f$$: what to discard from $$c$$ given new input and $$h$$. 
2. Input Gate $$i$$: What to remember given $$h$$. 
3. Output Gate $$o$$: what to reveal given $$c$$.    

Variant of LSTM: (GRU)    
1. Coalesce cell state into hidden state: $$h$$ acts as both long-term memory and current output.
2. Reset Gate $$(r_t)$$: how much past context to use for the candidate.
3. Update Gate $$(z_t)$$: blend ratio between old state and candidate.
4. Candidate Hidden State $$h_t$$: proposed hidden state using past+current input

Attention   
Recall: Word Embeddings. A word is represented as a 1d vector with values in the range [0,1] describing its features.      
It is difficult to understand the meaning of a word with fixed embeddings (aka. values within [0,1] describing features of the word) because immediate context and relationships between words matter. Using fixed embeddings, the contextual embedding of a word is calculated by the vector sum of $$t$$ previous word embeddings.     
Assigning weights does a decent job. (aka. weighting words based on their similarity). The contextual embedding of a word is calculated by the weighted vector sum of $$t$$ previous word embeddings.     
Last Attempt: Attention    
1. Query $$q_i$$: the word attended from. We are trying to figure out its meaning.
2. Key $$k_j$$: the word attended to. We are trying to figure out its relation to the query.
3. Value $$v_j$$: the context (result of the relation).
To find the meaning of word i, we take immediate context $$t$$ previous keys and figure out its value. Then we input into a softmax fcn to probabilistically determine the relations between keys and $$q_i$$.
In other words:

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d}})V
$$

where queries, keys, and values are represented as matrices $$Q, K, V$$. 

Transformers    
Uses attention and fixed word embeddings to discover the true meaning of a word. Fixed embeddings help with not losing the original actual meaning, while attention allows for the word to get its contextual embedding. Feedforward allows for all words to get a new refined contextual embedding given the newfound meaning obtained through attention. Residual connections allow for a pipe connecting fixed embeddings to output, which combines the results from contextual embeddings and fixed embeddings to get the final output meaning.     
Although the exact order and position of words aren't explicitly built-in, we can encode its position through positional encoding.     
Encoders map an input sequence into a sequence of continous representations $$z$$.     
Decoders transform $$z$$ into an output sequence of symbols one element at a time. For each step, it attends to the encoder in order to produce a symbol.     

Data    
We can improve the quality of the training set to not overfit and regularize through augmentation of the data (adding to the training set by slightly changing existing data such as cropping, rotation, and color).     

Graph Neural Networks    
Very similar to attention mechanism. Tries to figure out the meaning of a node through its connections (edges).     
