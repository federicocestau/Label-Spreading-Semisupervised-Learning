# Label-Spreading-Semisupervised-Learning
Label Spreading allows for each labeled point to accept information from its neighbors which may lead to a deviation from its original label.

Label spreading, which offers a slight better stability when the dataset is very noisy or dense. In these cases, standard label propagation might suffer a loss of precision due to the closeness of points with different labels. Conversely, label spreading is more robust because the Laplacian is normalized and abrupt transitions are more heavily penalized using weights. 
Label propagation computes a similarity matrix between samples and uses a KNN-based approach to propagate samples, while label spreading takes a similar approach but adds a regularization to be more robust to noise. This is done by hyperparameter gamma.

High gamma extends the influence of each individual point wide, hence creating a smooth transition in label probabilities. Meanwhile, low gamma leads to only the closest neighbors having influence over the label probabilities.

We have another parameter that is the main difference with label propagation, called soft clamping concept, control by parameter α (alpha) the clamping factor. A value in (0, 1) that specifies the relative amount that an instance should adopt the information from its neighbors as opposed to its initial label. alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.

The key idea of the two methods is essentially the same. The difference lies in the design of the transition matrix. Label propagation uses the graph Laplacian while Label spreading uses the normalized graph Laplacian.

However, to get the best results, it is often beneficial to combine these two sets of data. Such a situation is an excellent example of where we would want to use a Semi-Supervised Learning approach, with the Label Spreading algorithm being one of our options.

Label Spreading uses Soft clamping concept: Each point receives the information from its neighbors (first term) and also retains its initial information (second term). The parameter α (alpha) enables soft clamping by controlling the proportion of information received from neighbors vs. the initial label. Alpha close to 0 keeps all the initial label information (equivalent to hard clamping), while alpha close to 1 allows most of the initial label information to be replaced.

Parameters: 

https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html

kernel{‘knn’, ‘rbf’} or callable, default=’rbf’
gamma float, default=20
Parameter for rbf kernel.
n_neighbors int, default=7
Parameter for knn kernel which is a strictly positive integer.
alpha float, default=0.2
Clamping factor. A value in (0, 1) that specifies the relative amount that an instance should adopt the information from its neighbors as opposed to its initial label. alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.
max_iterint, default=30. Maximum number of iterations allowed.
tol float, default=1e-3
Convergence tolerance: threshold to consider the system at steady state.
n_jobs int, default=None
The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

