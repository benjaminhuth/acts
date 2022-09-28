(exatrkx_hyperparameters)=
# Exatrkx Hyperparameter Reference

## Common dataset options

| Hyperparameter | Description |
|----------------|-------------|
| pt_signal_cut    |     cut true hits by pt |
|  pt_background_cut    |     cut all hits by pt |
|  true_edges    |     modulewise_true_edges or | layerwise_true_edges. Two ways to create true edges, with modulewise_true_edges duplicates on the same module are dropped. |
|  noise    |     if False, cut hits with particle_id == 0. This is done regardless of the noise option if pt_background_cut > 0. |

## Common training options

| Hyperparameter | Description |
|----------------|-------------|
| max_epochs    |     how many epochs to train|
| lr    |     The learning rate of the Adam optimizer|
| patience    |     The patience of the learning-rate scheduler|
| factor    |     The factor of the learnign-rate scheduler|
| warmup | for how many global steps "warm-up" the learning rate (linear increase through the warmup period) |


## Embedding

| Hyperparameter | Description |
|----------------|-------------|
| spatial_channels    |     size of spatial hit-input (usually 3)|
| cell_channels    |     size of the cell-information hit-input|
| emb_hidden    |     size of the hidden layers in the network which encodes the embedding|
| emb_dim    |     size of the embedding (the size of the output layer of the encoding network)|
| nb_layers    |     number of the hidden layers in the encoding network.|
| activation    |     activation function in the encoding network|
| r_train    |     maximum radius to search for neighbors in the embedding space|
| knn    |     maximum number of neih|
| points_per_batch    |     How many points per batch query for for the network input. These are randomly selected from the queried ones (see options in regime)|
| randomization    |     How many random pairs to append to the batch, number = randomization \| len(edge_list)|
| weight    |     The weight of true pairs (false pairs have weight 1)|
| margin    |     Margin of the hinge-loss|
| regime    |     list of keywords influencing the behaviour of the algorithm (see below) |

| Regime options | Description |
|----------------|-------------|
| ci    |     if present, cluster information are added to the spacepoint data |
| norm    |     Normalize the output of the encoding network|
| query_all_points    |     query initial batch from all possible hits (default is only true ones)|
| query_noise_points    |     query initial batch only points with particle_id == 0 (shadowed by query_all_points)|
| hnm    |     Hard negative mining    |     Add additional edges to the training batch directly from the neighbourhood in the embedding space|
| low_purity    |     Only take a small subset of the hnm-pairs (??? dependent on r_train)|
| rp    |     Random points    |     Add additional random edges to the training batch|

## Filtering

| Hyperparameter | Description |
|----------------|-------------|
|in_channels    |     sum of spatial + cell input sizes|
|emb_channels    |     num of channels to input the embedding from the previous step (default to 0)|
|hidden    |     size of the hidden layers in the filtering network|
|nb_layer    |     number of layers in the network|
|layernorm    |     apply layernorm to output of each layer (all features of a sample are simultanously normalized)|
|batchnorm    |     apply batchnorm to output of each layer (each feature in a batch is normalized independently across-sample)|
|warmup    |     for how many global steps "warm-up" the learning rate (linear increase through the warmup period)|
|n_chunks    |     Number n how many chunks the filtering should be divided. Has memory-usage reasons I guess. This is used in a first step without gradient, to make a list of hard-negatives which can be learnt then|
|filter_cut    |     threshold for accepting the edge|
|ratio    |     how many hard-negatives resp. easy-indicies compared to true indices|
|weight    |  TODO  |
| regime    |     list of keywords influencing the behaviour of the algorithm (see below) |


| Regime options | Description |
|----------------|-------------|
|subset    |     Only take the edges which are in layerless_true_edges|
|ci    |     use cell-information|
|weighting    |     get weights from batch and set weights of false edges to 1 -- seems to be broken|
|pid    |     Use particle_id-match as truth for loss instead of batch.y|

## GNN

| Hyperparameter | Description |
|----------------|-------------|
| spatial_channels    |     input dimension for the spatial coordinates, usual 3|
| cell_channels    |     input dimension for the additional cell information|
| hidden    |     Size of the hidden layers in the different MLPs|
| hidden_activation    |     What activation to use for the different MLPs|
| layernorm    |     Wether to apply layernorm for the different MLPs|
| nb_node_layer    |     Number of layers for the node-encoder and the node-network|
| nb_edge_layer    |     Number of layers for the edge-encoder and the edge-network|
| n_graph_iters    |     How often do the message passing|
| aggregation    |     How to aggregate the features after the graph iterations (max, sum, sum_max possible)|
| edge_cut    |     Where to cut the sigmoided outputs to determine the edge prediction|
| weight    |     if given, used as the pos_weights-parameter for the binary crossentropy loss, else a weight is computed somehow (specify how?)|
| directed    |     if given and True, the graph is not trained with a bi-directional graph|
| initialization    |     Something that is passed to Python's eval(...) and is called with the layer weights on reset_parameters() member function|
| regime:|list of keywords influencing the behaviour of the algorithm (see below) |


| Regime options | Description |
|----------------|-------------|
| pid   |   See embedding
| weighting   |   Use weights from the batch for the loss
