#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.linalg import eigh
from node_embedding_attack.utils import *
from node_embedding_attack.embedding import *
from node_embedding_attack.perturbation_attack import *


# ### Load and preprocess the data

# In[2]:


graph = load_dataset('data/cora.npz')
adj_matrix = graph['adj_matrix']
labels = graph['labels']

adj_matrix, labels = standardize(adj_matrix, labels)
n_nodes = adj_matrix.shape[0]


# ### Set hyperparameters

# In[3]:


n_flips = 1000
dim = 32
window_size = 5


# ### Generate candidate edge flips

# In[4]:


candidates = generate_candidates_removal(adj_matrix=adj_matrix)


# ### Compute simple baselines

# In[5]:


b_eig_flips = baseline_eigencentrality_top_flips(adj_matrix, candidates, n_flips)
b_deg_flips = baseline_degree_top_flips(adj_matrix, candidates, n_flips, True)
b_rnd_flips = baseline_random_top_flips(candidates, n_flips, 0)


# ### Compute adversarial flips using eigenvalue perturbation

# In[6]:


our_flips = perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size)


# ### Evaluate classification performance using the skipgram objective

# In[7]:


for flips, name in zip([None, b_rnd_flips, b_deg_flips, None, our_flips],
                             ['cln', 'rnd', 'deg', 'eig', 'our']):
    
    if flips is not None:
        adj_matrix_flipped = flip_candidates(adj_matrix, flips)
    else:
        adj_matrix_flipped = adj_matrix
        
    embedding = deepwalk_skipgram(adj_matrix_flipped, dim, window_size=window_size)
    f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels)
    print('{}, F1: {:.2f} {:.2f}'.format(name, f1_scores_mean[0], f1_scores_mean[1]))


# ### Evaluate classification performance using the SVD objective

# In[8]:


for flips, name in zip([None, b_rnd_flips, b_deg_flips, None, our_flips],
                             ['cln', 'rnd', 'deg', 'eig', 'our']):
    
    if flips is not None:
        adj_matrix_flipped = flip_candidates(adj_matrix, flips)
    else:
        adj_matrix_flipped = adj_matrix
        
    embedding, _, _, _ = deepwalk_svd(adj_matrix_flipped, window_size, dim)
    f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels)
    print('{}, F1: {:.2f} {:.2f}'.format(name, f1_scores_mean[0], f1_scores_mean[1]))

