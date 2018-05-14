# Recommender System
import numpy as np
import pandas as pd

# Loading Dataset.(MovieLen dataset, it contains 100k movie ratings from 943 users and a selection of 1682 movies.)
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# Get number of users and itemss(movies)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

# Split train, test dataset
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.20)


###  Memory-Based Collaborative Filtering
###  Item-Item
###  User-Item

# Create user-item matrices
# Maping all user, item and rating
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
    
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    
# Calculate the cosine similarity
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine') # for User-Item
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine') # for Item-Item
    
# Make predictions
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings = mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# Evaluation
# RMSE

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# Precision & Recall

precision = 0
recall = 0
rcount = 0
topn = [5, 10, 15, 20]
for n in topn:
    for x in range(len(user_prediction)):
        TP = [i for i in test_data_matrix[x][user_prediction[x].argsort()[-n:][::-1].tolist()].tolist() if 0 < i]
        TPFN = [i for i in test_data_matrix[x] if i > 0]
        precision += (len(TP)/n)
        if(len(TPFN) > 0):
            recall += (len(TP)/len(TPFN))
            rcount += 1
    print("Top ",n," precision = ",precision/len(user_prediction))
    print("Top ",n," recall   = ",recall/rcount)
    
precision = 0
recall = 0
rcount = 0
topn = [5, 10, 15, 20]
for n in topn:
    for x in range(len(item_prediction)):
        TP = [i for i in test_data_matrix[x][item_prediction[x].argsort()[-n:][::-1].tolist()].tolist() if 0 < i]
        TPFN = [i for i in test_data_matrix[x] if i > 0]
        precision += (len(TP)/n)
        if(len(TPFN) > 0):
            recall += (len(TP)/len(TPFN))
            rcount += 1
    print("Top ",n," precision = ",precision/len(item_prediction))
    print("Top ",n," recall = ",recall/rcount)

    
###  Model-Based Collaborative Filtering
###  Matrix Factorization 
###  Singular value decomposition(SVD)

import scipy.sparse as sp
from scipy.sparse.linalg import svds

# get SVD components from train matrix. choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
