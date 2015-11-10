from sklearn import datasets
from scipy.spatial.distance import pdist, squareform
import scipy as scip
import numpy as np
import math

SPLICE_LOCATION = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/tools/splice_hw/"
DEGREE = 3

sigma = math.sqrt(1/(2 * 0.03125))
negSigmaSq = sigma * sigma * -1;

X, Y = datasets.load_svmlight_file(SPLICE_LOCATION+ "splice_noise_train.txt.scale")
X = X.toarray()

gamma = 1.0/X.shape[1]
pairwise_dists = squareform(pdist(X, 'euclidean'))

k_g = scip.exp(pairwise_dists ** 2 / negSigmaSq)

k_p = np.dot(X, X.T)
k_p = np.multiply(k_p, gamma)
k_p = np.power(k_p, DEGREE)

k_sum = k_g+ k_p

Xt, Yt = datasets.load_svmlight_file(SPLICE_LOCATION+"splice_noise_test.txt.scale")
Xt = Xt.toarray()

t_gamma = 1.0/Xt.shape[1]
t_pairwise = squareform(pdist(Xt, 'euclidean'))

k_gt = scip.exp(t_pairwise** 2 / negSigmaSq)

k_pt = np.dot(Xt, Xt.T)
k_pt = np.multiply(k_pt, t_gamma)
k_pt = np.power(k_pt, DEGREE)

k_sum_t = k_gt + k_pt
