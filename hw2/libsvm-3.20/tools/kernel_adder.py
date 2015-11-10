from sklearn import datasets
from scipy.spatial.distance import pdist, squareform
import scipy as scip
import numpy as np
import math

SPLICE_LOCATION = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/tools/splice_hw/"
FILE_NAME = "splice_noise_train.txt.scale"

GAMMA = 0.03125
DEGREE = 3
C = 5

sigma = math.sqrt(1/(2 * GAMMA))
negSigmaSq = sigma * sigma * -1;

X, Y = datasets.load_svmlight_file(SPLICE_LOCATION+FILE_NAME)
X = X.toarray()

gamma = 1.0/X.shape[1]
pairwise_dists = squareform(pdist(X, 'euclidean'))

GAUSSIAN = scip.exp(pairwise_dists ** 2 / negSigmaSq)
GAUSSIAN = np.insert(GAUSSIAN, 0, np.arange(1, X.shape[0]+1), axis=1)

POLY = np.dot(X, X.T)
POLY = np.multiply(POLY, gamma)
POLY = np.power(POLY, DEGREE)
POLY = np.insert(POLY, 0, np.arange(1, X.shape[0]+1), axis=1)

print "shape of poly is {0}".format(POLY.shape)

print "shape of gaussian is {0}".format(GAUSSIAN.shape)

print "found gaussian and poly"
SUM = GAUSSIAN + POLY

datasets.dump_svmlight_file(SUM, Y, 'ques6.sum.kernel')
print "finished dumping sum kernel"
datasets.dump_svmlight_file(GAUSSIAN, Y, 'ques6.gaussian.kernel')
print "finished dumping gaussian kernel"
datasets.dump_svmlight_file(POLY, Y, 'ques6.polynomial.kernel')
print "finished dumping polynomial kernel"
