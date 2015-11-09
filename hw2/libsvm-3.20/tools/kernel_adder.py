import pandas as pd
import math
from sklearn import datasets
import numpy.linalg as LA

SPLICE_LOCATION = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/tools/splice_hw/"
FILE_NAME = "splice_noise_train.txt.scale"

GAMMA = 0.03125
DEGREE = 3
C = 5

sigma = math.sqrt(1/(2 * GAMMA))
negSigmaSq = sigma * sigma * -1;

X, Y = datasets.load_svmlight_file(SPLICE_LOCATION+FILE_NAME)

K = []

for i in range(0,X.shape[0]):
    row_matrix = []
    row_matrix.append(i+1)
    for j in range(0,X.shape[0]):
        polynomial = (X[i].dot(X[j].transpose()) ** DEGREE )[0,0]
        gaussian = math.exp(LA.norm( (X[j] - X[i]).data, ord=2)**2/negSigmaSq)
        total = polynomial + gaussian
        row_matrix.append(total)
        print "computed for [{0} ,{1}] = {2} {3} {4}".format(i, j, total, polynomial, gaussian)
    K.append(row_matrix)

datasets.dump_svmlight_file(K, Y, 'ques6.kernel')
