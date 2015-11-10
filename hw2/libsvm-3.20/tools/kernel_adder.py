import pandas as pd
import math
from sklearn import datasets
import numpy.linalg as LA
import numpy as np

SPLICE_LOCATION = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/tools/splice_hw/"
FILE_NAME = "splice_noise_train.txt.scale"

GAMMA = 0.03125
DEGREE = 3
C = 5

sigma = math.sqrt(1/(2 * GAMMA))
negSigmaSq = sigma * sigma * -1;

X, Y = datasets.load_svmlight_file(SPLICE_LOCATION+FILE_NAME)

SUM = np.empty([X.shape[0], X.shape[0]+1])
GAUSSIAN = np.empty([X.shape[0], X.shape[0]+1])
POLY = np.empty([X.shape[0], X.shape[0]+1])

for i in range(0,X.shape[0]):

    SUM[i,0] = i+1
    GAUSSIAN[i,0] = i+1
    POLY[i,0] = i+1

    for j in range(0,X.shape[0]):
        polynomial = (X[i].dot(X[j].transpose()) ** DEGREE )[0,0]
        gaussian = math.exp(LA.norm( (X[j] - X[i]).data, ord=2)**2/negSigmaSq)
        total = polynomial + gaussian

        print "computed for [{0} ,{1}] = {2} {3} {4}".format(i, j, total, polynomial, gaussian)
        SUM[i, j+1] = total
        POLY[i, j+1] = polynomial
        GAUSSIAN[i, j+1] = gaussian

datasets.dump_svmlight_file(SUM, Y, 'ques6.sum.kernel')
datasets.dump_svmlight_file(GAUSSIAN, Y, 'ques6.gaussian.kernel')
datasets.dump_svmlight_file(POLY, Y, 'ques6.polynomial.kernel')
