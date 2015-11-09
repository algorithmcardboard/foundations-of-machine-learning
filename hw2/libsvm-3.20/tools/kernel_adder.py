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

SUM = []
GAUSSIAN = []
POLY = []

for i in range(0,X.shape[0]):
    sum_matrix = []
    poly_matrix = []
    gaussian_matrix = []

    sum_matrix.append(i+1)
    for j in range(0,X.shape[0]):
        polynomial = (X[i].dot(X[j].transpose()) ** DEGREE )[0,0]
        gaussian = math.exp(LA.norm( (X[j] - X[i]).data, ord=2)**2/negSigmaSq)
        total = polynomial + gaussian

        sum_matrix.append(total)
        poly_matrix.append(polynomial)
        gaussian_matrix.append(gaussian)

        print "computed for [{0} ,{1}] = {2} {3} {4}".format(i, j, total, polynomial, gaussian)
    SUM.append(sum_matrix)
    POLY.append(poly_matrix)
    GAUSSIAN.append(gaussian_matrix)

datasets.dump_svmlight_file(SUM, Y, 'ques6.sum.kernel')
datasets.dump_svmlight_file(GAUSSIAN, Y, 'ques6.gaussian.kernel')
datasets.dump_svmlight_file(POLY, Y, 'ques6.polynomial.kernel')
