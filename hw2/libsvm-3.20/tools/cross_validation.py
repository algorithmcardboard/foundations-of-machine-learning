import os, sys, traceback, getpass, time, re
from threading import Thread
from subprocess import *
import numpy as np


LIBSVM_PATH = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/"
TOOLS_PATH = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/tools/"

DATASET_PATH = TOOLS_PATH + "partitions/"
MODEL_PATH = DATASET_PATH + "models/"

C_START = -5
C_STEP = 0.5
C_END = 5 + C_STEP #By default np.arange doesn't include the c_stop

KERNEL_DEGREE = 1

cross_validation_pairs = {
        1:'file10',
        2:'file1',
        3:'file2',
        4:'file3',
        5:'file4',
        6:'file5',
        7:'file6',
        8:'file7',
        9:'file8',
        10:'file9'
        }

for c in np.arange(C_START, C_END, C_STEP):
    for training_set, holdout_set in sorted(cross_validation_pairs.items()):
        training_set = "train" + str(training_set)

        model_file = "c_{0}_d_{1}.{2}.model".format(c, KERNEL_DEGREE, training_set)

        cmd = '{0}/svm-train -t 1 -c {1} -d {2} {3} {4}'.format(LIBSVM_PATH, 5**c, KERNEL_DEGREE, DATASET_PATH+training_set, MODEL_PATH+model_file)

        #print "Training for " + training_set + ". Holdout set is " + holdout_set + " " + str(c)
        print cmd
