import os, sys, traceback, getpass, time, re
from threading import Thread
from subprocess import *
import numpy as np


LIBSVM_PATH = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/"
TOOLS_PATH = "/home/anirudhan/workspace/foudnations-of-machine-learning/hw2/libsvm-3.20/tools/"

DATASET_PATH = TOOLS_PATH + "partitions/"
MODEL_PATH = DATASET_PATH + "models/"

C_START = -8
C_STEP = 0.5
C_END = 6 + C_STEP #By default np.arange doesn't include the c_stop

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

        model_file = "c_{0}_d_{1}.{2}.model".format(c if c >=0 else "_"+str(abs(c)), KERNEL_DEGREE, training_set)

        cmd = '{0}/svm-train -t 1 -c {1} -d {2} {3} {4}'.format(LIBSVM_PATH, 5**c, KERNEL_DEGREE, DATASET_PATH+training_set, MODEL_PATH+model_file)

        result = Popen(cmd,shell=True,stdout=PIPE,stderr=PIPE,stdin=PIPE).stdout
        for line in result.readlines():
            print(line),
