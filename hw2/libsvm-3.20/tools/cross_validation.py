import numpy as np
import os, sys, traceback, getpass, time, re
from threading import Thread
from subprocess import *

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

for training_set, holdout_set in sorted(cross_validation_pairs.items()):
    training_set = "train" + str(training_set)

    print training_set, holdout_set
