import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('ques4', sep=' ')
X['cverror'] = 1 - (X['cvaccuracy']/100)
X['terror'] = 1 - (X['testaccuracy']/100)

train = plt.scatter(X['d'], X['cverror'], color='blue', marker='^', label='Train error')
test = plt.scatter(X['d'], X['terror'], color='red', marker='*', label='Test Error')
plt.xlabel('Polynomial Kernel degree')
plt.ylabel('error rate')
plt.title(r'CV error, $\log_{5}{C}$')
plt.legend((train, test), ('Train Error', 'Test Error'), scatterpoints=1, loc='upper left')
plt.savefig('ques4_1.png')
