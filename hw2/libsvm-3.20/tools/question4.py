import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv('ques4', sep=' ')
X['cverror'] = 1 - (X['cvaccuracy']/100)
X['terror'] = 1 - (X['testaccuracy']/100)

fig = plt.figure()
fig.set_size_inches(8.5,11)
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(223)

train = ax1.scatter(X['d'], X['cverror'], color='black', marker='^')
test = ax1.scatter(X['d'], X['terror'], color='black', marker='*')
ax1.set_xlabel('Polynomial Kernel degree')
ax1.set_ylabel('error rate')
ax1.set_title(r'CV error, $\log_{5}{C} = 1.0$')
ax1.legend((train, test), ('Train Error', 'Test Error'), scatterpoints=1, loc='upper left')

ax2 = fig.add_subplot(222)
nSV = ax2.scatter(X['d'], X['nSV'], color='black', marker='^', label='nSV')
nBSV = ax2.scatter(X['d'], X['nBSV'], color='black', marker='*', label='nBSV')
ax2.set_xlabel('Polynomial Kernel degree')
ax2.set_ylabel('number of support vectors')
ax2.set_title(r'CV error, $\log_{5}{C} = 1.0$')
ax2.legend((nSV, nBSV), ('nSV', 'nBSV'), scatterpoints=1, loc='upper left')

ax2 = fig.add_subplot(222)
nSV = ax2.scatter(X['d'], X['nSV'], color='black', marker='^', label='nSV')
nBSV = ax2.scatter(X['d'], X['nBSV'], color='black', marker='*', label='nBSV')
ax2.set_xlabel('Polynomial Kernel degree')
ax2.set_ylabel('number of support vectors')
ax2.set_title(r'CV error, $\log_{5}{C} = 1.0$')
ax2.legend((nSV, nBSV), ('nSV', 'nBSV'), scatterpoints=1, loc='upper left')

ax3 = fig.add_subplot(2,2,3)
rho = ax3.scatter(X['d'], X['rho'], color='black', marker='^', label='nSV')
ax3.set_xlabel('Polynomial Kernel degree')
ax3.set_ylabel('Soft Margin (rho * -1)')
ax3.set_title(r'CV error, $\log_{5}{C} = 1.0$')

plt.savefig('ques4_1.png')
