import csv
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt


KERNEL_DEGREE = 1
FILE_NAME = "deg" + str(KERNEL_DEGREE)

X = pd.read_csv(FILE_NAME+'.out.filtered', sep=' ')

X['cerror'] = 1 - X['accuracy']/100

K = X.groupby('c')

result = pd.DataFrame(columns=['C', 'Mean', 'Std'])

for name, group in K:
    result = result.append({'C':name, 'Mean':group['cerror'].mean(), 'Std':group['cerror'].std()}, ignore_index=True)

result['PStd'] = result['Mean']+result['Std']
result['NStd'] = result['Mean']-result['Std']
print result


#ax = result.plot(x='C', y=['Mean', 'PStd', 'NStd'])
#ax.get_figure().savefig('./'+ FILE_NAME + '.png')

plt.plot(result['C'], result['Mean'], 'k-', label='Mean')
plt.plot(result['C'], result['PStd'], 'g--', label='Mean + Standard Deviation')
plt.plot(result['C'], result['NStd'], 'r--', label='Mean - Standard Deviation')
plt.title('Polynomial Kernels, degree = '+str(KERNEL_DEGREE))
plt.ylabel('Cross validation error')
plt.xlabel(r'$\log_{5}{C}}$')

plt.savefig('./deg{0}.png'.format(KERNEL_DEGREE))
