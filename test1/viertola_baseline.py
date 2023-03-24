import numpy as np

print('Reading data...')
x_tr = np.loadtxt('test1/data/X_train.dat')
y_tr = np.loadtxt('test1/data/y_train.dat')
x_tst = np.loadtxt('test1/data/X_test.dat')
y_tst = np.loadtxt('test1/data/y_test.dat')
print('done!')

print('Computing baseline regression.')
mean_val = np.mean(y_tr)
mae = np.sum(np.abs(y_tst - mean_val)) / len(y_tst)
print('  Baseline accuracy (MAE):', mae)

