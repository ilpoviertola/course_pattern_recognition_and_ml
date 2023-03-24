import numpy as np

x_tr = np.loadtxt('test1/data/X_train.dat')
y_tr = np.loadtxt('test1/data/y_train.dat')
x_tst = np.loadtxt('test1/data/X_test.dat')
y_tst = np.loadtxt('test1/data/y_test.dat')
y_pred = np.zeros_like(y_tst)
print('Reading data... Done!')
print('Shape of training data', x_tr.shape)

mean_val = np.mean(y_tr)
mae = np.sum(np.abs(y_tst - mean_val)) / len(y_tst)

print('Computing 1-nn regression')
for ind, x in enumerate(x_tst):
    dist = np.linalg.norm(x_tr - x, axis=1)
    ind_pred = np.argmin(dist)
    y_pred[ind] = y_tr[ind_pred]

mae_nn = np.sum(np.abs(y_tst - y_pred)) / len(y_tst)

print('  Baseline accuracy (MAE):', mae)
print('  1NN regr. accuracy (MAE):', mae_nn)
