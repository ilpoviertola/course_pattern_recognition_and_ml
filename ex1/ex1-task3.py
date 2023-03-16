import matplotlib.pyplot as plt
import numpy as np

def logsig(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(13)
x_h = np.random.normal(1.1, 0.3, 5)
y_h = np.zeros(x_h.shape)
x_e = np.random.normal(1.9, 0.4, 5)
y_e = np.ones(x_e.shape)
x_tr = np.concatenate([x_h, x_e])
y_tr = np.concatenate([y_h, y_e])

# plt.figure()
# plt.plot(x_h, np.zeros([5,1]), 'co', label='hobbit')
# plt.plot(x_e, np.zeros([5,1]), 'mo', label='elf')
# plt.show()

w0_t, w1_t = .0, .0
epochs = 1000
mu = 0.4

for e in range(epochs):
    # Hebbian learning:
#    for x_ind, x in enumerate(x_tr):
#        y_hat = logsig(w1_t * x + w0_t)
#        w1_t = w1_t + mu * (y_tr[x_ind] - y_hat) * x
#        w0_t = w0_t + mu * (y_tr[x_ind] - y_hat) * 1

    # Gradient descent
    y_hat = logsig(w1_t * x_tr + w0_t)
    w1_t = w1_t - (2 * mu) / len(x_tr) * np.sum((y_hat - y_tr) * logsig(w1_t * x_tr + w0_t) * (1 - logsig(w1_t * x_tr + w0_t)) * x_tr) 
    w0_t = w0_t - (2 * mu) / len(x_tr) * np.sum((y_hat - y_tr) * logsig(w1_t * x_tr + w0_t) * (1 - logsig(w1_t * x_tr + w0_t)))

    if np.mod(e, 100) == 0 or e == epochs - 1:
        y_hat = logsig(w1_t * x_tr + w0_t)
        mse = np.sum((y_tr - y_hat) ** 2) / len(y_tr)
        plt.title(f'epoch={e}, w0={w0_t:.2f}, w1={w1_t:.2f}, MSE={mse:.2f}')
        plt.plot(x_h, y_h, 'co', label='hobbit')
        plt.plot(x_e, y_e, 'mo', label='elf')
        plt.plot(np.linspace(.0, 5., 50), logsig(w1_t * np.linspace(.0, 5., 50) + w0_t))
        plt.legend()
        plt.show()