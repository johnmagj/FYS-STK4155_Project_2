import numpy as np

def mini_batch(X, y, batch_size):
    num_points, num_features = np.shape(X)
    batch_indices = np.random.randint(low=num_points, size=batch_size)
    X_batch = np.zeros((batch_size, num_features))
    y_batch = np.zeros((batch_size,))
    for i, batch_i in enumerate(batch_indices):
        X_batch[i] = X[batch_i]
        y_batch[i] = y[batch_i,0]
    return X_batch, y_batch