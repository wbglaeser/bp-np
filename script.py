import numpy as np

# set network dimensions
INPUT_DIMS = 2
OUTPUT_DIMS_0 = 3
OUTPUT_DIMS_1 = 4
OUTPUT_DIMS_final = 3

# implement relu activation functdef relu(x) -> float:
def relu(x):
    return max(0, x)
vrelu = np.vectorize(relu)

# initiale input vector
I = np.random.rand(INPUT_DIMS, 1)

# initialse weights matrizes
def initialise_weights():
    W1 = np.random.rand(INPUT_DIMS, OUTPUT_DIMS_0)
    W2 = np.random.rand(OUTPUT_DIMS_0, OUTPUT_DIMS_1)
    W3 = np.random.rand(OUTPUT_DIMS_1, OUTPUT_DIMS_final)
    return W1, W2, W3

# compute forward pass
def compute_forward_pass(I, W1, W2, W3):
    H1 = np.matmul(W1.transpose(), I)
    H1 = vrelu(H1)
    H2 = np.matmul(W2.transpose(), H1)
    H2 = vrelu(H2)
    Y1 = np.matmul(W3.transpose(), H2)
    Softmax = compute_softmax(Y1)
    return Softmax

# compute loss
def compute_softmax(x):
    log = np.exp(x - x.max())
    return log/sum(log)

def compute_loss(Y1, label: int):
    return (-1) * np.log(Y1[label, 0])

def compute_softmax_derivative(softmax, target_vector):
    return softmax - target_vector.reshape(-1,1)

(W1, W2, W3) = initialise_weights()
softmax = compute_forward_pass(I, W1, W2, W3)
softmax_derivative = compute_softmax_derivative(softmax, np.array([[0,1,0]]))
loss = compute_loss(softmax, 1)

print(softmax)
print(softmax_derivative)
print(loss)
