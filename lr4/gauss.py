import numpy as np

def gauss(A :np.ndarray, B :np.ndarray) -> np.ndarray:
    A = np.copy(A)
    B = np.copy(B)
    _forwardMove(A, B)
    return _backwardMove(A, B)


def _forwardMove(A :np.ndarray, B :np.ndarray):
    for i in range(0, np.size(A, 0)):
        j = i + max(enumerate(A[i:, i]), key=lambda v: abs(v[1]))[0]
        if i != j:
            A[(i, j),:] = A[(j, i),:]
            B[(i, j),] = B[(j, i),]
        
        for j in range(i+1, np.size(A, 0)):
            k = A[j, i] / A[i, i]
            A[j, i:] -= k * A[i, i:]
            B[j] -= k * B[i]

def _backwardMove(A :np.ndarray, B :np.ndarray) -> np.ndarray:
    X = np.empty(B.size)
    for i in range(B.size-1, -1, -1):
        X[i] = (B[i] - A[i, i+1:].dot(X[i+1:])) / A[i, i]
    
    return X
