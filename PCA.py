import numpy as np

class eigenstruct:
    value: float
    vector: np.ndarray
    
    def __init__(self, value: float, vector: np.ndarray):
        self.value = value
        self.vector = vector
    
    def __lt__(self, other):
        return self.value < other.value


def PCA(dataMatrix: np.ndarray, alpha: float):
    mean = np.mean(dataMatrix, axis=0)
    Z = dataMatrix - mean
    COV = np.matmul(Z.T, Z) / (Z.shape[0])
    eigenvalues, eigenvectors = np.linalg.eig(COV)
    eigenstructs = [eigenstruct(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigenstructs.sort(reverse=True)
    total_variance = np.sum(eigenvalues)
    variance = 0
    i = 0
    while i < len(eigenstructs):
        variance += eigenstructs[i].value
        if variance / total_variance >= alpha:
            break
        i += 1
    
    U = np.array([eigenstructs[j].vector for j in range(i+1)])
    model_name = str(alpha) + "_" + str(dataMatrix.shape[0]) + "_" + str(dataMatrix.shape[1]) + ".pca"
    
    with open(model_name, "wb") as f:
        np.save(f, U)
    
    
    
    



    