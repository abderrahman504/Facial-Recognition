import numpy as np
import scipy as sp


def PCA(dataMatrix: np.ndarray, alpha: float):
    mean = np.mean(dataMatrix, axis=0)
    Z = dataMatrix - mean
    COV = np.matmul(Z.T, Z) / (Z.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(COV)
    indeces = np.arange(0, len(eigenvalues), 1)
    sorted_idx = [x for _,x in sorted(zip(eigenvalues, indeces))][::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    total_variance = np.sum(eigenvalues)
    variance = 0
    i = 0
    while i < len(eigenvalues):
        variance += eigenvalues[i]
        if variance / total_variance >= alpha:
            break
        i += 1
    
    U = np.array(eigenvectors[:, 0:i+1])
    model_name = "PCA_" + str(alpha) + "_" + str(dataMatrix.shape[0]) + "_" + str(dataMatrix.shape[1])
    
    with open(model_name, "wb") as f:
        np.save(f, U)
    
    return U
    
    
    
    
def faster_PCA(dataMatrix: np.ndarray, alpha: float):
    mean = np.mean(dataMatrix, axis=0)
    Z = dataMatrix - mean
    cov = np.matmul(Z.T, Z) / Z.shape[0]
    eigenvalues = sp.linalg.eigh(cov, values_only=true)
    batch_size = 300
    k = Z.shape[0]-1
    while (k >= 0):
        interval = k-batch_size+1, k
        
    


