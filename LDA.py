import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import KNeighborsClassifier

def LDA(D, labels):
    classes = np.unique(labels)
    n_classes = len(classes)
    n_features = D.shape[1]

    means = np.zeros((n_classes, n_features))
    S = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))

    overall_mean = np.mean(D, axis=0)

    for i, class_label in enumerate(classes):
        class_indices = np.where(labels == class_label)[0]
        class_data = D[class_indices]
        means[i] = np.mean(class_data, axis=0)
        diff = class_data - means[i]
        S += np.dot(diff.T, diff)
        SB += 10 * np.outer(means[i] - overall_mean, means[i] - overall_mean)

    dott = np.dot(np.linalg.inv(S), SB)
    print(dott.shape)

    eigenvalues, eigenvectors = np.linalg.eigh(dott)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    U = eigenvectors[:, :n_classes-2]
    
    with open(f"LDA_{D.shape[0]}_{n_classes}", "wb") as f:
        np.save(f, U)
    return U