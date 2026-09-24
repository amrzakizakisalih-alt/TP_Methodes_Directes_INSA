import numpy as np

# Factorisation LU
#  - Entree : A (matrice)
#  - Sortie : L, U (matrices)	
def lu(A):
    (n, p) = A.shape
    if n != p:
        raise ValueError("La matrice n'est pas carrée")
    
    L = np.eye(n)
    U = np.zeros((n, p))
    for k in range(n):
        for i in range(k, n):
            U[k, i] = A[k, i]
            for j in range(k):
                U[k, i] -= L[k, j] * U[j, i]
        
        for i in range(k+1, n):
            L[i, k] = A[i, k]
            for j in range(k):
                L[i, k] -= L[i, j] * U[j, k]
            if abs(U[k, k]) < 1e-12:
                raise ZeroDivisionError("division par zéro")
            L[i, k] /= U[k, k]
    
    return L, U

# Factorisation LU pour matrices à bande
#  - Entree : A (matrice), lb (largeur de bande)
#  - Sortie : L, U (matrices)    
def luBand(A, lb):
    (n, p) = A.shape
    if n != p:
        raise ValueError("La matrice n'est pas carrée")
    
    L = np.eye(n, dtype=A.dtype)
    U = np.zeros((n, p), dtype=A.dtype)
    
    for k in range(n):
        for i in range(k, min(k + lb, n)):
            U[k, i] = A[k, i]
            for j in range(max(0, k - lb + 1), k):
                U[k, i] -= L[k, j] * U[j, i]
        
        for i in range(k+1, min(k + lb + 1, n)):
            L[i, k] = A[i, k]
            for j in range(max(0, k - lb + 1), k):
                L[i, k] -= L[i, j] * U[j, k]
            if abs(U[k, k]) < 1e-12:
                raise ZeroDivisionError("division par zéro")
            L[i, k] /= U[k, k]
    
    return L, U

# Algorithme de descente pour matrices bandes
#  - Entree : L (matrice), b (vecteur), lb (largeur de bande)
#  - Sortie : x (solution)    
def descenteBand(L, b, lb):
    (n, p) = L.shape
    x = np.zeros(n, dtype=b.dtype)
    
    for i in range(n):
        s = 0
        for j in range(max(0, i - lb), i):
            s += L[i, j] * x[j]
        x[i] = (b[i] - s) / L[i, i]
    
    return x

# Algorithme de remontée pour matrices bandes
#  - Entree : U (matrice), b (vecteur), lb (largeur de bande)
#  - Sortie : x (solution)    
def remonteeBand(U, b, lb):
    (n, p) = U.shape
    x = np.zeros(n, dtype=b.dtype)
    
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, min(i + lb + 1, n)):
            s += U[i, j] * x[j]
        x[i] = (b[i] - s) / U[i, i]
    
    return x

# Algorithme de calcul du déterminant d'une matrice tri-diagonale
# - Entree : A (matrice)
# - Sortie : detA (float)        
def calculDetMatriceTri(A):
    (n, p) = A.shape
    if n != p:
        raise ValueError("La matrice n'est pas carrée")
    
    detA1 = A[0, 0]
    if n == 1:
        return detA1
    
    detA2 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if n == 2:
        return detA2
    
    detAnm1 = detA2
    detAnm2 = detA1
    
    for i in range(2, n):
        detAn = A[i, i] * detAnm1 - A[i, i-1] * A[i-1, i] * detAnm2
        detAnm2 = detAnm1
        detAnm1 = detAn
    
    return detAn   