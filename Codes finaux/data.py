import numpy as np
from random import random 
    
# Matrice deformation tige :
# - Entree : n (int)
# - Sortie : A (matrice), b (vecteur (np.array))
def generateMatDefDF(n):
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(n):
        A[i,i] = 2
        if(i<n-1):
            A[i,i+1] = -1
        if(i>0):
            A[i,i-1] = -1
            
        b[i] = np.exp(-5*(i*1.0/n - 0.5)**2)/((n+1)*(n+1))    
    return [A,b]
    
# Matrice ondes harmoniques (Helmholtz) :
# - Entrées : ww (frequence, float), n (int)
# - Sortie : A (matrice), b (vecteur (np.array))
def generateMatHelmDF(n,ww):
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(n):
        A[i,i] = 2 - ww*ww/((n+1)*(n+1))
        if(i<n-1):
            A[i,i+1] = -1
        if(i>0):
            A[i,i-1] = -1
        b[i] = np.exp(-5*(i*1.0/n - 0.5)**2) /((n+1)*(n+1))   
    return [A,b]        
    

# Matrice ondes transmission (Partie 2 TP) :
# - Entree : ww (requence,float), listrho (list t.q. listrho[0] = rho_0 et listrho[n+1] = rho_{n+1}), listxi (list t.q. listxi[0] = x1 et listxi[n] = x_{n+1})
# - Sortie : A (matrice (np.array)), b (vecteur (np.array))

## ATTENTION : listxi[0] = x1 et listrho[0] = rho0 !!!!!!!!!!!!!!!!!!!!!! 
def generateMatHelmAna(ww,listrho,listxi):
    n = len(listrho)-2
    A = np.zeros( (2*(n+2),2*(n+2)),dtype='complex')
    b = np.zeros(2*(n+2),dtype='complex')
    
    # Conditions de raccord en x1, x2, ..., xn+1 (lignes de la matrice A de 2 à 2n+3). 
    for i in range(n+1):
        rhoi = listrho[i]
        xiP1 = listxi[i]
        rhoip1 = listrho[i+1]
        
    # Continuité de u à l'interface x_i (ligne 2i+1)
        A[2*i+1, 2*i] = np.exp(1j * ww * np.sqrt(rhoi) * xiP1)
        A[2*i+1, 2*i+1] = np.exp(-1j * ww * np.sqrt(rhoi) * xiP1)
        A[2*i+1, 2*(i+1)] = -np.exp(1j * ww * np.sqrt(rhoip1) * xiP1)
        A[2*i+1, 2*(i+1)+1] = -np.exp(-1j * ww * np.sqrt(rhoip1) * xiP1)
        
        # Continuité de u' à l'interface x_i (ligne 2i+2)
        A[2*i+2, 2*i] = np.sqrt(rhoi) * np.exp(1j * ww * np.sqrt(rhoi) * xiP1)
        A[2*i+2, 2*i+1] = -np.sqrt(rhoi) * np.exp(-1j * ww * np.sqrt(rhoi) * xiP1)
        A[2*i+2, 2*(i+1)] = -np.sqrt(rhoip1) * np.exp(1j * ww * np.sqrt(rhoip1) * xiP1)
        A[2*i+2, 2*(i+1)+1] = np.sqrt(rhoip1) * np.exp(-1j * ww * np.sqrt(rhoip1) * xiP1)
        
            
    # Pour imposer T0 = 1 (ligne 1 de la matrice A)  
    A[0,:] = A[0,:]*0.
    # Completer ICI :
    A[0,0]=1
    b[0] = 1
    
    # Pour imposer Rn+1 = 0 (ligne 2n+4 de la matrice A)
    A[2*n+3,:] = A[2*n+3,:]*0.
    # Completer ICI :
    A[2*n+3,2*n+3]=1
    b[2*n+3]=0
    
        
    return [A,b]
    
    
## Generation des propriétés du milieu hétérogène
# - Entrée : n (entier)
# - Sortie : listrho (list), listxi (list)    
def listRhoEtXi(n):
    listrho = []
    listxi = []
    
    # rho0
    listrho.append(1.0)
    
    for i in range(n):
        listxi.append((i+0.)/(n))
        ## Milieu homogène
        #listrho.append(1.0) 
        
        ## Milieu périodique
        listrho.append(1.0+10*((i+1)%2)) 
        # if(i == n/2 or i == (n-1)/2):
#             listrho[-1] = 1
    
    # Ajout a la liste listxi de xn+1
    listxi.append(1.0)
    # Ajout a la liste listrho de rhon+1
    listrho.append(1.0)
    
    return [listrho,listxi]    
        