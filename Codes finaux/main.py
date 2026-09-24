import numpy as np
import affichage
import matplotlib.pyplot as plt

import algo
import data



"""
    Resolution du probleme A x = b par une methode directe type LU bande

    Partie 1 : Les matrices tridiagonales
    Partie 2 : Ondes en régime harmonique
"""

################################################################################
# Choix de l'application
################################################################################
print("Que souhaitez vous faire ?")
print("1. Calculer le determinant d'une matrice tridiagonale (Partie 1).")
print("2. Résoudre un système linéaire tridiagonale (Partie 1).")
print("3. Simuler une onde harmonique dans un milieu (Partie 2).")
print("4. Illustrer le phénomène de Band Gap (Partie 2).")
choixPartie = int(input("Votre choix : "))



################################################################################
# Partie 1 : Calcul de determinant et systeme tridiagonale
################################################################################

# Calcul du déterminant d'une matrice tridiagonale
if(choixPartie == 1):
    
    ## Parametres du probleme onde harmonique :
    ww = 100
    
    # Calcul du déterminant en fonction de n :
    Nmax = 100
    listdet = []
    for n in range(1,Nmax+1):
        ## Generation des données : (tester l'un et l'autre)
        [A,b] = data.generateMatDefDF(n)
        #[A,b] = data.generateMatHelmDF(n,ww)
        
        det = algo.calculDetMatriceTri(A)
        listdet.append(det)
        print(det)
    
    affichage.afficher([range(1,Nmax+1)],[listdet],['Det'])
################################################################################

################################################################################

# Résolution d'un système tridiagonale
if(choixPartie == 2):    
    
    ## Largeur de Bande :
    # Completer ICI :
    lb = 3
    
    ## Generation des données :
    n = 100
    h = 1.0/(n+1)
    ww = 10
    ## Generation des données : (tester l'un et l'autre)
    #[A,b] = data.generateMatDefDF(n)
    [A,b] = data.generateMatHelmDF(n,ww)
    
    # Factorisation Cholesky bande :
    [L,U] = algo.luBand(A,lb)
    
    ## Résolution du systeme linéaire :
    # Descente :
    # Completer ICI :
    y = algo.descenteBand(L,b,lb)
    # Remontee:
    # Completer ICI :
    sol = algo.remonteeBand(U, y, lb)

    
    # Affichage du résultat :
    xx = np.arange(h,1.0,h)
    affichage.afficher([xx],[sol],['']) # le dernier argument est pour mettre une legende au graphique 
    
    ## Test numerique pour valider le calcul :
    # Completer ICI pour implémenter un test de validation de vos méthodes :

    valeurs_n_test = [10, 50, 100, 200]  # Différentes tailles de matrices
    largeur_bande_n = 3  # Largeur de bande pour les matrices

    for n_test in valeurs_n_test:
        print(f"Test pour n = {n_test}")
    
        # Génération des matrices (tester différentes matrices)
        [A_n_test, b_n_test] = data.generateMatDefDF(n_test)
        [A_n_helm_test, b_n_helm_test] = data.generateMatHelmDF(n_test, 100)  # Exemple avec Helmholtz
    
        # Factorisation LU bande
        [L_n_test, U_n_test] = algo.luBand(A_n_test, largeur_bande_n)
    
        # Résolution avec descente et remontée
        y_n_test = algo.descenteBand(L_n_test, b_n_test, largeur_bande_n)
        sol_n_test = algo.remonteeBand(U_n_test, y_n_test, largeur_bande_n)
    
        # Validation numérique : A_n_test * sol_n_test doit être proche de b_n_test
        b_n_test_calcule = A_n_test @ sol_n_test  # Produit de la matrice A par la solution calculée
        erreur_n_test = np.linalg.norm(b_n_test_calcule - b_n_test)  # Calcul de l'erreur entre b_calculé et b
        print(f"Erreur pour n = {n_test} : {erreur_n_test}")

################################################################################

    n2 = 100 # Taille de la matrice

    # Génération de deux types de matrices
    [A_dft, b_dft] = data.generateMatDefDF(n2)  # Matrice DefDF
    [A_helm, b_helm] = data.generateMatHelmDF(n2, 100)  # Matrice Helmholtz avec ww=100

    # Test pour la Matrice DefDF
    print("Test pour la matrice DefDF :")
    [L_dft, U_dft] = algo.luBand(A_dft, lb)
    y_dft = algo.descenteBand(L_dft, b_dft, lb)
    soL_dft = algo.remonteeBand(U_dft, y_dft, lb)

    # Validation : A_dft * soL_dft doit être proche de b_dft
    b_dft_calcule = A_dft @ soL_dft  # Produit matrice-solution
    erreur_matrice_def = np.linalg.norm(b_dft_calcule - b_dft)  # Calcul de l'erreur
    print(f"Erreur pour la matrice DefDF : {erreur_matrice_def}")

    # Test pour la Matrice Helmholtz
    print("Test pour la matrice Helmholtz :")
    [L_helm, U_helm] = algo.luBand(A_helm, lb)
    y_helm = algo.descenteBand(L_helm, b_helm, lb)
    soL_helm = algo.remonteeBand(U_helm, y_helm, lb)

    # Validation : A_helm * soL_helm doit être proche de b_helm
    b_helm_calcule = A_helm @ soL_helm  # Produit matrice-solution
    erreur_matrice_helm = np.linalg.norm(b_helm_calcule - b_helm)  # Calcul de l'erreur
    print(f"Erreur pour la matrice Helmholtz : {erreur_matrice_helm}")

################################################################################

################################################################################
# Partie 2 : Ondes en régime harmonique
################################################################################

# Simuler une onde en régime harmonique dans un milieu hétérogène
if(choixPartie == 3):
    ## Parametres du probleme:
    n = 11
    ww = 3*np.pi
    
    # Construction des listes de parametres:
    [listrho,listxi] = data.listRhoEtXi(n)
    
    # Generation des donnees (matrice et second membre v) :
    [A,b] = data.generateMatHelmAna(ww,listrho,listxi)
    
    # Largeur de bande de la matrice:
    # Modifier ICI :
    lb = 4
    [L,U] = algo.luBand(A,lb)
    
    ## Résolution du systeme linéaire :
    # Descente :
    # Completer ICI :
    y =algo.descenteBand(L,b,lb)
    
    # Remontee:
    # Completer ICI :
    sol =algo.remonteeBand(U, y,lb)
    
    ## Affichage de l'onde dans tout le domaine
    affichage.afficherOndes(n,sol,listrho,listxi,ww)
    ## Animation en temps de la propagation de l'onde.
    affichage.animOndesHarmo(n,sol,listrho,listxi,ww)


# Illustration du phenomene de "Band Gap"
if(choixPartie == 4):
    ## Parametres du probleme:
    n = 11
    # Construction des listes de parametres:
    [listrho,listxi] = data.listRhoEtXi(n)
    
    # Liste de frequence:
    listww = np.arange(0.1,100,0.2)
    listR = listww*0. + 0j
    listT = listww*0. + 0j
    listSol = []
    
    for iw, ww in enumerate(listww):
        # Generation des donnees (matrice et second membre v) :
        [A,b] = data.generateMatHelmAna(ww,listrho,listxi)
    
        # Largeur de bande de la matrice:
        lb = 4
        
    
        ## Résolution du systeme linéaire :
        # Completer ICI :
        [L, U] = algo.luBand(A, lb)
        y = algo.descenteBand(L, b, lb)  # Résolution de L * y = b
        sol = algo.remonteeBand(U, y, lb)  # Résolution de U * sol = y

        
        
        listSol.append(sol)
        R0 = sol[1]
        TNp1 = sol[-2] # -2
        
        listR[iw] = R0
        listT[iw] = TNp1
        
        
    ## Animation evolution en fonction de w:
    affichage.animBandeGap(n,sol,listrho,listxi,listww,listSol)
    ## Affichage frequence interdite (band gap):
    affichage.afficher([listww,listww,listww],[np.abs(listR),np.abs(listT),np.abs(listR)**2+np.abs(listT)**2],['R0','Tn+1','R0^2 + (Tn+1)^2'])   
    
    
    
    
print ("\n\n")
print ("FIN \n")
########################################




