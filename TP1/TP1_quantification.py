# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:41:22 2024

@author: shiwenli
"""

"""
TP 1 - Entropie - Quantification
UE455 M1 E3A
Partie Quantification
"""

import numpy as np
import matplotlib.pyplot as plt

#import scipy.io
from scipy.io import wavfile

# Pour le signal audio, a partir de ligne 264
filename = 'Alarm05.wav'
samplerate, data = wavfile.read(filename)


def quant_midrise(x, Delta = 1, M = np.inf):
    """ Fonction pour la quantification de type Mid-rise """
    idx = np.floor(x / Delta)
    
    idx = np.clip(idx, -M/2, M/2 - 1)   # Limiter les index
    
    qx = idx * Delta + Delta / 2.0
    
    return idx, qx

def quant_midtread(x, Delta = 1, M = np.inf):
    """ Fonction pour la quantification de type Mid-tread """
    idx = np.floor((x + Delta / 2.0 ) / Delta)
    
    idx = np.clip(idx, -M/2, M/2 - 1)   # Limiter les index
    
    qx = idx * Delta
    
    return idx, qx

def entropy(x):
    """ Fonction pour calculer l'entropie sans memoire de la source x """
    x = x.tolist()      # Transformer type ndarray a type list
    # Initialiser un dictionnaire pour stocker
    frequences = {}
    longueur = len(x)
    
    symboles_set = set(x)   
    frequences = {symbole: x.count(symbole) for symbole in symboles_set}
    
    entropie = 0
    for frequence in frequences.values():
        probabilite = frequence / longueur
        entropie -= probabilite * np.log2(probabilite)
        
    return entropie

def distorsion_quad(x, qx):
    """ Fonction pour calculer la distorsion quadratique """
    dis_quad = sum((x - qx)**2) / len(x)
    # Ou seulement dis_quad = np.var(x-qx)
    
    return dis_quad

def distorsion_delta(x, v_delta, v_M = [2,4,8,16]):
    """ Fonction pour tracer la relation entre distorsion et delta"""
    for M in v_M:

        distorsions_M = []
        
        for Delta in v_delta:
            idx_m, qx_m = quant_midrise(x, Delta, M)
            
            dis_m = distorsion_quad(x, qx_m)
            distorsions_M.append(dis_m)
        
        # Determiner la valeur minimale
        min_distorsion = min(distorsions_M)
        min_distorsion_index = distorsions_M.index(min_distorsion)
        min_delta = valeurs_delta2[min_distorsion_index]
        print(f"Pour M={M}, quand Delta = {min_delta:.2f}, la distorsion est minimale: {min_distorsion:.2f}")
        
        # Afficher la trace
        plt.plot(v_delta, distorsions_M, label=f"M={M}")

    plt.legend()
    plt.xlabel('Delta')
    plt.ylabel('Distorsion')
    plt.grid()
    plt.show()

def distorsion_entropie(x, v_delta, v_M = [2,4,8,16]):
    """ Fonction pour tracer la relation entre distorsion et entropie"""
    for M in v_M:
        D_mr_M = []
        H_mr_M = []
        
        for i in range(len(v_delta)):
            idx_r_M, qx_r_M = quant_midrise(x, v_delta[i], M)
            dis_r_M = distorsion_quad(x, qx_r_M)
            D_mr_M.append(dis_r_M)
            entro_r_M = entropy(idx_r_M)
            H_mr_M.append(entro_r_M)
        
        plt.plot(H_mr_M, D_mr_M, label=f"M={M}")

    plt.legend()
    plt.xlabel('Entropie de idx')
    plt.ylabel('Distorsion')
    plt.grid()
    plt.show()


# Exemple
x1 = np.arange(-10, 10, 0.0001)  # Generer une liste de nombres de -10 à 10
Delta = 1                        # Pas de quantification

idx1, qx1 = quant_midrise(x1, Delta)
idx2, qx2 = quant_midtread(x1, Delta)

''' Manipulation-2 et 6 '''

#plt.subplot(2, 1, 1)            # Plot pour quantification Mid-rise
plt.plot(x1, qx1, '-r')
plt.legend(['mid_rise'])
plt.xlabel('Entree')
plt.ylabel('Sortie')
plt.grid()
plt.show()

#plt.subplot(2, 1, 2)            # Plot pour quantification Mid-tread       
plt.plot(x1, qx2, '-r')
plt.legend(['mid_tread'])
plt.xlabel('Entree')
plt.ylabel('Sortie')
plt.grid()
plt.show()

''' Manipulation-3 '''

# Generer un tableau de la variable gaussienne
N = 10000
x = np.random.normal(loc=0, scale=1, size=N)

''' Manipulation-4 '''

# Differentes valeurs de Delta a evaluer
valeurs_delta = np.arange(0.1, 5, 0.01)

distorsions = []
entropies = []

# Pour chaque valeur de Delta
for Delta in valeurs_delta:
    # Quantifier x avec le Delta actuel
    idx, qx = quant_midrise(x, Delta)
    
    # Calculer la distorsion
    distorsion_ac = distorsion_quad(x, qx)
    distorsions.append(distorsion_ac)
    
    # Calculer l'entropie de idx
    entropie_ac = entropy(idx)
    entropies.append(entropie_ac)

plt.plot(valeurs_delta, distorsions, label='distorsion')
plt.legend()
plt.xlabel('Delta')
plt.ylabel('Distorsion')
plt.grid()
plt.show()    

plt.plot(valeurs_delta, entropies, label='entropie')
plt.legend()
plt.xlabel('Delta')
plt.ylabel('Entropie')
plt.grid()
plt.show()   

''' Manipulation-5 et 7 '''

arr_delta = np.arange(0.01, 1, 0.01)
D_mr = []
H_mr = []
D_mt = []
H_mt = []

for i in range(len(arr_delta)):
    idx_r, qx_r = quant_midrise(x, arr_delta[i])
    dis_r = distorsion_quad(x, qx_r)
    D_mr.append(dis_r)
    entro_r = entropy(idx_r)
    H_mr.append(entro_r)
    
    idx_t, qx_t = quant_midtread(x, arr_delta[i])
    dis_t = distorsion_quad(x, qx_t)
    D_mt.append(dis_t)
    entro_t = entropy(idx_t)
    H_mt.append(entro_t)
    
plt.plot(H_mr, D_mr)
plt.legend(['mid_rise'])
plt.xlabel('Entropie de idx')
plt.ylabel('Distorsion')
plt.grid()
plt.show()

plt.plot(H_mt, D_mt)
plt.legend(['mid_tread'])
plt.xlabel('Entropie de idx')
plt.ylabel('Distorsion')
plt.grid()
plt.show()

''' Manipulation-9 '''

valeurs_delta2 = np.arange(0.1, 5, 0.01)

valeurs_M = [2, 4, 8, 16]

print("\n La variance = 1")

distorsion_delta(x, valeurs_delta2, valeurs_M)

''' Manipulation-10 '''

arr_delta2 = np.arange(0.01, 1, 0.01)

distorsion_entropie(x, arr_delta2)

''' Manipulation-11 '''

x2 = np.random.normal(loc=0, scale=2, size=N)
x3 = np.random.normal(loc=0, scale=4, size=N)

print("\n La variance = 2")

distorsion_delta(x2, valeurs_delta2, valeurs_M)

print("\n La variance = 4")

distorsion_delta(x3, valeurs_delta2, valeurs_M)

''' Manipulation-12 '''

# Source Laplacienne
x4 = np.random.laplace(loc=0, scale=1, size=N)
x5 = np.random.laplace(loc=0, scale=2, size=N)

print("\n Source Laplacienne et la variance = 1")

distorsion_delta(x4, valeurs_delta2, valeurs_M)
distorsion_entropie(x4, arr_delta2)

print("\n Source Laplacienne et la variance = 2")

distorsion_delta(x5, valeurs_delta2, valeurs_M)
distorsion_entropie(x5, arr_delta2)


''' Manipulation-14 '''

print("\n Le signal audio")
print(f"La dimmension du signal : {data.ndim}")
print(f"Sample Rate est : {samplerate}")
print(f"Bits des echatillons est : {data.dtype}")

''' Manipulation-15 '''

# Convertir les échantillons en temps
time = np.arange(len(data)) / samplerate

# Extraire les canaux gauche et droit
voix_gauche = data[:, 0]  # 1 canal (gauche)
voix_droit = data[:, 1]   # 2 canal (droit)

# Tracer les canaux gauche et droit
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(time, voix_gauche)
plt.title('Voix gauche')
plt.xlabel('Temps')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(time, voix_droit)
plt.title('Voix droit')
plt.xlabel('Temps')
plt.ylabel('Amplitude')

plt.show()

''' Manipulation-16 '''

valeurs_delta3 = np.arange(1, 2000, 10)

print("\n Distorsion du signal audio")
distorsion_delta(voix_gauche, valeurs_delta3)








