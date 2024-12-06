# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 08:12:18 2024

@author: shiwenli
"""

"""
TP 1 - Entropie - Quantification
UE455 M1 E3A
Partie Entropie
"""

import numpy as np

#import scipy.io
from scipy.io import wavfile

# Lecture de 'Declaration1789.txt'
with open('Declaration1789.txt', 'r', encoding='utf-8') as file:
    declaration_text = file.read()

filename = 'Alarm05.wav'
samplerate, data = wavfile.read(filename)

    
def entropy(x):
    """ Fonction pour calcule l'entropie sans memoire de la source x """
        
    # Initialiser un dictionnaire pour stocker
    frequences = {}
    longueur = len(x)
    
    # ndarray ne peut pas utiliser la methode set()
    symboles_set = set(x)   
    print(f"Alphabet de x est \n {symboles_set}")
    frequences = {symbole: x.count(symbole) for symbole in symboles_set}
    
    entropie = 0
    for frequence in frequences.values():
        probabilite = frequence / longueur
        entropie -= probabilite * np.log2(probabilite) 
        
    return entropie

def entropy2(x):
    """ Fonction pour calcule l'entropie de Markov d'ordre 1 de la source x """
    longueur_p = len(x)   # Longueur de la source
    
    symboles_set_mar = set(x) 
    longueur_set = len(symboles_set_mar)    # Nombre d'alphabet
    # Frequence d'apparition
    frequences_p = {symbole: x.count(symbole) / longueur_p for symbole in symboles_set_mar}
    
    # Vecteur de probabilite
    symbole_array = np.array(list(frequences_p.values()))
    print(f"Le vecteur de probabilite est \n {symbole_array}")
    
    symbole_index = {symbole: index for index, symbole in enumerate(symboles_set_mar)}
    
    # Pour strocker les frequences des paire
    transition_matrice = np.zeros((longueur_set, longueur_set))
    
    for i in range(longueur_p - 1):
        symbole_current = x[i]
        symbole_next = x[i+1]
        
        current_index = symbole_index[symbole_current]
        next_index = symbole_index[symbole_next]
        
        # Stocker les paires dans la matrice de transition
        transition_matrice[current_index, next_index] += 1

        
    # Normalisation de la matrice de transition (matrice de probabilite)
    transition_matrice /= transition_matrice.sum(axis=1, keepdims=True)
    
    # Affichage de la matrice de transition avant normalisation
    #print(f"La matrice de transition est \n {transition_matrice}")
    
    entropie2 = 0

    # Calcul de l'entropie du modele de Markov d'ordre 1
    for i in range(longueur_set):
        entropie2 += -sum(transition_matrice[i,:] * symbole_array[i] * np.log2(transition_matrice[i,:] + 1e-10))
    
    return entropie2

def entropy3(x):
    """ Fonction pour calcule l'entropie sans memoire de la source x """
    
    alphabet = []
    occurrences = []
    # Pour ndarray
    x=list(x)
    for i in range(0,len(x)):
        if x[i] not in alphabet:
            alphabet.append(x[i])
            occurrences.append(x.count(x[i]))

    freq = np.array(occurrences)/len(x)
    entropie3 = -sum(freq*np.log2(freq))

    return entropie3

# Exemple d'utilisation de la fonction entropy
x = [14, 2, 2, 1, 2, 1, 1, 5, 3, 2, 14, 3]
entropie_estimee = entropy(x)
print(f"Entropie estimée de l'exemple: {entropie_estimee:.2f} bits/symbole")

entropie2_estimee = entropy2(x)
print(f"Entropie de Markov de l'exemple: {entropie2_estimee:.2f} bits/symbole")

entropie_text = entropy(declaration_text)
print(f"Entropie estimée du texte: {entropie_text:.2f} bits/symbole")

entropie2_text = entropy2(declaration_text)
print(f"Entropie de Markov du texte: {entropie2_text:.2f} bits/symbole")

entropie_audio = entropy3(data[:, 0])
print(f"Entropie d'audio: {entropie_audio:.2f} bits/symbole")


