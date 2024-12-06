# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 00:44:43 2024

@author: shiwenli
"""

import numpy as np

def entropy(x):
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
    entropie = -sum(freq*np.log2(freq))

    return entropie
