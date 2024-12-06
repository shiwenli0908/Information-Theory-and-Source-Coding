# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:22:34 2024

@author: shiwenli

UE455 TP2 Partie 1 - JPEG
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dctn, idctn
from quant import Quant, Zig

import entropy

"""
Partie 1 - JPEG
"""

'4.1 Lecture et affichage d’une image'
# Manipulation 1
I = plt.imread('lenna.256.pgm')
plt.imshow(I, cmap='gray')
plt.title("Image originale")
plt.show()

# Manipulation 2
(rows, cols) = I.shape
print("L'image est de taille:", (rows, cols))

# Manipulation 3
moyenne = np.floor(I.mean())
I = I.astype(np.float32)- moyenne

plt.imshow(I, cmap='gray')
plt.title("Image modifiee")
plt.show()

'4.2 Transformée en cosinus discrête'
# Manipulation 5
B = 8
nb_blocks_rows = int(rows/B)
nb_blocks_cols = int(cols/B)
It = np.zeros((B*B, nb_blocks_rows * nb_blocks_cols), np.float32)
for r in range(nb_blocks_rows):
    for c in range(nb_blocks_cols):
        currentblock = dctn(I[r * B:(r+1) * B, c * B:(c+1) * B], norm='ortho')
        It[:, r * nb_blocks_cols + c] = currentblock.flatten()

plt.imshow(It, cmap='gray')
plt.title("Matrice des coefficients par fréquence")
plt.show()

# Manipulation 6
idx = 0
coefs = It[idx, :]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Coefficients à la position {}".format(idx))

ax1.hist(coefs, bins=64)
ax1.set_xlabel('Histogramme')
ax2.imshow(coefs.reshape((nb_blocks_rows, nb_blocks_cols)), cmap='gray')
plt.show()

# Manipulation 7
lignes = [1, 8, 20, 40]     # Ligne 2, 9, 21, 41

for idx in lignes:
    coefs = It[idx, :]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Coefficients à la position {}".format(idx))
    
    ax1.hist(coefs, bins=64)
    ax1.set_xlabel('Histogramme')
    ax2.imshow(coefs.reshape((nb_blocks_rows, nb_blocks_cols)), cmap='gray')
plt.show()

# Manipulation 8
var = np.var(It, axis=1)
plt.semilogy(var)
plt.grid()
plt.title("Variance des blocs")
plt.xlabel('idx')
plt.ylabel('Variance')
plt.show()

# Manipulation 9
Itz = It[Zig, :]
var_zigzag = np.var(Itz, axis=1)

plt.semilogy(var, label='Coefficients de DCT originals')
plt.semilogy(var_zigzag, label='Coefficients de Zig-zag')
plt.grid()
plt.title('Comparaison des variances des Coefficients')
plt.xlabel('idx')
plt.ylabel('Variance')
plt.legend()
plt.show()

'4.3 Quantification'
# Manipulation 10
gamma = 1
q = gamma * Quant.flatten()
It_quant = np.fix(It.transpose()/q).transpose()

print(It_quant[:6, :6])

'4.4 Codage entropique'
# Manipulation 11
H = entropy.entropy(It_quant.flatten().tolist())

print(f"Entropie : {H:.3f} bits/symbole")

# Manipulation 12
entropies = [entropy.entropy(It_quant[i, :].tolist()) for i in range(It_quant.shape[0])]

H_moyenne = np.mean(entropies)

print(f"Entropie moyenne : {H_moyenne:.3f} bits/symbole")
print(f"Entropie de 1 : {entropies[0]:.3f} bits/symbole")

# Manipulation 13
It_pred = It_quant.copy()
It_pred[0, 1:] = It_quant[0, 1:] - It_quant[0, 0:-1]

H_codage_1 = entropy.entropy(It_pred.flatten().tolist())

print(f"Entropie apres codage: {H_codage_1:.3f} bits/symbole")

'4.5 Reconstitution de l’image quantifiée'

It_rec = It_pred.copy()
for i in range(1, nb_blocks_rows * nb_blocks_cols):
    It_rec[0, i] = It_rec[0, i] + It_rec[0, i-1]

It_rec = (It_rec.transpose()*q).transpose()

I_hat = np.zeros_like(I)
for r in range(nb_blocks_rows):
    for c in range(nb_blocks_cols):
        I_hat[r * B:(r+1) * B, c * B:(c+1) * B] = \
        idctn(It_rec[:, r*nb_blocks_cols+c].reshape(B, B),norm='ortho')
 
sigma2 = np.var((I - I_hat).flatten())

PSNR = 10 * np.log10(255 * 255 / sigma2)

plt.imshow(I_hat, cmap='gray')
plt.title("Image reconstruit")
plt.show()

plt.imshow(I, cmap='gray')
plt.title("Image originale")
plt.show()

# Manipulation 14, 15
gammas = np.linspace(0.1, 10, 10)

psnrs = []
debits = []

for gamma in gammas:
    q = gamma * Quant.flatten()
    It_quant2 = np.fix(It.transpose()/q).transpose()
    
    debit = entropy.entropy(It_quant2.flatten().tolist())
    debits.append(debit)
    
    It_pred2 = It_quant2.copy()
    It_pred2[0, 1:] = It_quant2[0, 1:] - It_quant2[0, 0:-1]
    
    It_rec2 = It_pred2.copy()
    for i in range(1, nb_blocks_rows * nb_blocks_cols):
        It_rec2[0, i] = It_rec2[0, i] + It_rec2[0, i-1]
        
    It_rec2 = (It_rec2.transpose()*q).transpose()
    
    I_hat2 = np.zeros_like(I)
    for r in range(nb_blocks_rows):
        for c in range(nb_blocks_cols):
            I_hat2[r * B:(r+1) * B, c * B:(c+1) * B] = \
            idctn(It_rec2[:, r*nb_blocks_cols+c].reshape(B, B),norm='ortho')

    sigma2 = np.var((I - I_hat2).flatten())
    psnr = 10 * np.log10(255 * 255 / sigma2)
    psnrs.append(psnr)
            
    plt.imshow(I_hat2, cmap='gray')
    plt.title("Image reconstruit, PSNR = {:.2f}dB".format(psnr))
    plt.show()
    
plt.figure()
plt.plot(debits, psnrs, label='Courbe PSNR')
plt.grid()
plt.xlabel('Debit')
plt.ylabel('PSNR (dB)')
plt.title('PSNR en foction de Debits')
plt.legend()
plt.show()

plt.figure()
plt.plot(gammas, psnrs, label='Courbe PSNR')
plt.grid()
plt.xlabel('Gammas')
plt.ylabel('PSNR (dB)')
plt.title('PSNR en foction de Gammas')
plt.legend()
plt.show()

plt.figure()
plt.plot(gammas, debits, label='Courbe Debits')
plt.grid()
plt.xlabel('Gammas')
plt.ylabel('Debits')
plt.title('Debits en foction de Gammas')
plt.legend()
plt.show()










