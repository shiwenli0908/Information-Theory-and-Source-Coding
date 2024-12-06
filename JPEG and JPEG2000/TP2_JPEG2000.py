# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 01:18:49 2024

@author: shiwenli

UE455 TP2 Partie 2 - JPEG 2000
"""

import matplotlib.pyplot as plt
import numpy as np
import pywt
import copy

import entropy

"""
Partie 2 - JPEG 2000
"""

'5.1 Lecture et affichage d’une image'
I = plt.imread('lenna.256.pgm')
plt.imshow(I, cmap='gray')
plt.title("Image originale")
plt.show()

moyenne = np.floor(I.mean())
I = I.astype(np.float32)- moyenne

'5.2 Décomposition en sous-bandes'
coeffs2 = pywt.dwt2(I, 'db4')

# Manipulation 3
LL, (LH, HL, HH) = coeffs2

fig = plt.figure(figsize=(4, 4))
titles = ['Approximation', 'Détails horizontaux', 'Détails verticaux', 'Détails diagonaux']

for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap='gray')
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
fig.tight_layout()
plt.show()

niveaux = 2
I_dwt = pywt.wavedec2(I, 'db4', level=niveaux)

# Manipulation 4
I_dwt_norm = copy.deepcopy(I_dwt)

# Normalisation des coefficients pour améliorer la lisibilité
# Composante d’approximation
I_dwt_norm[0] /= np.abs(I_dwt_norm[0]).max()

# Composantes de détail
for detail_level in range(niveaux):
    I_dwt_norm[detail_level + 1] = [d/np.abs(d).max() for d in I_dwt_norm[detail_level + 1]]
    
# Affichage des coefficients normalisés
arr, slices = pywt.coeffs_to_array(I_dwt_norm)
plt.imshow(arr, cmap=plt.cm.gray)
plt.show()

# Manipulation 5
niveau3 = 3     # Change le nombre de niveaux
I_dwt3 = pywt.wavedec2(I, 'db4', level=niveau3)
I_dwt_norm3 = copy.deepcopy(I_dwt3)

I_dwt_norm3[0] /= np.abs(I_dwt_norm3[0]).max()

for detail_level in range(niveau3):
    I_dwt_norm3[detail_level + 1] = [d/np.abs(d).max() for d in I_dwt_norm3[detail_level + 1]]
    
arr3, slices3 = pywt.coeffs_to_array(I_dwt_norm3)
plt.imshow(arr3, cmap=plt.cm.gray)
plt.show()


I_hat = pywt.waverec2(I_dwt, 'db4')

plt.imshow(I_hat, cmap='gray')
plt.title("Image")
plt.show()

# Manipulation 6
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.imshow(I, cmap='gray')
ax1.set_xlabel('Originale')

ax2.imshow(I_hat, cmap='gray')
ax2.set_xlabel('Reconstruite')

ax3.imshow(I - I_hat, cmap='gray')
ax3.set_xlabel('Difference')

fig.tight_layout()
plt.show()

'5.3 Quantification'
# Manipulation 7
fig = plt.figure(figsize=(4, 4))
titles = ['Histogramme LL', 'Histogramme LH', 'Histogramme HL', 'Histogramme HH']

for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.hist(a.flatten(), bins=64)
    ax.set_title(titles[i], fontsize=10)
    
fig.tight_layout()
plt.show()

# Manipulation 8
q = 32
I_quant = copy.deepcopy(I_dwt)

# Quantification des coefficients d’approximation
I_quant[0] = np.fix(I_dwt[0] / q)

# Quantification des coefficients de détail
for detail_level in range(niveaux):
    I_quant[detail_level + 1] = [np.fix(d/q) for d in I_dwt[detail_level + 1]]

arr2, slices2 = pywt.coeffs_to_array(I_quant)
plt.imshow(arr2, cmap=plt.cm.gray)
plt.show()

# Manipulation 9
I_rec = copy.deepcopy(I_quant)

I_rec[0] = I_quant[0] * q

for detail_level in range(niveaux):
    I_rec[detail_level + 1] = [d * q for d in I_quant[detail_level + 1]]
I_hat = pywt.waverec2(I_rec, 'db4')

plt.imshow(I_hat, cmap='gray')
plt.title("Image reconstruit")
plt.show()

sigma2 = np.var((I - I_hat).flatten())
PSNR = 10 * np.log10(255 * 255 / sigma2)

print(f"PSNR est : {PSNR:.2f} dB")
print(f"Distorsion est : {sigma2:.3f}")

# Manipulation 10
pas_quant = np.linspace(1, 40, 40)

psnrs = []
distorsions = []

for pas in pas_quant:
    
    I_quant2 = copy.deepcopy(I_dwt)
    I_quant2[0] = np.fix(I_dwt[0] / pas)

    for detail_level in range(niveaux):
        I_quant2[detail_level + 1] = [np.fix(d/pas) for d in I_dwt[detail_level + 1]]
    
    I_rec2 = copy.deepcopy(I_quant2)

    I_rec2[0] = I_quant2[0] * pas

    for detail_level in range(niveaux):
        I_rec2[detail_level + 1] = [d * pas for d in I_quant2[detail_level + 1]]
    I_hat2 = pywt.waverec2(I_rec2, 'db4')
    
    sigma2 = np.var((I - I_hat2).flatten())
    psnr = 10 * np.log10(255 * 255 / sigma2)

    psnrs.append(psnr)
    distorsions.append(sigma2)

plt.figure()
plt.plot(pas_quant, psnrs, label='Courbe PSNR')
plt.grid()
plt.xlabel('Pas de quantification')
plt.ylabel('PSNR (dB)')
plt.title('PSNR en foction de Pas de quantification')
plt.legend()
plt.show()

plt.figure()
plt.plot(pas_quant, distorsions, label='Courbe Distorsion')
plt.grid()
plt.xlabel('Pas de quantification')
plt.ylabel('Distorsion')
plt.title('Distorsion en foction de Pas de quantification')
plt.legend()
plt.show()

'5.4 Codage entropique'

# Manipulation 11
def bit_par_pixel(I_quant):
    
    H = []
    R = 0;
    size = 0;
    H.append(entropy.entropy(I_quant[0].flatten().tolist()))
    R += H[-1] * I_quant[0].size
    size += I_quant[0].size
    for detail_level in range(niveaux):
        for d in I_quant[detail_level + 1]:
            H.append(entropy.entropy(d.flatten().tolist()))
            R += H[-1] * d.size
            size += d.size
    R = R / size
    
    return R

bits = bit_par_pixel(I_quant)
print(f"Bit par pixel : {bits:.3f} bits/symbole")

# Manipulation 11
pas_quant2 = np.linspace(1, 40, 10)

psnrs2 = []
debits = []

for pas in pas_quant2:
    
    I_quant3 = copy.deepcopy(I_dwt)
    I_quant3[0] = np.fix(I_dwt[0] / pas)

    for detail_level in range(niveaux):
        I_quant3[detail_level + 1] = [np.fix(d/pas) for d in I_dwt[detail_level + 1]]
    
    debit = bit_par_pixel(I_quant3)
    debits.append(debit)
    
    I_rec3 = copy.deepcopy(I_quant3)
    
    I_rec3[0] = I_quant3[0] * pas

    for detail_level in range(niveaux):
        I_rec3[detail_level + 1] = [d* pas for d in I_quant3[detail_level + 1]]
    
    I_hat3 = pywt.waverec2(I_rec3, 'db4')
    
    sigma2 = np.var((I - I_hat3).flatten())
    psnr2 = 10 * np.log10(255 * 255 / sigma2)

    psnrs2.append(psnr2)
    
    plt.imshow(I_hat3, cmap='gray')
    plt.title("Image reconstruit, PSNR = {:.2f}dB".format(psnr2))
    plt.show()
    
plt.figure()
plt.plot(debits, psnrs2, label='Courbe PSNR')
plt.grid()
plt.xlabel('Debits')
plt.ylabel('PSNR (dB)')
plt.title('PSNR en foction de Debits')
plt.legend()
plt.show()













