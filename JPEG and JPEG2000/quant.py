# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:57:36 2024

 Quant[8,8] = typical luminance quant matrix for JPEG
 Zig[64]  = zigzag scan index for 8x8 block for JPEG

@author: Kieffer
"""

import numpy as np

Quant = np.array([ [16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 58, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68,109,103, 77], 
          [24, 35, 55, 64, 81,104,113, 92],
          [49, 64, 78, 87,103,121,120,101],
          [72, 92, 95, 98,112,100,103, 99]]) 

# Zig array shows indices required for zigzag indexing
# of an [8,8] matrix. 

Zig = np.array([ 0,  1,  8, 16,  9,  2, 3, 10, 
       17, 24, 32, 25, 18, 11,  4,  5, 
       12, 19, 26, 33, 40, 48, 41, 34, 
       27, 20, 13,  6,  7, 14, 21, 28, 
       35, 42, 49, 56, 57, 50, 43, 36, 
       29, 22, 15, 23, 30, 37, 44, 51, 
       58, 59, 52, 45, 38, 31, 39, 46, 
       53, 60, 61, 54, 47, 55, 62, 63])



