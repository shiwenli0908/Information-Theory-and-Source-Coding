# Information Theory and Source Coding

This repository contains three practical exercises from the Information Theory and Source Coding course.
The course is UE455 in the M1 E3A program at Paris-Saclay University, taught by Professor *Michel Kieffer*.

## TP1 Entropy and Quantification

The objective of this TP1 is to estimate the entropy of a source, for example a text, an image, or an audio file. A second part is devoted to the study of two types of scalar quantizers and to evaluate the trade-off in bitrate and distortion during a scalar quantization.

## TP2 JPEG and JPEG2000

An image encoder such as JPEG or JPEG 2000 is mainly composed of three blocks. The transformation block, the quantizer and the entropy encoder. The *transformation* block is used to transform the image to be encoded in such a way as to group into subsets the information that is actually useful for the correct reconstruction of an image. In the JPEG standard, the transformation uses a discrete cosine transformation. In the JPEG standard, it is done using a pyramidal decomposition using a wavelet transform. The *quantization* block is used to force a certain number of transformed coefficients to zero, which will allow, thanks to the last block, the *entropy coding*, to achieve high compression rates, while maintaining acceptable quality.

This practical work will analyze the effect of the three blocks forming an image encoder and see, depending on the choices made, what performance can be achieved, both in terms of image quality and bit rate.

## TP3 Learned data compression

Lossy compression involves a trade-off between throughput and distortion. This TP implements a model similar to an autoencoder to compress images from the MNIST dataset. 

The method is based on the paper [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704).



