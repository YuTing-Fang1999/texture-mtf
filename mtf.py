import cv2
import numpy as np
from math import e

# compute the average of over all directions
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

######### compute I_hat(m, n) ######### 
# read img
I = cv2.imread('Dead-leaves-test-target.jpg', cv2.IMREAD_GRAYSCALE)
# crop img to NxN square
N = min(I.shape)
# let N be the even number
if N%2!=0: N+=1
I = I[:N, :N] 
# Take the fourier transform of the image.
I_hat = np.fft.fft2(I)
# Shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
# [-N/2 + 1, N/2] => [0, N]
I_hat = np.fft.fftshift(I_hat) 
# get the real part
I_hat = np.abs(I_hat)
print(I_hat[0,0])
######### compute c(N) ######### 
eta = 1.93
Denominator = 0
for m in range(-N//2 + 1, N//2 + 1): # m = [-N/2 + 1, N/2]
    for n in range(-N//2 + 1, N//2 + 1): # n = [-N/2 + 1, N/2]
        if m==0 and n == 0: continue
        Denominator += 1 / pow((m**2 + n**2), eta/2)
cN = (I.var() / Denominator) * (N**4)

######### compute T_hat(m, n) ######### 
T_hat = np.zeros((N,N))
for m in range(-N//2 + 1, N//2 + 1): # m = [-N/2 + 1, N/2]
    for n in range(-N//2 + 1, N//2 + 1): # n = [-N/2 + 1, N/2]
        if m==0 and n == 0: continue
        T_hat[m-(-N//2+1),n-(-N//2+1)] = cN / pow((m**2 + n**2), eta/2)
# when m==0 and n == 0
T_hat[0-(-N//2+1),0-(-N//2+1)] = 1

######### compute K(m, n) ######### 
K = I_hat / T_hat 
print(K[0,0])
# The one-dimensional texture MTF is the average of over all directions.
MTF = azimuthalAverage(K)

# contrast sensitivity function (CSF) can be used to weigh the
# different spatial frequencies, leading to a single acutance value
b = 0.2
c = 0.8
# CSF(v) = a * pow(v, c) * pow(e, -b*v)
# ∫ CSF(v) dv = 1 
# ∫ a * pow(v, c) * pow(e, -b*v) dv = 1
# a * ∫ pow(v, c) * pow(e, -b*v) dv = 1
# a = 1 / ∫ pow(v, c) * pow(e, -b*v) dv
a = 1 / np.sum([ pow(v, c) * pow(e, -b*v) for v in range(MTF.shape[0])])

CSF = [ a * pow(v, c) * pow(e, -b*v) for v in range(MTF.shape[0])]
A = np.sum([ MTF[v] * CSF[v] for v in range(MTF.shape[0])])
print(A)




