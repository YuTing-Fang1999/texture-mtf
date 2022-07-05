import cv2
import numpy as np
from math import e

# compute the average of over all directions
def radialAverage(arr):
    assert arr.shape[0] == arr.shape[1]

    N = arr.shape[0]
    # Calculate the indices from the image
    y, x = np.indices(arr.shape)
    center = np.array([N//2, N//2])
    r = np.hypot(x - center[0], y - center[1])

    # 依半徑大小將 r 的 index 由小排到大
    ind = np.argsort(r.flat)
    # 依 index 取 r (由小排到大的半徑)
    r_sorted = r.flat[ind]
    # 依 index 取 img 的值
    i_sorted = arr.flat[ind]

    # 將 r 轉為整數
    r_int = r_sorted.astype(int)

    # 找出半徑改變的位置 rind=[0,8] 代表在0~1、8~9之間改變 => 0, 1~8, 9~24
    deltar = r_int - np.roll(r_int, -1)  # shift and substract
    rind = np.where(deltar!=0)[0]       # location of changed radius

    # 對陣列的值做累加
    csim = np.cumsum(i_sorted, dtype=float)
    # 累加的值
    tbin = csim[rind]
    # 算出累加的區間
    tbin[1:] -= csim[rind[:-1]]

    nr = rind - np.roll(rind, 1)
    nr = nr[1:]
    # 第一個值(圓心)不用除
    tbin[1:] /= nr

    return tbin

############################ compute I_hat(m, n) 

# read img
I = cv2.imread('Dead-leaves-test-target.jpg', cv2.IMREAD_GRAYSCALE)

# crop img to NxN square
N = min(I.shape)

# let N be the odd number
if N%2==0: N-=1
I = I[:N, :N] 

# Take the fourier transform of the image.
I_hat = np.fft.fft2(I)

# shift
# [-N/2, N/2] => [0, N]
# I(0,0) => I(N//2, N//2)
I_hat = np.fft.fftshift(I_hat) 

# get the real part
I_hat = np.abs(I_hat)

# I(0,0) => I(N//2, N//2) = N * N * E(I)
# print(I_hat[N//2,N//2])
# print(np.sum(I))

############################ compute c(N)

eta = -1.93
Denominator = 0
for m in range(0, N):
    for n in range(0, N):
        if m==N//2 and n == N//2: continue
        Denominator += (1 / pow(((m-N//2)**2 + (n-N//2)**2), eta/2))
cN = (I.var() / Denominator) * (N**4)

############################ compute T_hat(m, n)

T_hat = np.zeros((N,N))
for m in range(0, N):
    for n in range(0, N):
        if m==N//2 and n == N//2: continue
        T_hat[m,n] = cN / ((m-N//2)**2 + (n-N//2)**2)**(eta/2)
# when m==0 and n == 0
T_hat[N//2,N//2] = I_hat[N//2,N//2]

############################ compute K(m, n)

K = I_hat / T_hat 
# print(K[N//2, N//2])

############################ compute MTF

# The one-dimensional texture MTF is the average of over all directions.
MTF = radialAverage(K)

############################ compute CSF

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

############################ compute Acutance

A = np.sum([ MTF[v] * CSF[v] for v in range(MTF.shape[0])])
print(A)




