import numpy as np
image = np.arange(121).reshape(11,11)
N = image.shape[0]
"""
Calculate the azimuthally averaged radial profile.

image - The 2D image
center - The [x,y] pixel coordinates used as the center. The default is 
            None, which then uses the center of the image (including 
            fracitonal pixels).

"""
# Calculate the indices from the image
y, x = np.indices(image.shape)
center = np.array([N//2, N//2])
r = np.hypot(x - center[0], y - center[1])

# 依半徑大小將 r 的 index 由小排到大
ind = np.argsort(r.flat)
# print(ind)
# 依 index 取 r (由小排到大的半徑)
r_sorted = r.flat[ind]
# print(r_sorted)
# 依 index 取 img 的值
i_sorted = image.flat[ind]
# print(i_sorted)

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

print(tbin)