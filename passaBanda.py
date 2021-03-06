import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('Lenna.png',0)
h,w=img.shape
H=np.zeros((h,w))
#Alternar o raio e a cor do circulo muda o filtro para low-pass ou high-pass
#Gaussian filter is better for low-pass aplications
cv2.circle(H, (h//2,w//2), 75,1,-1)
cv2.circle(H, (h//2,w//2), 5,0,-1)
plt.imshow(H, cmap='gray')
plt.show()

F=np.fft.fft2(img)
#G=np.fft.fft2(H)
G=np.fft.fftshift(H)

Fm=F*G
fm=np.fft.fftshift(Fm)
fm=np.absolute(fm)
fm=np.log(fm+1)
plt.imshow(fm, cmap='jet')
plt.show()

HPF=np.fft.ifft2(Fm)
f=np.absolute(HPF)

plt.imshow(f, cmap='gray')
plt.show()


