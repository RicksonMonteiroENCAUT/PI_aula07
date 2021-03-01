import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Para vizualizar as informações no domínio do espaço é necessário normalizar os valores
deslocar frequencia, valor absoluto, escala log. Para operações usa-se a transformada direta sem normalizações.

'''
def gaussian(S, sigma=1, vmax=0.05):
    '''
        Retorna uma gaussiana de média zero e desvio padrão sigma
        G(u, v) = 1/(2*pi*sigma^2) * exp(-(u^2 + v^2)/(2*sigma^2))
        no intervalo [-vmax, vmax] x [-vmax, vmax]
        deslocada com fftshift
        O array retornado tem a mesma dimensão da imagem (S)
    '''
    Nv, Nu =S.shape
    u=Nu* np.linspace(-vmax,vmax, Nu)
    v=Nv*np.linspace(-vmax, vmax, Nv)

    U, V = np.meshgrid(u, v)

    sigma2 = sigma**2
    G = np.exp(-(U*U + V*V) / 2. / sigma2)
    G=np.fft.fftshift(G)

    return G
cmap='gray'
img = cv2.imread('Lenna.png',0)
#Transformada
F= np.fft.fft2(img)

#normalizando
Fm=np.absolute(F)
Fm=np.fft.fftshift(Fm)
Fm=np.log(Fm)

#mostrando a FFt em escala logaritimica
plt.figure()
plt.title('FFT em escala log')
plt.imshow(Fm, cmap=cmap)
plt.colorbar()
plt.show()

sigma = 7
G = gaussian(img, sigma)
#Aplicando o filtro gaussiano
Fg=F*G

Fga= np.absolute(Fg)
Fga=np.fft.fftshift(Fga)
Fga = np.log(Fga+1)
plt.figure()
plt.title('Imagem filtrada')
plt.imshow(Fga, cmap=cmap)
plt.colorbar()
plt.show()

#Retornando para o domínio do espaço
f_blur=np.fft.ifft2(Fg)
f_blur=np.absolute(f_blur)
plt.figure()
plt.title('Imagem filtrada')
plt.imshow(f_blur, cmap=cmap)
plt.show()
plt.imshow(img, cmap=cmap)
plt.show()
