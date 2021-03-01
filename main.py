import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as fp, misc
"""
RASCUNHO-OLHAR OPCAO2.PY
"""
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


# lê imagem de entrada (f) em escala de cinza
#f = misc.face(gray=True)
f= cv2.imread('Lenna.png',0)
plt.figure()
cmap = 'gray'
plt.title('imagem original')
plt.imshow(f, cmap=cmap)
plt.colorbar()

#calculando a FFT 2D
F = fp.fft2(f)

#tratando a FFT para os gráficos
Fm = np.absolute(F)
Fm /= Fm.max()
Fm = fp.fftshift(Fm)
Fm = np.log(Fm)

#mostrando a FFT em escala logaritmica
plt.figure()
plt.title('FFT em escala log')
plt.imshow(Fm, cmap=cmap)
plt.colorbar()

# gerando a funcao gaussiana (o filtro)
sigma = 5
G = gaussian(f, sigma)

# aplicando o filtro gaussiano (gaussian blur)
Fg = F*G
plt.figure()
plt.title('FFT filtrada em escala log')
Fga = np.absolute(Fg)
Fga = fp.fftshift(Fga)
Fga = np.log(Fga+1e-6)
plt.imshow(Fga, cmap=cmap)

# obtendo a transformada inversa, que é o sinal original (a imagem) filtrado
f_blurred = fp.ifft2(Fg)
f_blurred = np.absolute(f_blurred)

plt.figure()
plt.title('Imagem filtrada')
plt.imshow(f_blurred, cmap=cmap)
plt.colorbar()


plt.show()





