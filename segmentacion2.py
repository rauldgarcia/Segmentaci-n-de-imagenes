import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
import cv2

#Carga de imagen
imagen = cv2.imread('image1.jpg')
#cv2.imshow('imagen',imagen)
resolucionx=imagen.shape[1]
resoluciony=imagen.shape[0]

aic=np.array([[0]])
bic=np.array([[0]])

for k in range(2,16):
    imagen = cv2.imread('image1.jpg')
    gray=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    aux=np.asarray(gray)
    list=aux.reshape(1,resoluciony*resolucionx)
    list2=np.zeros((1,resoluciony*resolucionx))
    matrix=np.append(list,list2)
    data=pd.DataFrame(matrix)
    gmm=GaussianMixture(k,covariance_type='full',random_state=0).fit(data)
    medias=gmm.means_
    lmedias,cmedias=medias.shape
    label=gmm.predict(data)
    etiqueta=label.reshape(2,resoluciony*resolucionx)
    labels=np.delete(etiqueta,1,axis=0)
    mlabels=labels.reshape(resoluciony,resolucionx)

    mu=np.zeros((lmedias,3))
    cont=np.zeros((lmedias,1))
    for y in range(resoluciony):
            for x in range(resolucionx):
                for z in range(lmedias):
                    if mlabels[y][x]==z:
                        (b, g, r) = imagen[y, x]
                        mu[z][0]+=b
                        mu[z][1]+=g
                        mu[z][2]+=r
                        cont[z]+=1

    mus=mu/cont

    for y in range(resoluciony):
            for x in range(resolucionx):
                for z in range(lmedias):
                    if mlabels[y][x]==z:
                        imagen[y][x]=(mus[z][0],mus[z][1],mus[z][2])

    naic=np.array([[gmm.aic(data)]])
    aic=np.append(aic,naic,axis=0)
    nbic=np.array([[gmm.bic(data)]])
    bic=np.append(bic,nbic,axis=0)

    name='image'+str(k)+'.jpg'
    print(name)
    cv2.imwrite(name,imagen)

aic=np.delete(aic,0,0)
bic=np.delete(aic,0,0)
plt.plot(aic,label='aic')
plt.title('AIC y BIC')
plt.plot(bic,label='bic')
plt.legend()
plt.show()