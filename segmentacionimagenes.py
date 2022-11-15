from email.base64mime import header_length
from email.mime import image
from re import M
from sys import maxunicode
import numpy as np
import cv2
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.special import expit
from scipy.stats import multivariate_normal
inicio=time.time()

#Carga de imagen
imagen = cv2.imread('imagen3.jpg')
#cv2.imshow('imagen',imagen)
resolucionx=imagen.shape[1]
resoluciony=imagen.shape[0]
print("La resolución de la imagen es:")
print('X:',resolucionx)
print('Y:',resoluciony)
print("\nEl total de pixeles es:")
print(resolucionx*resoluciony)

#creación de vector medio aleatorio y matriz de covarianza aleatoria y lambdas
mu1=([random.random(), random.random(), random.random()])
mu2=([random.random(), random.random(), random.random()])
lambda1=0.50+(random.randint(-9,9)/100)
lambda2=1-lambda1
r1=random.random()
r2=random.random()
r3=random.random()
sigma1=np.array([
    [1, r1, r2],
    [r1, 1, r3],
    [r2, r3, 1]
])
r1=random.random()
r2=random.random()
r3=random.random()
sigma2=np.array([
    [1, r1, r2],
    [r1, 1, r3],
    [r2, r3, 1]
])

mrik1=np.zeros((resoluciony,resolucionx))
mrik2=np.zeros((resoluciony,resolucionx))
sumrik1=0
sumrik2=0
sumriks=0
sumrik1x=0
sumrik2x=0
sumrik1xr=0
sumrik1xg=0
sumrik1xb=0
sumrik2xr=0
sumrik2xg=0
sumrik2xb=0

#prints
print(mu1)
print(mu2)
print(lambda1)
print(lambda2)

#calculo de rik
for y in range(resoluciony):
    for x in range(resolucionx):
        (b, g, r) = imagen[y, x]
        rgb=np.array([r,g,b])
        rgb=rgb/255;
        n1=lambda1*multivariate_normal.pdf(rgb,mu1,sigma1)
        n2=lambda2*multivariate_normal.pdf(rgb,mu2,sigma2)
        rik1=n1/(n1+n2)
        rik2=n2/(n1+n2)
        mrik1[y][x]=rik1
        mrik2[y][x]=rik2
        sumrik1=sumrik1+rik1
        sumrik2=sumrik2+rik2
        sumriks=sumriks+rik1+rik2
        sumrik1xr=sumrik1xr+(rik1*(r/255))
        sumrik1xg=sumrik1xg+(rik1*(g/255))
        sumrik1xb=sumrik1xb+(rik1*(b/255))
        sumrik2xr=sumrik2xr+(rik2*(r/255))
        sumrik2xg=sumrik2xg+(rik2*(g/255))
        sumrik2xb=sumrik2xb+(rik2*(b/255))

#calculo de lambdas
lambda1=sumrik1/sumriks
lambda2=sumrik2/sumriks

#calculo de mus
mu1[0]=sumrik1xr/sumrik1
mu1[1]=sumrik1xg/sumrik1
mu1[2]=sumrik1xb/sumrik1
mu2[0]=sumrik2xr/sumrik2
mu2[1]=sumrik2xg/sumrik2
mu2[2]=sumrik2xb/sumrik2

#prints
print(mu1)
print(mu2)
print(lambda1)
print(lambda2)

fin=time.time()
print("El tiempo de ejecución es:")
print(fin-inicio)