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

#Transpuesta de matriz
def transpose(matrix):
        
    result = [[None for i in range(len(matrix))] for j in range(len(matrix[0]))]
    
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            result[i][j] = matrix[j][i]
            
    return result

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
    [0.93572708, 0.37072349, 0.17011359],
    [0.37072349, 0.87168998, 0.2964515 ],
    [0.17011359, 0.2964515,  0.51052869]
])
r1=random.random()
r2=random.random()
r3=random.random()
sigma2=np.array([
    [0.93572708, 0.37072349, 0.17011359],
    [0.37072349, 0.87168998, 0.2964515 ],
    [0.17011359, 0.2964515,  0.51052869]
])

#prints
print(mu1)
print(mu2)
print(lambda1)
print(lambda2)
print(sigma1)
print(sigma2)


#inicio de iteraciones
for i in range(5):

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
            sumrik1+=rik1
            sumrik2+=rik2
            sumriks+=rik1+rik2
            sumrik1xr+=(rik1*(r/255))
            sumrik1xg+=(rik1*(g/255))
            sumrik1xb+=(rik1*(b/255))
            sumrik2xr+=(rik2*(r/255))
            sumrik2xg+=(rik2*(g/255))
            sumrik2xb+=(rik2*(b/255))

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

    sumrik1xmu1=0
    sumrik2xmu2=0

    #calculo de sigmas
    for y in range(resoluciony):
        for x in range(resolucionx):
            (b, g, r) = imagen[y, x]
            rgb=np.array([[r,g,b]])
            rgb=rgb/255;
            rgb_mu1=rgb-mu1
            rgb_mu2=rgb-mu2
            rgb_mu1t=transpose(rgb_mu1)
            rgb_mu2t=transpose(rgb_mu2)
            rik1xmu1=mrik1[y][x]*rgb_mu1*rgb_mu1t
            rik2xmu2=mrik2[y][x]*rgb_mu2*rgb_mu2t
            sumrik1xmu1+=rik1xmu1
            sumrik2xmu2+=rik2xmu2

    sigma1=sumrik1xmu1/sumrik1
    sigma2=sumrik2xmu2/sumrik2

#prints
print("Datos nuevos")
print(mu1)
print(mu2)
print(lambda1)
print(lambda2)
print(sigma1)
print(sigma2)

for y in range(resoluciony):
        for x in range(resolucionx):
           imagen[y, x] = (mu1[2], mu1[1], mu1[0]) 

cv2.imshow('imagen',imagen)
fin=time.time()
print("El tiempo de ejecución es:")
print(fin-inicio)