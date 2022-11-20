from email.base64mime import header_length
from email.mime import image
from re import M
from sys import maxunicode
import numpy as np
import cupy as cp
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
import numba
from numba import njit, prange
inicio=time.time()

#Carga de imagen
imagen = cv2.imread('imagen1.jpg',1)
#cv2.imshow('imagen',imagen)
resolucionx=imagen.shape[1]
resoluciony=imagen.shape[0]
print("La resolución de la imagen es:")
print('X:',resolucionx)
print('Y:',resoluciony)
print("\nEl total de pixeles es:")
print(resolucionx*resoluciony)

#Creacion de arrays r,g,b
vr=cp.zeros((resoluciony,resolucionx))
vg=cp.zeros((resoluciony,resolucionx))
vb=cp.zeros((resoluciony,resolucionx))

@njit(parallel=True)
def color (image):
    for y in prange(resoluciony):
        for x in prange(resolucionx):
            (b, g, r) = image[y, x]
            vr[y][x]=r/255
            vg[y][x]=g/255
            vb[y][x]=b/255

    return vr,vg,vb

#creación de vector medio aleatorio y matriz de covarianza aleatoria y lambdas
vr,vg,vb=color(imagen)
mu1=cp.array([random.random(), random.random(), random.random()])
mu2=cp.array([random.random(), random.random(), random.random()])
mu3=cp.array([random.random(), random.random(), random.random()])
lambda1=0.33+(random.randint(-9,9)/100)
lambda2=0.33+(random.randint(-9,9)/100)
lambda3=1-lambda1-lambda2
sigma1=cp.array([
    [0.93572708, 0.37072349, 0.17011359],
    [0.37072349, 0.87168998, 0.2964515 ],
    [0.17011359, 0.2964515,  0.51052869]
])
sigma2=cp.array([
    [0.93572708, 0.37072349, 0.17011359],
    [0.37072349, 0.87168998, 0.2964515 ],
    [0.17011359, 0.2964515,  0.51052869]
])
sigma3=cp.array([
    [0.9500897, 0.29943573, 0.68852249],
    [0.29943573, 0.0980334, 0.24556276],
    [0.68852249, 0.24556276, 0.76094257]
])

#prints
print(mu1.shape)
print(mu2.shape)
print(mu3.shape)
print(lambda1)
print(lambda2)
print(lambda3)
print(sigma1.shape)
print(sigma2.shape)
print(sigma3.shape)

#inicio de iteraciones
for i in range(1):

    mrik1=cp.zeros((resoluciony,resolucionx))
    mrik2=cp.zeros((resoluciony,resolucionx))
    mrik3=cp.zeros((resoluciony,resolucionx))
    sumrik1=0
    sumrik2=0
    sumrik3=0
    sumriks=0
    sumrik1x=0
    sumrik2x=0
    sumrik3x=0
    sumrik1xr=0
    sumrik1xg=0
    sumrik1xb=0
    sumrik2xr=0
    sumrik2xg=0
    sumrik2xb=0
    sumrik3xr=0
    sumrik3xg=0
    sumrik3xb=0

    #calculo de rik
    for y in range(resoluciony):
        for x in range(resolucionx):
            rgb=cp.array([vr[y][x],vg[y][x],vb[y][x]])
            n1=lambda1*multivariate_normal.pdf(rgb.get(),mu1.get(),sigma1.get())
            n2=lambda2*multivariate_normal.pdf(rgb.get(),mu2.get(),sigma2.get())
            n3=lambda3*multivariate_normal.pdf(rgb.get(),mu3.get(),sigma3.get())
            rik1=n1/(n1+n2+n3)
            rik2=n2/(n1+n2+n3)
            rik3=n3/(n1+n2+n3)
            mrik1[y][x]=rik1
            mrik2[y][x]=rik2
            mrik3[y][x]=rik3
            sumrik1+=rik1
            sumrik2+=rik2
            sumrik3+=rik3
            sumriks+=rik1+rik2+rik3
            sumrik1xr+=(rik1*(vr[y][x]))
            sumrik1xg+=(rik1*(vg[y][x]))
            sumrik1xb+=(rik1*(vb[y][x]))
            sumrik2xr+=(rik2*(vr[y][x]))
            sumrik2xg+=(rik2*(vg[y][x]))
            sumrik2xb+=(rik2*(vb[y][x]))
            sumrik3xr+=(rik3*(vr[y][x]))
            sumrik3xg+=(rik3*(vg[y][x]))
            sumrik3xb+=(rik3*(vb[y][x]))

    #calculo de lambdas
    lambda1=sumrik1/sumriks
    lambda2=sumrik2/sumriks
    lambda3=sumrik3/sumriks

    #calculo de mus
    mu1[0]=sumrik1xr/sumrik1
    mu1[1]=sumrik1xg/sumrik1
    mu1[2]=sumrik1xb/sumrik1
    mu2[0]=sumrik2xr/sumrik2
    mu2[1]=sumrik2xg/sumrik2
    mu2[2]=sumrik2xb/sumrik2
    mu3[0]=sumrik3xr/sumrik3
    mu3[1]=sumrik3xg/sumrik3
    mu3[2]=sumrik3xb/sumrik3

    sumrik1xmu1=0
    sumrik2xmu2=0
    sumrik3xmu3=0

    #calculo de sigmas
    for y in range(resoluciony):
        for x in range(resolucionx):
            rgb=cp.array([vr[y][x],vg[y][x],vb[y][x]])
            rgb_mu1=rgb-mu1
            rgb_mu2=rgb-mu2
            rgb_mu3=rgb-mu3
            rgb_mu1t=cp.transpose(rgb_mu1.reshape(1,3))
            rgb_mu2t=cp.transpose(rgb_mu2.reshape(1,3))
            rgb_mu3t=cp.transpose(rgb_mu3.reshape(1,3))
            rik1xmu1=mrik1[y][x]*rgb_mu1*rgb_mu1t
            rik2xmu2=mrik2[y][x]*rgb_mu2*rgb_mu2t
            rik3xmu3=mrik3[y][x]*rgb_mu3*rgb_mu3t
            sumrik1xmu1+=rik1xmu1
            sumrik2xmu2+=rik2xmu2
            sumrik3xmu3+=rik3xmu3

    sigma1=sumrik1xmu1/sumrik1
    sigma2=sumrik2xmu2/sumrik2
    sigma3=sumrik3xmu3/sumrik3

#prints
print("Datos nuevos")
print(mu1.shape)
print(mu2.shape)
print(mu3.shape)
print(lambda1)
print(lambda2)
print(lambda3)
print(sigma1.shape)
print(sigma2.shape)
print(sigma3.shape)

for y in range(resoluciony):
        for x in range(resolucionx):
            rgb=cp.array([vr[y][x],vg[y][x],vb[y][x]])
            n1=lambda1*multivariate_normal.pdf(rgb.get(),mu1.get(),sigma1.get())
            n2=lambda2*multivariate_normal.pdf(rgb.get(),mu2.get(),sigma2.get())
            n3=lambda3*multivariate_normal.pdf(rgb.get(),mu3.get(),sigma3.get())
            
            if n1>n2 and n1 >n3:
                imagen[y, x] = (mu1[2].get()*255, mu1[1].get()*255, mu1[0].get()*255)

            if n2>n1 and n2>n3:
                imagen[y, x] = (mu2[2].get()*255, mu2[1].get()*255, mu2[0].get()*255) 

            if n3>n1 and n3>n2:
                imagen[y, x] = (mu2[2].get()*255, mu2[1].get()*255, mu2[0].get()*255) 

fin=time.time()
print("El tiempo de ejecución es:")
print(fin-inicio)
cv2.namedWindow('imagen')
cv2.imshow('imagen',imagen)
cv2.waitKey(0)