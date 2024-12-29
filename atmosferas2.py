# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:36:23 2024

@author: Eneko y Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io.ascii import read
import pandas as pd
import os

#rutas para los dos modelos de atmósfera (se han modificado los archivos para que contengan solo las tablas de interés)
ruta_5000=read('C:/Users/Lenovo/Downloads/Máster/Primero/Atmósferas/Entregable 2/modelo5.dat.txt')
ruta_8000=read('C:/Users/Lenovo/Downloads/Máster/Primero/Atmósferas/Entregable 2/modelo8.dat.txt')

#aquí escribimos la ruta en la que queremos que salgan los resultados
ruta='C:/Users/Lenovo/Downloads/Máster/Primero/Atmósferas/Entregable 2'

#creamos un diccionario con las tablas
dicc={"m5": ruta_5000,
      "m8": ruta_8000}

######Apartado 1

plt.figure(1)
plt.plot(dicc["m5"]["Depth"]/10**7,dicc["m5"]["lgTauR"],color='C3')
plt.xlabel(r'$r\,\,[\mathrm{cm}\cdot 10^7]$')
plt.ylabel(r'$\log(\tau)$')
plt.ylim(-5,2)
plt.title('Profundidad óptica de Rosseland para el modelo de 5000K')
plt.grid(which='both',alpha=0.5)
name='lgtaur5.pdf' #en cada plot, modificar aquí el nombre de archivo si queremos
path = os.path.join(ruta,name) #concatena 'ruta' y 'name' para crear una sola ruta en cada caso
plt.savefig(path, dpi=300, transparent=True)

plt.figure(2)
plt.plot(dicc["m8"]["Depth"]/10**7,dicc["m8"]["lgTauR"])
plt.xlabel(r'$r\,\, [\mathrm{cm}\cdot 10^7]$')
plt.ylabel(r'$\log(\tau)$')
plt.ylim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Profundidad óptica de Rosseland para el modelo de 8000K')
name='lgtaur8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

######Apartado 2

#T
plt.figure(3)
plt.plot(dicc["m5"]["lgTauR"],dicc["m5"]["T"],color='C3')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$T\,[\mathrm{K}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Temperatura para el modelo de 5000K')
name='T5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

plt.figure(4)
plt.plot(dicc["m8"]["lgTauR"],dicc["m8"]["T"])
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$T\,[\mathrm{K}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Temperatura para el modelo de 8000K')
name='T8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#Pe
plt.figure(5)
plt.plot(dicc["m5"]["lgTauR"],dicc["m5"]["Pe"]/1000,color='C3')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{e}}\,\,[\mathrm{kdyne·cm}^{-2}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Presión electrónica para el modelo de 5000K')
name='Pe5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

plt.figure(6)
plt.plot(dicc["m8"]["lgTauR"],dicc["m8"]["Pe"]/1000)
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{e}}\,\,[\mathrm{kdyne·cm}^{-2}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Presión electrónica para el modelo de 8000K')
name='Pe8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#Pg
plt.figure(16)
plt.plot(dicc["m5"]["lgTauR"],dicc["m5"]["Pg"]/1000,color='C3')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{g}}\,\,[\mathrm{kdyne·cm}^{-2}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Presión del gas para el modelo de 5000K')
name='Pg5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

plt.figure(17)
plt.plot(dicc["m8"]["lgTauR"],dicc["m8"]["Pg"]/1000)
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{g}}\,\,[\mathrm{kdyne·cm}^{-2}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Presión del gas para el modelo de 8000K')
name='Pg8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#Pe/Pg
plt.figure(7)
plt.plot(dicc["m5"]["lgTauR"],dicc["m5"]["Pe"]/dicc["m5"]["Pg"],color='C3')
plt.xlabel(r'$\log(\tau)$')
plt.xlim(-5,2)
plt.ylabel(r'$P_{\mathrm{e}}/P_{\mathrm{g}}$')
plt.grid(which='both',alpha=0.5)
plt.title('Ratio Pe/Pg para el modelo de 5000K')
name='PePg5.pdf'
path = os.path.join(ruta,name)
plt.savefig('C:/Users/Lenovo/Downloads/Máster/Primero/Atmósferas/Entregable 2/PePg5.pdf', dpi=300, transparent=True)

plt.figure(8)
plt.plot(dicc["m8"]["lgTauR"],dicc["m8"]["Pe"]/dicc["m8"]["Pg"])
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{e}}/P_{\mathrm{g}}$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Ratio Pe/Pg para el modelo de 8000K')
name='PePg8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#Prad/Pg
plt.figure(9)
plt.plot(dicc["m5"]["lgTauR"],(dicc["m5"]["Prad"]/dicc["m5"]["Pg"])*1000,color='C3')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{rad}}/P_{\mathrm{g}}\cdot 10^3$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Ratio Prad/Pg para el modelo de 5000K')
name='PradPg5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

plt.figure(10)
plt.plot(dicc["m8"]["lgTauR"],(dicc["m8"]["Prad"]/dicc["m8"]["Pg"])*1000)
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{rad}}/P_{\mathrm{g}}\cdot 10^3$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Ratio Prad/Pg para el modelo de 8000K')
name='PradPg8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#Prad
plt.figure(18)
plt.plot(dicc["m5"]["lgTauR"],dicc["m5"]["Prad"],color='C3')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{rad}}\,\,[\mathrm{dyne·cm}^{-2}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Presión de radiación para el modelo de 5000K')
name='Prad5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

plt.figure(19)
plt.plot(dicc["m8"]["lgTauR"],dicc["m8"]["Prad"])
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$P_{\mathrm{rad}}\,\,[\mathrm{dyne·cm}^{-2}]$')
plt.xlim(-5,2)
plt.grid(which='both',alpha=0.5)
plt.title('Presión de radiación para el modelo de 8000K')
name='Prad8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

######Apartado 3
Tgris5=[]
Tgris8=[]
#formula de T para cuerpo gris
for ii in range(0,len(dicc["m5"]["lgTauR"])):
    Tgris5.append(5000*((3/4)*(10**dicc["m5"]["lgTauR"][ii]+(2/3)))**(1/4))
    Tgris8.append(8000*((3/4)*(10**dicc["m8"]["lgTauR"][ii]+(2/3)))**(1/4))
    
#ploteamos las distribuciones de T
plt.figure(11)
plt.plot(dicc["m5"]["lgTauR"],Tgris5,color='C3')
plt.plot(dicc["m8"]["lgTauR"],Tgris8,color='C0')
plt.text(0.95*dicc["m5"]["lgTauR"][0],1.15*Tgris5[0],r'$T_{\mathrm{eff}}=5000K$')
plt.text(0.95*dicc["m8"]["lgTauR"][0],1.15*Tgris8[0],r'$T_{\mathrm{eff}}=8000K$')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$T\,[\mathrm{K}]$')
plt.xlim(-5,2)
plt.grid(alpha=0.5)
plt.title('Temperaturas de cuerpo gris')
name='Tgris.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#la T frente a la Tgris
plt.figure(12)
plt.plot(Tgris5,dicc["m5"]["T"],color='C3')
plt.plot(Tgris5,Tgris5,alpha=0.5,linestyle='dashed', color='black')
plt.xlabel(r'$T\,(gris)\,[\mathrm{K}]$')
plt.ylabel(r'$T\,[\mathrm{K}]$')
plt.xlim(Tgris5[0],Tgris5[len(Tgris5)-1])
plt.grid(alpha=0.5)
plt.title('Comparación de temperaturas para el modelo de 5000K')
name='TTgris5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

plt.figure(13)
plt.plot(Tgris8,dicc["m8"]["T"])
plt.plot(Tgris8,Tgris8,alpha=0.5,linestyle='dashed', color='black')
plt.xlabel(r'$T\,(gris)\,[\mathrm{K}]$')
plt.ylabel(r'$T\,[\mathrm{K}]$')
plt.xlim(Tgris8[0],Tgris8[len(Tgris8)-1])
plt.grid(alpha=0.5)
plt.title('Comparación de temperaturas para el modelo de 8000K')
name='TTgris8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

######Apartado 4
kb_cgs = 1.38e-16 # Constante de Botlzmann en erg/K
kb = kb = 8.617e-5  # Constante de Boltzmann en eV/K
UHm = 1 # Funciones de partición
UHI = 2
UHII = 1
ChiHmHI = 0.765 # Energía de Ionización en eV
ChiHIHII = 13.6

#ecuación de Saha
def Saha(T,Pe):    
    Ne = Pe/(kb_cgs*T) # Número de electrones a partir de la presión electrónica
    NHmHI = 2.07e-16*Ne* (UHm/UHI) * T**(-3/2)* np.e**(ChiHmHI/(kb*T))     # Ec de Saha H-/HI
    NHIHII = 2.07e-16*Ne* (UHI/UHII) * T**(-3/2)* np.e**(ChiHIHII/(kb*T))  # Ec de Saha HI/HII
    NH = (NHmHI*NHIHII*Ne)/(1-(NHmHI*NHIHII))     #Numero de H- resolviendo el sistema de ecuaciones
    NHI = (NH)/(NHmHI)
    NHII = (NHI)/(NHIHII)
    return(NH,NHI,NHII,Ne)

chi = [0,10.206,12.095] # Chi en eV para n=1,2,3
gn = [2,8,18]   # g_n para n=1,2,3

#ecuación de Boltzmann
def Boltzmann(T):    # Se da la poblacion en el estado n=1 y el n que se quiere calcular (2 o 3)
    n2_n1 = (gn[1]/gn[0] * np.e**(-chi[1]/(kb*T)))
    n3_n1 = (gn[2]/gn[0] * np.e**(-chi[2]/(kb*T)))
    return(n2_n1,n3_n1)

#encontramos el valor de logtauR correspondiente a tauR=0.5 (tau1), 5.0 (tau2) y 1.0 (tau3)
#definimos también los valores de T y Pe correspondientes
for ii in range(0,len(dicc["m5"]["lgTauR"])):
    tau=10**dicc["m5"]["lgTauR"][ii]
    if tau>=0.5:
        a=tau
        b=10**dicc["m5"]["lgTauR"][ii-1]
        if abs(a-0.5)<abs(b-0.5):
            tau1=a
            T1_5=dicc["m5"]["T"][ii]
            T1_8=dicc["m8"]["T"][ii]
            Pe1_5=dicc["m5"]["Pe"][ii]
            Pe1_8=dicc["m8"]["Pe"][ii]
        else:
            tau1=b
            T1_5=dicc["m5"]["T"][ii-1]
            T1_8=dicc["m8"]["T"][ii-1]
            Pe1_5=dicc["m5"]["Pe"][ii-1]
            Pe1_8=dicc["m8"]["Pe"][ii-1]
        break
for kk in range(ii,len(dicc["m5"]["lgTauR"])):
    tau=10**dicc["m5"]["lgTauR"][kk]    
    if tau>=1.0:
        a=tau
        b=10**dicc["m5"]["lgTauR"][kk-1]
        if abs(a-1)<abs(b-1):
            tau2=a
            T3_5=dicc["m5"]["T"][kk]
            T3_8=dicc["m8"]["T"][kk]
            Pe3_5=dicc["m5"]["Pe"][kk]
            Pe3_8=dicc["m8"]["Pe"][kk]
        else:
            tau2=b
            T3_5=dicc["m5"]["T"][kk-1]
            T3_8=dicc["m8"]["T"][kk-1]
            Pe3_5=dicc["m5"]["Pe"][kk-1]
            Pe3_8=dicc["m8"]["Pe"][kk-1]
        break
for jj in range(kk,len(dicc["m5"]["lgTauR"])):
    tau=10**dicc["m5"]["lgTauR"][jj]    
    if tau>=5.0:
        a=tau
        b=10**dicc["m5"]["lgTauR"][jj-1]
        if abs(a-5)<abs(b-5):
            tau2=a
            T2_5=dicc["m5"]["T"][jj]
            T2_8=dicc["m8"]["T"][jj]
            Pe2_5=dicc["m5"]["Pe"][jj]
            Pe2_8=dicc["m8"]["Pe"][jj]
        else:
            tau2=b
            T2_5=dicc["m5"]["T"][jj-1]
            T2_8=dicc["m8"]["T"][jj-1]
            Pe2_5=dicc["m5"]["Pe"][jj-1]
            Pe2_8=dicc["m8"]["Pe"][jj-1]

        break

#cálculo de poblaciones (niveles de ionización)
Pob1_5=Saha(T1_5,Pe1_5) #en tau=0.5 para el modelo a 5kK
Pob2_5=Saha(T2_5,Pe2_5) #en tau=5.0 para el modelo a 5kK
Pob1_8=Saha(T1_8,Pe1_8) #en tau=0.5 para el modelo a 8kK
Pob2_8=Saha(T2_8,Pe2_8) #en tau=5.0 para el modelo a 8kK
Pob3_5=Saha(T3_5,Pe3_5) #en tau=1.0 para el modelo a 5kK
Pob3_8=Saha(T3_8,Pe3_8) #en tau=1.0 para el modelo a 8kK

#cálculo de poblaciones (niveles de excitación)

#guardamos la población total de HI en cada caso (con la ecuación de cierre)
n1_5=Pob1_5[1]
n2_5=Pob2_5[1]
n1_8=Pob1_8[1]
n2_8=Pob2_8[1]
n3_5=Pob3_5[1]
n3_8=Pob3_8[1]

PobHI1_5=Boltzmann(T1_5)
n1_1_5=n1_5/(PobHI1_5[0]+PobHI1_5[1]+1)
n2_1_5=n1_1_5*PobHI1_5[0]
n3_1_5=n1_1_5*PobHI1_5[1]
PobHI1_5=[n1_1_5,n2_1_5,n3_1_5]

PobHI1_8=Boltzmann(T1_8)
n1_1_8=n1_8/(PobHI1_8[0]+PobHI1_8[1]+1)
n2_1_8=n1_1_8*PobHI1_8[0]
n3_1_8=n1_1_8*PobHI1_8[1]
PobHI1_8=[n1_1_8,n2_1_8,n3_1_8]

PobHI2_5=Boltzmann(T2_5)
n1_2_5=n2_5/(PobHI2_5[0]+PobHI2_5[1]+1)
n2_2_5=n1_2_5*PobHI2_5[0]
n3_2_5=n1_2_5*PobHI2_5[1]
PobHI2_5=[n1_2_5,n2_2_5,n3_2_5]

PobHI2_8=Boltzmann(T2_8)
n1_2_8=n2_8/(PobHI2_8[0]+PobHI2_8[1]+1)
n2_2_8=n1_2_8*PobHI2_8[0]
n3_2_8=n1_2_8*PobHI2_8[1]
PobHI2_8=[n1_2_8,n2_2_8,n3_2_8]

PobHI3_5=Boltzmann(T3_5)
n1_3_5=n3_5/(PobHI3_5[0]+PobHI3_5[1]+1)
n2_3_5=n1_3_5*PobHI3_5[0]
n3_3_5=n1_3_5*PobHI3_5[1]
PobHI3_5=[n1_3_5,n2_3_5,n3_3_5]

PobHI3_8=Boltzmann(T3_8)
n1_3_8=n3_8/(PobHI3_8[0]+PobHI3_8[1]+1)
n2_3_8=n1_3_8*PobHI3_8[0]
n3_3_8=n1_3_8*PobHI3_8[1]
PobHI3_8=[n1_3_8,n2_3_8,n3_3_8]


#guardamos los resultados en un dataframe (5000K)
pob0_5=np.concatenate((Pob1_5,PobHI1_5))
pob5_0=np.concatenate((Pob2_5,PobHI2_5))
pob1_0=np.concatenate((Pob3_5,PobHI3_5))
datapob5=[pob0_5,pob5_0,pob1_0]
df_pob5=pd.DataFrame(datapob5,columns=['H-','HI','HII','Ne','HI, n=1','HI, n=2','HI, n=3',])
name='poblaciones5.csv'
path = os.path.join(ruta,name)
df_pob5.to_csv(path, encoding='utf-8', index=False)

#guardamos los resultados en un dataframe (8000K)
pob0_5=np.concatenate((Pob1_8,PobHI1_8))
pob5_0=np.concatenate((Pob2_8,PobHI2_8))
pob1_0=np.concatenate((Pob3_8,PobHI3_8))
datapob8=[pob0_5,pob5_0,pob1_0]
df_pob8=pd.DataFrame(datapob8,columns=['H-','HI','HII','Ne','HI, n=1','HI, n=2','HI, n=3',])
name='poblaciones8.csv'
path = os.path.join(ruta,name)
df_pob8.to_csv(path, encoding='utf-8', index=False)

######Apartado 5

# Definimos las ctes:
c = 3e18 # Vel de la luz en A/s
h = 4.135667696e-15 # Constante de planck en eV*s
R = 1.0968e-3    # Cte de Rydberg en A-1

#Libre-Libre (HI)
def free_free(T,l,n_e,n_k):
    g_ff = 1+ ((0.3456 / (l*R)**(1/3)) * ((l*kb*T/(h*c)) + (1/2)))  # Factor de Gaunt libre-libre
    sigma_ff = 3.7e8 * (g_ff / (T**(1/2)*(c/l)**3) )    # Sección eficaz en cm2
    k_ff = sigma_ff * n_e * n_k * ( 1 - np.e**( (-h*(c/l)) / (kb*T)) )
    return(k_ff)

#Ligado- Libre (HI)
def bound_free(T,l,n,n_i):
    k_bf=[]
    l0=(n**2)/R
    for ii in range(0,len(l)):
        if l[ii]<=l0:
            g_bf = 1- (0.3456/(l[ii]*R)**(1/3))* ((l[ii]*R/(n**2)) - (1/2)) # Factor de Gaunt ligado-libre
            sigma_bf=2.815e29 * ( g_bf / ( (n**5)*(c/l[ii])**3 ) )   # Sección eficaz en cm2
            k_bf.append(sigma_bf * n_i*(1 - np.e**( (-h*c/l[ii]) / (kb*T) )) )
        else:
            g_bf = (1- (0.3456/(l[ii]*R)**(1/3)))* ((l[ii]*R/(n**2)) - (1/2))
            sigma_bf=0   # nulo pasado el umbral
            k_bf.append(sigma_bf * n_i*(1 - np.e**( (-h*c/l[ii]) / (kb*T) )) )
    return(k_bf)

#Libre-libre (H-)
def free_freeHm(T,Pe,l,n_HI):
    theta= 5040/T
    ltheta=np.log10(theta)
    ll=np.log10(l)
    f0= -2.2763 - 1.6850*ll + 0.76661*(ll**2) - 0.053346*(ll**3)
    f1= 15.2827 - 9.2846*ll + 1.99381*(ll**2) - 0.142631*(ll**3)
    f2 =-197.789 + 190.266*ll - 67.9775*(ll**2) + 10.6913*(ll**3) - 0.625151*(ll**4)
    sigma_ff = (10**-26) * 10**(f0+ f1*ltheta+ f2*ltheta**2)
    k_ff_Hm = Pe*sigma_ff*n_HI
    return(k_ff_Hm)

#Ligado-libre (H-)
def bound_freeHm(T,Pe,l,n_HI,n_Hm):
    l0=16220 #umbral de ionización (H- --> HI)
    theta= 5040/T
    a0=1.99654
    a1=-1.18267e-5
    a2=2.64243e-6
    a3=-4.40524e-10
    a4=3.23992e-14
    a5=-1.39568e-18
    a6=2.78701e-23
    k_bf_Hm=[]
    for ii in range(0,len(l)):
        if l[ii]<=l0:
            sigma_bf=( a0 + (a1*l[ii]) + (a2*(l[ii]**2)) + (a3*(l[ii]**3)) + (a4*(l[ii]**4)) + (a5*(l[ii]**5)) + (a6*(l[ii]**6)) ) *10e-18 
            k_bf_Hm.append(4.158e-10 * sigma_bf * Pe * theta**(5/2) * 10**(0.754*theta) * n_HI)
            #k_bf_Hm=sigma_bf * n_Hm *(1 - np.e**( (-h*c/l[ii]) / (kb*T) ))
        else:
            sigma_bf=0
            k_bf_Hm.append(4.158e-10 * sigma_bf * Pe * theta**(5/2) * 10**(0.754*theta) * n_HI)
    return(k_bf_Hm)

#electron-sacattering
def electron(Ne,l):
    k_e=Ne*6.25e-25*(l/l)
    return(k_e)


#valores de lambda para las tablas (cantos de absorción) en A
#consideramos tres valores, uno para cada estado de excitación n=1,2,3
l1=1**2/R
l2=2**2/R
l3=3**2/R

#delta lambda
dl1=0.001*l1
dl2=0.001*l2
dl3=0.001*l3

l=[l1-dl1,l1+dl1,l2-dl2,l2+dl2,l3-dl3,l3+dl3]
l=np.array(l) #definimos un array con los valores y se lo pasamos a las funciones

#obtenemos los resultados numéricos para la opacidad ff y bf (5000K)
dataHI_ff=free_free(T3_5,l,Pob3_5[3],Pob3_5[2])
dataHI_bf1=bound_free(T3_5,l,1,PobHI3_5[0])
dataHI_bf2=bound_free(T3_5,l,2,PobHI3_5[1])
dataHI_bf3=bound_free(T3_5,l,3,PobHI3_5[2])
dataHm_ff=free_freeHm(T3_5,Pe3_5,l,Pob3_5[1])
dataHm_bf=bound_freeHm(T3_5,Pe3_5,l,Pob3_5[1],Pob3_5[0])
datae=electron(Pob3_5[3],l)

#creamos un dataframe con los resultados
dataop5=[dataHI_ff,dataHI_bf1,dataHI_bf2,dataHI_bf3,dataHm_ff,dataHm_bf,datae]
df_opacidad5=pd.DataFrame(dataop5,columns=l)
name='opacidades5.csv'
path = os.path.join(ruta,name)
df_opacidad5.to_csv(path, encoding='utf-8', index=False)

#obtenemos los resultados numéricos para la opacidad ff y bf (8000K)
dataHI_ff=free_free(T3_8,l,Pob3_8[3],Pob3_8[2])
dataHI_bf1=bound_free(T3_8,l,1,PobHI3_8[0])
dataHI_bf2=bound_free(T3_8,l,2,PobHI3_8[1])
dataHI_bf3=bound_free(T3_8,l,3,PobHI3_8[2])
dataHm_ff=free_freeHm(T3_8,Pe3_8,l,Pob3_8[1])
dataHm_bf=bound_freeHm(T3_8,Pe3_8,l,Pob3_8[1],Pob3_8[0])
datae=electron(Pob3_8[3],l)

#creamos un dataframe con los resultados
dataop8=[dataHI_ff,dataHI_bf1,dataHI_bf2,dataHI_bf3,dataHm_ff,dataHm_bf,datae]
df_opacidad8=pd.DataFrame(dataop8,columns=l)
name='opacidades8.csv'
path = os.path.join(ruta,name)
df_opacidad8.to_csv(path, encoding='utf-8', index=False)

#definimos una serie de longitudes de onda (en A)

#definimos una serie de longitudes de onda (en A)
l=np.linspace(500, 20000,1000)

#obtenemos gráfica para 5000K
sol=free_free(T3_5,l,Pob3_5[3],Pob3_5[2])
sol1=bound_free(T3_5,l,1,PobHI3_5[0])
sol2=bound_free(T3_5,l,2,PobHI3_5[1])
sol3=bound_free(T3_5,l,3,PobHI3_5[2])
solHm=free_freeHm(T3_5,Pe3_5,l,Pob3_5[1])
solHm2=bound_freeHm(T3_5,Pe3_5,l,Pob3_5[1],Pob3_5[0])
sole=electron(Pob3_5[3],l)
total=sol+sol1+sol2+sol3+solHm+solHm2+sole

plt.figure(14)
plt.plot(l,sol,label=r'$\kappa_{\mathrm{ff}}(\mathrm{HI})$',linestyle='dashed')
plt.plot(l,sol1,label=r'$\kappa_{\mathrm{bf}}(\mathrm{HI}), n=1$',linestyle='dashed')
plt.plot(l,sol2,label=r'$\kappa_{\mathrm{bf}}(\mathrm{HI}), n=2$',linestyle='dashed')
plt.plot(l,sol3,label=r'$\kappa_{\mathrm{bf}}(\mathrm{HI}), n=3$',linestyle='dashed')
plt.plot(l,solHm,color='purple',label=r'$\kappa_{\mathrm{ff}}(\mathrm{H}^-)$',linestyle='dotted')
plt.plot(l,solHm2,label=r'$\kappa_{\mathrm{bf}}(\mathrm{H}^-)$',linestyle='dotted')
plt.plot(l,sole,color='navy',label=r'$\kappa_{\mathrm{e}}$',linestyle='dashdot')
plt.plot(l,total,color='black',label=r'$\kappa_{\mathrm{tot}}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda\,\,[\mathrm{\AA}]$')
plt.ylabel(r'$\kappa\,\,[\mathrm{cm}^{-1}]$')
plt.xlim(500,20000)
plt.ylim(10**-14,10**2)
plt.grid(which='both',alpha=0.5)
plt.legend(loc="upper right", ncols=2, shadow=False, title="Leyenda", fancybox=True, fontsize='small')
plt.title(r'Opacidades a $\tau_{\mathrm{Ross}}=1$ para el modelo de 5000K')
name='opacidad5.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

#obtenemos gráfica para 8000K
sol=free_free(T3_8,l,Pob3_8[3],Pob3_8[2])
sol1=bound_free(T3_8,l,1,PobHI3_8[0])
sol2=bound_free(T3_8,l,2,PobHI3_8[1])
sol3=bound_free(T3_8,l,3,PobHI3_8[2])
solHm=free_freeHm(T3_8,Pe3_8,l,Pob3_8[1])
solHm2=bound_freeHm(T3_8,Pe3_8,l,Pob3_8[1],Pob3_8[0])
sole=electron(Pob3_8[3],l)
total=sol+sol1+sol2+sol3+solHm+solHm2+sole

plt.figure(15)
plt.plot(l,sol,label=r'$\kappa_{\mathrm{ff}}(\mathrm{HI})$',linestyle='dashed')
plt.plot(l,sol1,label=r'$\kappa_{\mathrm{bf}}(\mathrm{HI}), n=1$',linestyle='dashed')
plt.plot(l,sol2,label=r'$\kappa_{\mathrm{bf}}(\mathrm{HI}), n=2$',linestyle='dashed')
plt.plot(l,sol3,label=r'$\kappa_{\mathrm{bf}}(\mathrm{HI}), n=3$',linestyle='dashed')
plt.plot(l,solHm,color='purple',label=r'$\kappa_{\mathrm{ff}}(\mathrm{H}^-)$',linestyle='dotted')
plt.plot(l,solHm2,label=r'$\kappa_{\mathrm{bf}}(\mathrm{H}^-)$',linestyle='dotted')
plt.plot(l,sole,color='navy',label=r'$\kappa_{\mathrm{e}}$',linestyle='dashdot')
plt.plot(l,total,color='black',label=r'$\kappa_{\mathrm{tot}}$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda\,\,[\mathrm{\AA}]$')
plt.ylabel(r'$\kappa\,\,[\mathrm{cm}^{-1}]$')
plt.xlim(500,20000)
plt.ylim(10**-12,1)
plt.grid(which='both',alpha=0.5)
plt.legend(loc="best", ncols=2, shadow=False, title="Leyenda", fancybox=True, fontsize='small')
plt.title(r'Opacidades a $\tau_{\mathrm{Ross}}=1$ para el modelo de 8000K')
name='opacidad8.pdf'
path = os.path.join(ruta,name)
plt.savefig(path, dpi=300, transparent=True)

# Ligado-Ligado (lineas de absorción)
m_e = 9.10938e-28 # masa del electron en gramos
ec = 4.80326e-10 # carga del electrón statC
c_cgs= 3e10 # velocidad de la luz en cm/s

# para las líneas de Balmer
def bound_bound_Balmer(n,Pob,T):  # n=0 H-alfa, n=1 H-beta #poblaciones de HI dadas por Boltzmann
    # niveles superior e inferior para cada caso
    if n==0: 
        low=2
        up=3
        n_low=Pob[1]
        n_up=Pob[2]
    if n==1:
        low=2
        up=4
        n_low=Pob[1]
        n_up=0 #suponemos que no hay HI en n=4 
    g_low=2*(low**2) #funciones de partición g=2n^2
    g_up=2*(up**2)    
    g_bb = 0.869-(3/up**2)  # factor de Gaunt para líneas de Balmer
    f = ( 2**5/(3**(3/2)*np.pi) ) * ( g_bb/(low**5*up**3) ) * ( (1/low**2) - (1/up**2) )**(-3) #oscilador
    sigma_bb = f * (np.pi*ec**2) / (m_e * c_cgs)
    k_bb = sigma_bb * ( n_low - n_up * (g_low/g_up) )
    #k_bb= sigma_bb * ( 1 - np.e**(-h*(c/l)/(kb*T)) )
    return(k_bb)

#obtenemos los resultados de Balmer
sol0=bound_bound_Balmer(0, PobHI3_5,T3_5)
sol1=bound_bound_Balmer(1,PobHI3_5,T3_5)
print('El coeficiente de absorción para las líneas de Balmer en el modelo de 5000K es el siguiente', 'Halfa:', sol0, 'Hbeta:', sol1)
sol0=bound_bound_Balmer(0, PobHI3_8,T3_8)
sol1=bound_bound_Balmer(1,PobHI3_8,T3_8)
print('El coeficiente de absorción para las líneas de Balmer en el modelo de 8000K es el siguiente', 'Halfa:', sol0, 'Hbeta:', sol1)

#para las líneas de Lyman
def bound_bound_Lyman(n,Pob,T): # n=0 L-alfa, n=1 L-beta #poblaciones de HI dadas por Boltzmann
    # niveles superior e inferior para cada caso
    if n==0: 
        low=1
        up=2
        n_low=Pob[0]
        n_up=Pob[1]
    if n==1:
        low=1
        up=3
        n_low=Pob[0]
        n_up=Pob[2]
    g_low=2*(low**2) #funciones de partición g=2n^2
    g_up=2*(up**2)    
    g_bb = [0.717,0.765]  # factor de Gaunt para líneas de Lyman (Baker y Menzel 1938)
    f = ( 2**5/(3**(3/2)*np.pi) ) * ( g_bb[n]/(low**5*up**3) ) * ( (1/low**2) - (1/up**2) )**(-3) #oscilador
    sigma_bb = f * (np.pi*ec**2) / (m_e * c_cgs)
    k_bb = sigma_bb * ( n_low - n_up * (g_low/g_up) )
    #k_bb= sigma_bb * ( 1 - np.e**(-h_cgs*(c_cgs/l)/(kb_cgs*T)) )
    return(k_bb)

#obtenemos los resultados de Lyman
sol0=bound_bound_Lyman(0, PobHI3_5,T3_5)
sol1=bound_bound_Lyman(1,PobHI3_5,T3_5)
print('El coeficiente de absorción para las líneas de Lyman en el modelo de 5000K es el siguiente', 'Lalfa:', sol0, 'Lbeta:', sol1)
sol0=bound_bound_Lyman(0, PobHI3_8,T3_8)
sol1=bound_bound_Lyman(1,PobHI3_8,T3_8)
print('El coeficiente de absorción para las líneas de Lyman en el modelo de 8000K es el siguiente', 'Lalfa:', sol0, 'Lbeta:', sol1)