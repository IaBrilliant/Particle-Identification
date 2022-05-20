#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:46:30 2020

@author: brilliant
"""
import matplotlib.pyplot as plt
import scipy as sp 
from scipy.stats import poisson 
import numpy as np 
import pickle
from tqdm import tqdm, trange

#Protons: 42.4 mm + 8.5 mm
#Kaons: 53.2 mm + 10.6 mm
#Pions: 56.9 mm + 11.3 mm

Pad = np.zeros((40, 40))

def ident(h, v): 
    """Setting the coordinate system i.e. the identification matrix I"""
    x_min = 5 * h
    x_max = 5 * (h+1)
    y_min = 5 * v
    y_max = 5 * (v+1)
    return (x_min, x_max, y_min, y_max)               

def overlap(I, A): 
    """Estimating the area of the overlapped region"""
    count = 0
    x = np.linspace(I[0], I[1], num = A)
    y = np.linspace(I[2], I[3], num = A)
    for i in range (0, A):
        for j in range(0, A): #Shapes vary between Kaon and Pion mesons
           # Data below for Pions
             if sp.sqrt((x[i]-100)**2 + (y[j]-100)**2) <= 68.2 \
             and sp.sqrt((x[i]-100)**2 + (y[j]-100)**2) >= 56.9: 
                 count += 1
    S = (count/A**2)*(25)  
    return S

def probability(L):
    """Calculating a uniform probability circle"""
    A = 25 
    P_array = L
    Area_tot = 0
    for i in tqdm(range(40), desc = "Calculating Area"):
#    for i in range(0, 40):
        for j in range(0, 40):
            Area = overlap(ident(L, i, j), A)
            Area_tot = Area_tot + Area
    for i in tqdm(range(40), desc = "Converting to Probability"):
#    for i in range(0, 40):
        for j in range(0, 40):
            Area = overlap(ident(L, i, j), A)
            Prob = Area / Area_tot #Normalising
            P_array[i][j] = Prob
    return P_array

def fluctuation(Prob_0):
    """Introducing Poisson fluctuations to the spatial probality distribution"""
    Pois = Prob_0 * 1000
    S = 0
    for i in range(0, 40):
        for j in range(0, 40):
            Pois[i][j] = round(Pois[i][j])
            r = poisson.rvs(Pois[i][j], size = 1)
            Pois[i][j] = r[0]
            S = S + r[0]
    New_Prob = Pois / S
    return New_Prob
   
def distributor(Prob_1, N_p): 
    """Generates photons to hit the detector"""
    OneD_prob = Prob_1.flatten()  
    OneD = np.arange(0, 1600, 1)
    Count = np.random.choice(OneD, p=OneD_prob, size=N_p)
    Results = np.zeros(1600)
    for el in Count:
        Results[el] += 1
    Results = np.reshape(Results, (40, 40))
    return Results  


P0 = probability(Pad)
f = open('prob_dist_pion5(test)', 'wb')
pickle.dump(P0, f)
f.close()
print('P0 is saved')

f = open('prob_dist_pion5(test)', 'rb')
P0 = pickle.load(f)
f.close()

N_image = 1 #Number of samples to generate
N_photon = 36 #Number of photons hitting the detector

for i in range(0, N_image):
    New = fluctuation(P0)
    Sample = distributor(New, N_photon)
    fig, ax = plt.subplots(figsize = (7.4, 6))
    s = np.arange(0, 40, 1)
    for j in range(len(s)):
        ax.add_artist(plt.axvline(x = s[j], color = 'white', linewidth = '0.05'))
        ax.add_artist(plt.axhline(y = s[j], color = 'white', linewidth = '0.05'))
    ax.add_artist(plt.pcolormesh(Sample, cmap = 'plasma', vmin = 0, vmax = 5)) 
#   plt.savefig("New_kaon/Test"+str(i)+".png")
    plt.colorbar(orientation = 'vertical')
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40], ['-100', '-75', '-50', '-25', '0', '25', '50', '75', '100']) 
    plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40], ['-100', '-75', '-50', '-25', '0', '25', '50', '75', '100'])
    plt.xlabel('X coordinate (mm)')
    plt.ylabel('Y coordinate (mm)')
#    plt.savefig("Kaon.png", dpi = 250)
    plt.show()

        
        
        
        
        
        
        
