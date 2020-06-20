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

Pad = np.zeros((40, 40))

def ident(M, h, v): #where k is a horisontal index, v is a vertical of matrix M 
    x_min = 5 * h
    x_max = 5 * (h+1)
    y_min = 5 * v
    y_max = 5 * (v+1)
    return (x_min, x_max, y_min, y_max)               

def overlap(I, A): #where I is the identification matrix (x_min, x_max, y_min, y_max)
    count = 0
    x = np.linspace(I[0], I[1], num = A)
    y = np.linspace(I[2], I[3], num = A)
    for i in range (0, A):
        for j in range(0, A):
             if sp.sqrt((x[i]-100)**2 + (y[j]-100)**2) <= 68.2 and sp.sqrt((x[i]-100)**2 + (y[j]-100)**2) >= 56.9: 
                 count += 1
    S = (count/A**2)*(25)  #area of the overlapped region
    return S

def probability(L): #L is a random matrix with 40x40 
    A = 25 
    P_array = L
    Area_tot = 0
    for i in range(0, 40):
        for j in range(0, 40):
            Area = overlap(ident(L, i, j), A)
            Area_tot = Area_tot + Area
    for i in range(0, 40):
        for j in range(0, 40):
            Area = overlap(ident(L, i, j), A)
            Prob = Area / Area_tot #This is done so that the total probability is 1
            P_array[i][j] = Prob
    return P_array

def fluctuation(Prob_0): #input - probability matrix 
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
#    return Pois        #output - new probability matrix with weights
   

def distributor(Prob_1, N_p): # N_p is the number of photons; Prob is a 2D array of probabilities 
    OneD_prob = Prob_1.flatten()  #horisontal propagation  
    OneD = np.arange(0, 1600, 1)
    Count = np.random.choice(OneD, p=OneD_prob, size=N_p)
    Results = np.zeros(1600)
    for el in Count:
        Results[el] += 1
    Results = np.reshape(Results, (40, 40))
    return Results


#P0 = probability(Pad)
#f = open('prob_dist_proton5', 'wb')
#pickle.dump(P0, f)
#f.close()
#print('P0 is saved')
f = open('prob_dist_kaon5', 'rb')
P0 = pickle.load(f)
f.close()

'''
#fig, ax = plt.subplots(figsize = (7.4, 6))
#Pr = fluctuation(P0) 
#s = np.arange(0, 40, 1)
#for i in range(len(s)):
#    ax.add_artist(plt.axvline(x = s[i], color = 'lightgrey', linewidth = '0.1'))
#    ax.add_artist(plt.axhline(y = s[i], color = 'lightgrey', linewidth = '0.1'))
#ax.add_artist(plt.pcolormesh(Pr, cmap = 'winter', vmin = 0, vmax = 0.012))
#plt.colorbar(orientation = 'vertical')
#plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40], ['-100', '-75', '-50', '-25', '0', '25', '50', '75', '100']) #later I can change 0 to -100
#plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40], ['-100', '-75', '-50', '-25', '0', '25', '50', '75', '100'])
#plt.xlabel('X coordinate (mm)')
#plt.ylabel('Y coordinate (mm)')
#plt.savefig("Prob_distribution.png", dpi = 250)
'''
N_image = 500
N_photon = 36

for i in range(0, N_image):
    New = fluctuation(P0)
    Sample = distributor(New, N_photon)
    fig, ax = plt.subplots(figsize = (6, 6))
#    fig, ax = plt.subplots(figsize = (7.4, 6))
    s = np.arange(0, 40, 1)
    for j in range(len(s)):
        ax.add_artist(plt.axvline(x = s[j], color = 'k', linewidth = '0.05'))
        ax.add_artist(plt.axhline(y = s[j], color = 'k', linewidth = '0.05'))
    ax.add_artist(plt.pcolormesh(Sample, cmap = 'plasma', vmin = 0, vmax = 5)) #add  edgecolors='k', linewidths=0.05
    plt.axis('off')
    plt.savefig("New_kaon/Test"+str(i)+".png")
#    plt.colorbar(orientation = 'vertical')
#    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40], ['-100', '-75', '-50', '-25', '0', '25', '50', '75', '100']) #later I can change 0 to -100
#    plt.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40], ['-100', '-75', '-50', '-25', '0', '25', '50', '75', '100'])
#    plt.xlabel('X coordinate (mm)')
#    plt.ylabel('Y coordinate (mm)')
#    plt.savefig("Pion_sample2.png", dpi = 250)
    plt.show()

        
        
        
        
        
        
        