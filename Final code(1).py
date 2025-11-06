#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:41:05 2024

@author: liamheylin
"""




import math as m
import numpy as np
import matplotlib.pyplot as plt



# Define inputs
S	= 60
K	= 50
H1 = 65
H2 = 75
r	= 0.05
T	= 1
sigma	= 0.3
n = 260
nr = 200000
C1 = 0
C2 = 0





# --- Calculation Other Values   ----------------------------------------------
# Here we refer to the risk-neutral valuation.
nu = r - 0.5 * sigma**2
dt = T / n              # <- h

# --- Define Result Matrix    -------------------------------------------------

Thetic_Antithetic_payoffs = np.zeros(( nr, 1))
S_val = np.zeros((nr, n+1))
rand = np.random.randn(int(nr/2), n)


    

# Generate the Thetic and Antithetic share paths   
S_val[:,0] = S
for i in range(0, nr, 2):
    for j in range(1,n+1):
        S_val[i,j] = S_val[i,j-1] * m.exp(nu*dt + sigma * dt**0.5 * rand[int(i/2),j-1])
        S_val[i+1,j] = S_val[i+1,j-1] * m.exp(nu*dt - sigma * dt**0.5 * rand[int(i/2),j-1])
        



            



#Thetic,Antithetic payoffs
for i in range(nr):
    
    St = S_val[i,-1]
    C1 = np.sum(S_val[i, :] > H1)
    C2 = np.sum(S_val[i, :] > H2)
    AverageC = (C1 + C2)/2

    
 
    if St > K and C2 >= 150:
        Thetic_Antithetic_payoffs[i] = (St - K + 30)
    elif St > K and C1 >= 100 and C2 < 150 and AverageC < 125:
        Thetic_Antithetic_payoffs[i] = (St - K + 10)
    elif St > K and C1 >= 100 and C2 < 150:
        Thetic_Antithetic_payoffs[i] = (St - K)
    elif St > K and C1 < 100:
        Thetic_Antithetic_payoffs[i] = 10
    elif St <= K:
        Thetic_Antithetic_payoffs[i] = 0
        

        
    

        

#Average of thetic and antithetic payoffs and control variates
FinPayOff = np.zeros(( int(nr/2), 1))
for i in range(0,nr,2):
    FinPayOff[int(i/2)] = 0.5*(Thetic_Antithetic_payoffs[i] + Thetic_Antithetic_payoffs[i+1])
    
    
    


#Discounting payoffs
PDisc = np.exp( -r * T ) * FinPayOff 
price = PDisc.mean()
std = np.std(PDisc)



ExcerciseCount = 0
for i in range(0,nr):
    if  Thetic_Antithetic_payoffs[i] > 0:
        ExcerciseCount += 1


# Calculate the 95% confidence interval
confidence_interval = 1.96 * std / (nr ** 0.5)

# Added this in to check if our expected mean was the same    
print('The price of the Monte Carlo Method with', nr, 'simulations is %.5f.' % price)
print('The standard deviation is %.5f.' % std) 


# Print the 95% confidence interval
print('95% Confidence interval :',(round(price - confidence_interval,5), round(price + confidence_interval,5)))
print('The option is exercised',round(ExcerciseCount/nr*100,2),'% of the time')







# Plot the distribution of FinPayOff
plt.figure(figsize=(8, 6))
plt.hist(FinPayOff, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of FinPayOff')
plt.xlabel('FinPayOff')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


