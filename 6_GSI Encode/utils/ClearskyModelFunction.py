import numpy as np
import pandas as pd

def HLJ (data,elev):
    theta = np.radians(data['SZA'])
    Ehi = data['EtHI']
    
    k0 = 0.4327 - 0.00821*(((6000 - elev)/1000)**2)
    k1 = 0.5055 + 0.00595*(((6500 - elev)/1000)**2)
    k2 = 0.2711 + 0.01858*(((2500 - elev)/1000)**2)
  
    DNI = (Ehi/np.cos(theta))*(k0 + k1*np.exp(-k2/np.cos(theta)))
    DHI = Ehi*0.2710 - 0.2939*DNI*np.cos(theta)
    GHI = DNI*np.cos(theta) + DHI
    
    return GHI,DNI,DHI


def FR(data,elev):
    theta = np.radians(data['SZA'])
    Ehi = data['EtHI']

    k = np.exp(((-0.000118*elev) - (1.638**-9)*(elev**2))/np.cos(theta))
    DNI = Ehi*(0.5**k)
    DHI = 0.43*DNI*np.cos(theta)
    GHI = DNI*np.cos(theta) + DHI
    
    return GHI,DNI,DHI


def DPP(data):
    
    theta = data['SZA']
    
    B0 = 950*(1 - np.exp(-0.075*(90 - theta))) * np.cos(np.radians(theta))
    DHI = 14.29 + 21.04*((np.pi/2) - np.radians(theta))
    GHI = B0 + DHI
    DNI = B0/np.cos(np.radians(theta))
 
    return GHI,DNI,DHI              


def BD(data):
    
    Ehi = data['EtHI']
    GHI = 0.7*Ehi

    return GHI


def KC(data):
    
    theta = np.radians(data['SZA'])
    
    GHI = 910*np.cos(theta) - 30

    return GHI


def RS(data):
    
    theta = data['SZA']
    
    GHI = 1159*(np.cos(np.radians(theta))**1.179)*np.exp(-0.0019*(90 - theta))

    return GHI


def BD14(data):
    
    theta = np.radians(data['SZA'])
    
    GHI = 1007.31*np. cos(theta) - 74.09
    
    return GHI


def BR(data):
    
    theta = np.radians(data['SZA'])
    
    DNI = 926*(np.cos(theta)**0.29)
    DHI = 131*(np.cos(theta)**0.6) 
    GHI = DNI*np.cos(theta) + DHI
    
    return GHI,DNI,DHI


def Sc(data):
    
    theta = np.radians(data['SZA'])
    Ehi = data['EtHI']
    
    DNI = 1127*(0.888)**(1/np.cos(theta))
    DHI = 94.23*(np.cos(theta)**0.5)
    GHI = DNI*np.cos(theta) + DHI
    
    return GHI,DNI,DHI


def SP(data):  
    
    theta = np.radians(data['SZA'])
    Ehi = data['EtHI']
    
    DNI = 1.842*((Ehi/np.cos(theta))/2)*np.cos(theta)/(0.3135 + np.cos(theta))
    GHI = 4.5*((Ehi/np.cos(theta))/120) + 1.071*DNI*np.cos(theta)
    DHI = GHI - DNI*np.cos(theta)
    
    return GHI,DNI,DHI

def HS(data): 
    
    theta = np.radians(data['SZA'])
    Ehi = data['EtHI']
    
    DNI = 1098*np.exp(-0.057/np.cos(theta))
    DHI = 94.23*(np.cos(theta)**0.5)
    GHI = DNI*np.cos(theta) + DHI
  
    return GHI,DNI,DHI