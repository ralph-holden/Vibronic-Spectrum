# -*- coding: utf-8 -*-
"""
Functions for Time Dependant Quantum Mechanics Coursework 2024

Ralph Holden - Imperial College London
"""
# # # IMPORTS # # #
import numpy as np
from numpy.fft import fft,ifft,fftshift,ifftshift,fftfreq

import math as math
from scipy.constants import hbar, c, electron_mass, proton_mass
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.optimize import curve_fit

import pickle

#%pylab qt
from qdyn import propagator, animate_dynamics, kinetic_prop

from numpy.polynomial.hermite import hermval

# reduced units
atu = 0.02419 *10**-15 #s
a0  = 0.5292  *10**-10 #m
m_e = 9.109   *10**-31 #kg
E_h = 4.360   *10**-18 #J

proton_mass_ratio = proton_mass / electron_mass


hbar_reduced = hbar / E_h * atu
c_reduced = c / a0**2 * atu

abs_coeff_real = (atu / E_h) * ( atu / a0**2 ) * ( 1 / atu ) #? 'undo' (reciprocal) reduced hbar, (reciprocal) c and frequency coefficients

# potential parameters
De_g = 0.057169
De_e = 0.019963

delt_g = 1.3907
delt_e = 1.3790

x0_g = 5.038
x0_e = 5.715

Te_g = 0.1
Te_e = 0.1718
delta_Te = Te_e - Te_g

# functions: finding eigenvalues & gwp
def eigen_ho(x, v, m, k):
    """Calculates the eigenfunction of the harmonic oscillator system.
    Arguments
    x: is a space coordinate.
    v: is the vibrational quantum number.
    m: is the mas of the system.
    k: is the force constant of the harmonic potential.
    """
    hermite_sum=np.zeros(v+1)
    hermite_sum[-1]=1
    return 1/(2**v * math.factorial(v))**0.5 * (((m*k)**0.5)/np.pi)**0.25 * np.e**(-x**2 * ((m*k)**0.5)/2) * hermval((m*k)**0.25 * x,hermite_sum)

def gwp(x,xc,pc,alpha,gamma):
    """Gaussian wavepacket in one dimension."""
    return (2/np.pi*np.real(alpha))**0.25 * np.exp(-alpha*(x-xc)**2 + 1j*pc*(x-xc) + 1j*gamma)

# functions: dipole, ground state potential, excited state potential
def V_flat(x):
    "A flat potential"
    return 0*x

def V_step(x,pos,height):
    "Step potential"
    result=np.zeros(len(x))
    result[x>pos]=height
    return result

def V_harm(x,x0,k):
    "Harmonic oscillator"
    return k/2*(x-x0)**2

def V_rydberg(x):
    '''Only need rydberg for the excited state'''
    De = De_e
    delt = delt_e
    x0 = x0_e
    return De * (1 - (1 + delt * (x - x0)) * np.e**( -1*delt * (x - x0) )) + Te_e

def V_rydberg_g(x):
    '''For trialling rydberg of ground state'''
    De = De_g
    delt = delt_g
    x0 = x0_g
    return De * (1 - (1 + delt * (x - x0)) * np.e**( -1*delt * (x - x0) )) + Te_g

def tdp(x):
    '''Transition Dipole Moment'''
    return 18.4/x**2 * np.e**(-0.304*(x-6.49)**2)

def tdp_shifted(x):
    '''Transition Dipole Moment - SHIFTED for I-127'''
    return 18.4/(x+x0_g)**2 * np.e**(-0.304*((x+x0_g)-6.49)**2)    

def g(x):
    '''Ground state potential'''
    k_g = De_g * delt_g**2
    return V_harm(x, x0_g, k_g)
    
def e(x):
    '''Excited state potential'''
    k_e = De_e * delt_e**2
    return V_harm(x, x0_e, k_e)      

# wavepacket class
class Wavepkt:
    # reduced units
    atu = 0.02419 *10**-15 #s
    a0  = 0.5292  *10**-10 #m
    m_e = 9.109   *10**-31 #kg
    E_h = 4.360   *10**-18 #J

    # potential parameters
    De_g = 0.057169
    De_e = 0.019963

    delt_g = 1.3907
    delt_e = 1.3790

    x0_g = 5.038
    x0_e = 5.715
    
    Te_g = 0.1
    Te_e = 0.1718
    Te = Te_e - Te_g
    
    k_g = De_g * delt_g**2
    k_e = De_e * delt_e**2
    
    def __init__(self, timestep, nsteps, xmin, xmax, spacesteps, m = 1, estate='ground'):
        self.dt =              timestep
        self.nsteps =          nsteps
        self.m =               m
        self.xmin =            xmin
        self.xmax =            xmax
        self.spacesteps =      spacesteps
        self.x_grid =          np.linspace(xmin,xmax,spacesteps)
        
        self.wf_dynamics =     np.zeros((self.nsteps+1,len(self.x_grid)),dtype=np.complex128)
        self.frequency =       2*np.pi*fftshift(fftfreq(self.nsteps+1))
        self.autocorrelation = []
        self.gauss_w =         None
        self.absorption =       None
        self.fpeaks =          []
        self.frequency_absorption = None
        
        if estate == 'ground':
            self.De = self.De_g
            self.delt = self.delt_g
            self.x0 = self.x0_g 
            self.k = self.k_g
        elif estate == 'excited':
            self.De = self.De_e
            self.delt = self.delt_e
            self.x0 = self.x0_e
            self.k = self.k_e
    
    # wf initialisation 
    def initialise_wf_eigen(self,vlevels_list):
        '''initialise the wf with gaussian wp'''
        for i in vlevels_list:
            self.wf_dynamics[0] += eigen_ho(self.x_grid-self.x0, v=i, m=self.m, k=self.k)
    
    def initialise_wf_gwp(self,gwp_list):
        '''Initialise the wf with gwp'''
        xc =    gwp_list[0] 
        pc =    gwp_list[1]
        alpha = gwp_list[2]
        gamma = gwp_list[3]    
        self.wf_dynamics[0] += gwp(self.x_grid, xc , pc , alpha , gamma)

    # wf propagation
    def propagate_wf(self):
        for step in range(self.nsteps):
            psi = propagator(self.x_grid, self.wf_dynamics[step], self.m, self.dt, V_harm, self.x0, self.k)
            self.wf_dynamics[step+1] = psi
            
    def propagate_other(self, other_x0, other_k):
        for step in range(self.nsteps):
            psi = propagator(self.x_grid, self.wf_dynamics[step], self.m, self.dt, V_harm, other_x0, other_k)
            self.wf_dynamics[step+1] = psi
          
    def propagate_rydberg(self):
        for step in range(self.nsteps):
            psi = propagator(self.x_grid, self.wf_dynamics[step], self.m, self.dt, V_rydberg)
            self.wf_dynamics[step+1] = psi
            
    def propagate_rydberg_moresteps(self,new_steps):
        current_len = len(self.wf_dynamics)
        for step in range(current_len, current_len + new_steps):
            psi = propagator(self.x_grid, self.wf_dynamics[step], self.m, self.dt, V_rydberg)
            self.wf_dynamics[step+1] = psi
            
    def propagate_rydberg_tdp(self):
        '''Multiplies by transition dipole moment FOR excitation, rather than just for autocorrelation in powerseries_lp
        Need to use with powerseries function, NOT powerseries_lp'''
        self.wf_dynamics[0] = self.wf_dynamics[0] * tdp(self.x_grid)
        for step in range(self.nsteps):
            psi = propagator(self.x_grid, self.wf_dynamics[step], self.m, self.dt, V_rydberg)
            self.wf_dynamics[step+1] = psi
      
    def propagate_rydberg_tdp_g(self):
        '''Multiplies by transition dipole moment FOR excitation, rather than just for autocorrelation in powerseries_lp
        Need to use with powerseries function, NOT powerseries_lp'''
        self.wf_dynamics[0] = self.wf_dynamics[0] * tdp(self.x_grid)
        for step in range(self.nsteps):
            psi = propagator(self.x_grid, self.wf_dynamics[step], self.m, self.dt, V_rydberg_g)
            self.wf_dynamics[step+1] = psi
      
    # functions that break up dynamics analysis
    def calc_autocorrelation(self):
        for i in np.conj(self.wf_dynamics[0])*self.wf_dynamics: 
            yprov = np.trapz(i,self.x_grid)
            self.autocorrelation.append(yprov)
            
    def calc_byparts_autocorrelation(self, div = 8):
        '''Calculating the autocorrellation function by parts, to reduced the size of any one array'''
        whole = len(self.wf_dynamics) - 1
        for num in range(div):
            if num == 0:
                part = self.wf_dynamics[ 0 : int( whole/div * (num + 1) ) + 1 ]
            else:
                part  = self.wf_dynamics[ int( whole/div * num ) + 1 : int( whole/div * (num + 1) ) + 1 ]
            for i in np.conj(self.wf_dynamics[0]) * part: 
                yprov = np.trapz(i,self.x_grid)
                self.autocorrelation.append(yprov)
            
    def plot_autocorrelation(self):
        plt.figure(figsize = [8,5])
        plt.title('Autocorrelation')
        plt.xlabel('Time')
        plt.ylabel('Weighting')
        plt.grid(linestyle=':')
        #plt.xlim(0)
        plt.plot(np.linspace(0,self.dt*self.nsteps,len(self.autocorrelation)) , self.autocorrelation)
        plt.show()
    
    def calc_powerseries(self):
        # Obtain the inverse Fourier transform of gauss_t. Note that the values are shifted by fftshit()
        self.gauss_w=ifftshift(ifft(self.autocorrelation,norm="ortho"))
    
    def plot_powerseries(self):
        plt.figure(figsize = [8,5])
        plt.title('Power Series')
        plt.xlabel('Frequency')
        plt.ylabel('Weighting')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.gauss_w))
        plt.show()
        
    def calc_absorption(self, E_zeropoint = 0.10053029471290863):
        self.frequency_absorption = self.frequency - E_zeropoint
        self.absorption = (self.gauss_w) * self.frequency_absorption
        
    def plot_absorption(self):
        plt.figure(figsize = [8,5])
        plt.title('Absorption Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Intensity')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.absorption))
        plt.show()
        
    def plot_absorption_real(self, cut = 103200):
        wavelength = 1/(self.frequency_absorption * 219474.6) * 10**7
        plt.figure(figsize = [8,5])
        plt.title('Absorption Spectrum')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.grid(linestyle=':')
        plt.plot(wavelength[cut:], abs(self.absorption[cut:]))
        plt.show()
        
    def plot_all(self):
        plt.figure(figsize = [8,15])
        
        plt.subplot(3, 1, 1)
        plt.title('Autocorrelation')
        plt.xlabel('Time')
        plt.ylabel('Weighting')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.autocorrelation))
        
        plt.subplot(3, 1, 2)
        plt.title('Power Series')
        plt.xlabel('Frequency')
        plt.ylabel('Weighting')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.gauss_w)) 
        
        plt.subplot(3, 1, 3)
        plt.title('Absorption Spectrum')
        plt.xlabel('Frequency')
        plt.ylabel('Intensity')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.absorption)) 
        
        plt.tight_layout(pad = 2)
        plt.show()
    
    # functions that do it all together
    def powerseries(self):
        # find the autocorrelation function, array progressing through time
        y = []
        for i in np.conj(self.wf_dynamics[0])*self.wf_dynamics: 
            yprov = np.trapz(i,self.x_grid)
            y.append(yprov)
        # Obtain the inverse Fourier transform of gauss_t. Note that the values are shifted by fftshit()
        self.gauss_w=ifftshift(ifft(y,norm="ortho")) 
        # plot!
        plt.figure(figsize = [8,5])
        plt.title('Power Series')
        plt.xlabel('Frequency')
        plt.ylabel('Weighting')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.gauss_w)) # Plot the transformed function
        plt.show()
        
    def powerseries_lp(self):
        '''Light perturbated power series
        If H_2 is time dependant only, 
        autocorrellation function between wf (over all time) tdp product can be used 
        '''
        # find the autocorrelation function, array progressing through time, ALL multiplied by tdp
        y = []
        for i in np.conj(self.wf_dynamics[0]*tdp(self.x_grid))*(self.wf_dynamics*tdp(self.x_grid)): 
            yprov = np.trapz(i,self.x_grid)
            y.append(yprov)
        self.gauss_w=ifftshift(ifft(y,norm="ortho")) 
        # plot!
        plt.figure(figsize = [8,5])
        plt.title('Light Pertibated Power Series')
        plt.xlabel('Frequency')
        plt.ylabel('Weighting')
        plt.grid(linestyle=':')
        plt.plot(self.frequency,abs(self.gauss_w)) # Plot the transformed function
        plt.show()
      
    # finding peak positions
    def find_peak(self, fmin, fmax, name = 'peak' ):
        '''Gives the frequency of the power spectrum peak in a given range
        '''
        
        selection = np.logical_and(self.frequency > fmin, self.frequency < fmax) 
        f_range = self.frequency[selection]
        gauss_range = self.gauss_w[selection] 
       
        gpeak = np.max(gauss_range)
        fpeak = f_range[gauss_range == gpeak]
        
        self.fpeaks.append( [ name+'_{}'.format( len(self.fpeaks) ), fpeak[0] ] )
      
    # loading and saving data
    def save(self, filename):
        with open(filename+'.dat','wb') as data_f:
            pickle.dump([self.frequency, self.gauss_w], data_f)
            
    def load(self, filename):
        with open(filename+'.dat','rb') as data_f:
            loaded_data = pickle.load(data_f)
            self.frequency = loaded_data[0]
            self.gauss_w = loaded_data[1]
    



