"""
Spectral models for quick spectral analysis
Each model class is registered in SPEC_MODELS dictionary with an associated short-hand name
"""

import numpy as np

kev_to_erg = 1.60218e-9  # erg/keV

class Model:
    '''Base class for spectral models'''
    def calc(self, E, par):
        '''Calculate F given energy array E and parameter dict params'''
        raise NotImplementedError

##### Simple models #####

class Constant(Model):
    '''Constant (K) with energy'''
    def calc(self, E, par):
        return par['K'] * np.ones_like(E)

##### Power-law type models #####

class PowerLaw(Model):
    '''Power-law with index = -gamma)'''
    def calc(self, E, par):
        return par['K'] * E**(-par['gamma'])
    
class PowerLawFlux(Model):
    '''Variant of power-law model, where K is the integrated flux between e1 and e2, in ph/cm2/s'''
    def calc(self, E, par):
        return par['K'] * ((par['gamma']-1) * E**(-par['gamma'])) / (par['e1']**(1-par['gamma']) - par['e2']**(1-par['gamma']))
    
class PowerLawErg(Model):
    '''Variant of power-law model, where K is the integrated flux between e1 and e2, in erg/cm2/s'''
    def calc(self, E, par):
        return (par['K']/kev_to_erg) * (par['gamma'] - 2) * E**(-par['gamma']) / (par['e1']**(2-par['gamma']) - par['e2']**(2-par['gamma']))
    
class BrokenPL(Model):
    '''Power-law with index = -gamma)'''
    def calc(self, E, par):
        lowE, highE = E[E <= par['Eb']], E[E > par['Eb']]
        return np.concatenate((par['K'] * lowE**(-par['gamma1']),\
                               par['K'] * par['Eb']**(par['gamma2'] - par['gamma1']) * highE**(-par['gamma2'])))
    
##### Power-law with cut-off #####
    
class CutOffPL(Model):
    '''Power-law (K, -gamma) with an exponential cut-off (Ecut)'''
    def calc(self, E, par):
        return par['K'] * E**(-par['gamma']) * np.exp(-E/par['Ecut'])
    
class SuperCutOffPL(Model):
    '''Power-law (K, -gamma) with an exponential cut-off (Ecut), with the exponential argument raised to a power (beta)'''
    def calc(self, E, par):
        return par['K'] * E**(-par['gamma']) * np.exp(- (E/par['Ecut'])**par['beta'])
    
##### Other empirical models #####

class GRBM(Model):
    def calc(self, E, par):
        dalpha = par['alpha1'] - par['alpha2']
        lowE, highE = E[E < par['Ec'] * dalpha], E[E >= par['Ec'] * dalpha]
        return np.concatenate((par['K'] * (lowE/100.)**par['alpha1'] * np.exp(-lowE/par['Ec']),\
                               par['K'] * (dalpha * par['Ec']/100.)**dalpha * (highE/100.)**par['alpha2'] * np.exp(-dalpha)))


SPEC_MODELS = {
    'constpol': Constant(),
    'powerlaw':PowerLaw(),
    'cutoffpl': CutOffPL(),
    'plflux': PowerLawFlux(),
    'plerg': PowerLawErg(),
    'bknpl': BrokenPL(),
    'supcutoffpl': SuperCutOffPL(),
    'grbm': GRBM()
}
"""Dictionary mapping model names to model instances"""
