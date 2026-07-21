"""
Module for quick spectral analysis
Based off comibis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
from datetime import datetime
from astropy.io import fits
from astropy.table import Table
from matplotlib.ticker import LogFormatter

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # prevents 3ML from changing plt style
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 14

from .spec_model import *

############### Constants ###############

kev_to_erg = 1.60218e-9  # erg/keV
# dico containing the units, multiplicative factors and energy exponents (ex: F(erg/s/cm2) = keV_to_erg * F**2)
spec_dico = {'RATE':[r'Count s$^{-1}$ keV$^{-1}$',1,0], 'FLUX':[r'ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$',1,0], 'EFLUX':[r'ph cm$^{-2}$ s$^{-1}$',1,1],
             'EEFLUX':[r'keV cm$^{-2}$ s$^{-1}$',1,2], 'ERG':[r'erg cm$^{-2}$ s$^{-1}$',kev_to_erg,2]}
res_dico = {'RES':[0,r'$\sigma$'],'REDCHI2':[1,r'$\chi^2_{red}$']}

############### Classes ###############

class Response:
    ''' import all the useful response info (RMF, ARF, energy bins) into a single object'''

    def __init__(self, resp_dir, rmf_ebound_ext='EBOUNDS', rmf_matrix_ext='MATRIX', rmf_table_name='MATRIX',
                 rmf_name=None, arf_name=None):
        '''energy of each channel of incident/true photons = J = size of ARF
            detected/reconstructed photons of simu = I
        '''
        # RMF
        self.rmf_file_name = f'{resp_dir}/{rmf_name}.fits'
        hdul_rmf = fits.open(f'{self.rmf_file_name}') # original RMF
        self.rmf_ext = hdul_rmf[rmf_matrix_ext].data
        self.rmf_ebd = hdul_rmf[rmf_ebound_ext].data
        # sometimes RMF probability table is stored with varying row sizes
        # implying the actual matrix is triangular
        if isinstance(self.rmf_ext[rmf_table_name], fits.column._VLF):
            _n_J = len(self.rmf_ext['ENERG_LO'])
            _n_I = len(self.rmf_ebd['E_MIN'])
            self.rmf_mat = np.zeros((_n_J, _n_I))
            for _j, _row in enumerate(self.rmf_ext[rmf_table_name]):
                self.rmf_mat[_j, :len(_row)] = _row
        else:
            self.rmf_mat = self.rmf_ext[rmf_table_name]

        self.I = (self.rmf_ebd['E_MAX'] + self.rmf_ebd['E_MIN'])/2
        self.dI = self.rmf_ebd['E_MAX'] - self.rmf_ebd['E_MIN']
        self.J = (self.rmf_ext['ENERG_HI'] + self.rmf_ext['ENERG_LO'])/2
        self.dJ = self.rmf_ext['ENERG_HI'] - self.rmf_ext['ENERG_LO']
        print(f'Imported response with ({len(self.J)}x{len(self.I)}) channels (N_true x N_detected).')
        # ARF (cm2) as a function of channel (J)
        if arf_name is None:
            # for some instruments like SPI, the effective area is included in the RMF
            self.arf_file_name = None
            self.arf = np.ones(shape = len(self.J))
        else:
            self.arf_file_name = f'{resp_dir}/{arf_name}.fits'
            hdul_arf = fits.open(f'{self.arf_file_name}')
            self.arf = hdul_arf[1].data['SPECRESP']

    def make_rbn_mat(self, E_bounds_rbn):
        '''creates a bool matrix that tells if channel I is inside [Ei, Ei+1] of Compton spec'''
        self.rbn_IE_matrix = np.array([(self.I>erbn[0])&(self.I<=erbn[1]) for erbn in E_bounds_rbn]).T
        # return self.rbn_IE_matrix
    
    def plot_rmf(self, vmax=4e-3, cmap='magma', interpolation='none'):
        I_min, I_max, J_min, J_max = self.I[0], self.I[-1], self.J[0], self.J[-1]
        fig, ax= plt.subplots(1,1,figsize=(10,8))
        cb=ax.imshow(self.rmf_mat, extent=[I_min, I_max, J_min, J_max ],
                    origin='lower', aspect= (I_min - I_max)/(J_min - J_max), vmax=vmax, cmap=cmap, interpolation=interpolation)
        ax.set_xlabel('Reconstructed Energy (keV)')
        ax.set_ylabel('True Energy (keV)')
        plt.colorbar(cb)
        return ax
    
    def plot_arf(self, color='k'):
        fig, ax= plt.subplots(1,1,figsize=(8,6))
        ax.step(self.J, self.arf, color=color,where='mid' )
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel(r'Effective area (cm$^2$)')
        return ax


class Spectrum:
    '''
    Spectrum superclass
    '''
    def __init__(self, src_name='', instrument=None):
        self.src_name = src_name
        self.instrument = instrument
    
    def import_spectrum(self, file_path, ext_name='SPECTRUM', rate_col='RATE', rate_err_col='STAT_ERR', spec_sys_error=None):
        self.sys_error=spec_sys_error
        hdul = fits.open(file_path)
        spec = hdul[ext_name].data
        self.header = hdul[ext_name].header
        self.rate = np.array(spec[rate_col], dtype=np.float64)
        self.rate_err = np.array(spec[rate_err_col], dtype=np.float64)
        print(f'Imported {self.src_name} spectrum with {len(self.rate)} channels.')
        # add systematic error quadratically if desired
        if spec_sys_error is not None:
            self.rate_err = np.sqrt(self.rate_err**2 + (spec_sys_error * self.rate)**2)

    def make_fit_ebd(self, e_min_fit, e_max_fit):
        self.E_bds_mask = (self.E >= e_min_fit) & (self.E <= e_max_fit) # bool vector to select E bounds for fit
    
    def select_ebd_list(self, ebds_list):
        ''''
        given a list of energy bounds [[e1, e2], [e3, e4], ...]
        create a mask to select specific channels during fit from OR operation
        '''
        self.E_bds_mask = np.ones_like(self.E) # bool vector to select E bounds for fit
        for (emin, emax) in ebds_list:
            self.E_bds_mask |= (self.E >= emin) & (self.E <= emax) 
        
    def plot_raw_spec(self, logscale=True, figsize=(8,6)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.errorbar(x=self.E, xerr=self.dE/2, y = self.rate, yerr=self.rate_err, fmt='.')
        ax.set_xlabel('E (keV)')
        ax.set_ylabel('Rate (Counts/s/keV)')
        if logscale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        return ax
        
    
class SPISpectrum(Spectrum):
    '''spectrum sub-class for INTEGRAL/SPI
    notably, the response does not need rebinning, as it is done by rmfgen tool
    so resp.dJ = spectrum.dE is always true
    '''
    def import_energy_bands(self, resp):
        '''for SPI, only need to copy energy bounds from the response'''
        assert len(resp.I) == len(self.rate), f"Spectrum size ({len(resp.J)}) and response energies size ({len(self.rate)}) are not matching!"
        self.E = resp.I
        self.dE = resp.dI
        self.chan_num = len(self.E)


class Fit:
    '''Fit class used for spectral fitting procedure based on chi-square stat'''

    def __init__(self,  model_name):
        self.model_name = model_name
        if model_name not in SPEC_MODELS:
            raise ValueError(f"Unknown spectral model: {model_name}. Available models: {list(SPEC_MODELS.keys())}")
        self.model = SPEC_MODELS[model_name]

    def calc_rate(self, resp: Response):
        '''convert model flux spec into rates with the response'''
        model_flux = self.model.calc(resp.J, self.model_parameters) # flux (ph/cm2/s/keV) is always calculated using the RMF energies (J)
        self.rate = ((model_flux * resp.dJ * resp.arf)@resp.rmf_mat / resp.dI) # conversion flux->count/s with instrument response (RMF+ARF) and energy bin size (rmf_dI, rmf_dJ)
        return self.rate
    
    def calc_residual(self, fit_params, spectrum: Spectrum, resp: Response):
        '''return residual array on desired energy bounds, without re-bin
        requires to have matching size for response and spectrum
        '''
        self.model_parameters = fit_params # update model parameter
        return ((spectrum.rate - self.calc_rate(resp)) / spectrum.rate_err)[spectrum.E_bds_mask] # the length of this array will be used for the reduced chi2
    
    def calc_rebin_residual(self, fit_params, spectrum: Spectrum, resp: Response):
        '''return residual array on desired energy bounds, with re-bining to match spectrum and response'''
        self.model_parameters = fit_params
        # compute resp.dI@resp.rbn_IE_matrix / spectrum.dE first and merge with calc_residual?
        self.model_spec_rbn = self.calc_rate(resp) * resp.dI @ resp.rbn_IE_matrix / spectrum.dE # model rate and matrix re-binning operation
        return ((self.model_spec_rbn - spectrum.rate) / spectrum.rate_err)[spectrum.E_bds_mask] # the length of this array will be used for the reduced chi2

    def fit_spec(self, spectrum: Spectrum, resp: Response, lmfit_params, e_min_fit=None, e_max_fit=None):
        '''fit the spectrum with model and response'''
        spectrum.make_fit_ebd(e_min_fit, e_max_fit)
        minner = lm.Minimizer(self.calc_residual, lmfit_params, fcn_args=(spectrum, resp))
        result = minner.minimize(method='leastsq')
        self.make_JE_rebin_mat(resp, spectrum)
        self.make_model_df(result, resp)
        self.make_spec_df(resp, result, spectrum)
        return result, minner

    def make_JE_rebin_mat(self, resp: Response, data: Spectrum, plot=False):
        '''create a re-binning matrix from true photon of RMF (J) to data energies (E)
        useful for plotting models and data together
        '''
        self.rbn_JE_matrix = np.array([(resp.J >= erbn[0]) & (resp.J <= erbn[1]) for erbn in zip(data.E - data.dE/2, data.E + data.dE/2)]).T
        if plot:
            fig, ax=plt.subplots(1,1)
            ax.imshow(self.rbn_JE_matrix, origin='lower', aspect='auto', interpolation='none')

    def make_model_df(self, result, resp: Response):
        '''create df with all model spectrum related info'''
        self.model_parameters = result.params # update params after fit
        self.df_model =  pd.DataFrame({'E':resp.I, 'E_ERR':resp.dI/2, 'RATE':self.calc_rate(resp)})
        self.df_model['FLUX'] = self.model.calc(resp.I, self.model_parameters)
        # self.df_spec['FLUX'] = calc_model_flux(resp.I, self.model_parameters, self.model_name)
    

    def make_spec_df(self, resp: Response, result, spectrum: Spectrum):
        '''
        create df with all data spectrum related info
        compute the flux (in ph/s/cm2/keV) of Compton spec from the count-rate
        this is done by estimating the inverse response: Cm = Fm * R => R^-1 = Fm / Cm, where Fm and Cm are the flux and count-rate of the model
        '''
        self.df_spec = pd.DataFrame({'E':spectrum.E, 'E_ERR':spectrum.dE/2, 'RATE':spectrum.rate, 'RATE_ERR':spectrum.rate_err})
        model_rate = self.rate # predicted count-rate (ct/s/keV)
        model_flux = self.model.calc(resp.J, result.params) * resp.dJ @ self.rbn_JE_matrix / spectrum.dE # predicted flux (ph/s/cm2/keV)
        flux_rate_ratio = model_flux/model_rate #  equivalent to an inverse response (~ ph/ct)
        self.df_spec['FLUX'] = flux_rate_ratio * self.df_spec.RATE # ph/s/cm2/keV
        self.df_spec['FLUX_ERR'] = flux_rate_ratio *  self.df_spec.RATE_ERR 

        # create another df only inside the fit energy bounds
        self.df_spec_fit = self.df_spec[spectrum.E_bds_mask].copy()
        self.df_spec_fit['RES'] = result.residual
        self.df_spec_fit['RES_ERR'] = 1.
        if result.nfree==0:
            print('No degree of freedom in fit!')
            return 0
        else:
            self.df_spec_fit['REDCHI2'] = (result.residual)**2*(result.ndata/result.nfree)
            self.df_spec_fit['REDCHI2_ERR'] = 0.
            return 1
    

    def rebin_spec_fact(self, rebin_fact=5, rbn_type= 'linear'):
        ''' rebin the spectrum by grouping channels to obtain N/rebin_fact channels
            linear: group rebin_fact adjacent, log: group on log scale to have larger grouping towards higher energies
            returns a df with the same unit for RATE (here dE = 2 * E_ERR)
        '''
        if rbn_type=='linear': self.df_model['RBN'] = self.df_model.index//rebin_fact # assign a re-binning number for each channel
        if rbn_type=='log':  self.df_model['RBN'] = np.int64(np.logspace(np.log10(len(self.df_model)//rebin_fact), 0, len(self.df_model)))
        group_rbn = self.df_model.groupby('RBN') # group df by re-binning number
        group_rbn_sum, group_rbn_mean = group_rbn.sum(), group_rbn.mean()
        return pd.DataFrame({'E':group_rbn_mean.E, 'E_ERR':group_rbn_sum.E_ERR, 'RATE':group_rbn_sum.RATE/(2 * group_rbn_sum.E_ERR),
                'FLUX':group_rbn_sum.FLUX/(2 * group_rbn_sum.E_ERR)})
    
    def calc_flux_model(self, e_flux_min, e_flux_max, flux_type='eeuf', N_flux_bin=100, verbose=1):
        '''sum (uf) model over a given energy band to obtain the flux
            can be in 'euf'= ph/cm2/s, or 'eeuf' = erg/cm2/s 
        '''
        flux_E = np.linspace(e_flux_min, e_flux_max, N_flux_bin) # energy bins for integration
        dE = np.diff(flux_E)
        uf = self.model.calc(flux_E, self.model_parameters)[:-1] # ph/cm2/s/keV remove last one for "left rectangle" integral
        euf = kev_to_erg * uf * flux_E[:-1] # erg/cm2/s/keV
        f_dico = {key: np.sum(f * dE) for (key,f) in zip(['euf', 'eeuf'],[uf, euf])} 
        if verbose:
            print('flux in {} - {} keV: {:.2e} erg/cm2/s , {:.2e} ph/cm2/s'.format(e_flux_min, e_flux_max, f_dico['eeuf'],f_dico['euf']))
        return f_dico[flux_type]
    

############### Fit functions ###############

    def plot_fit(self, e_min_fit, e_max_fit, spec_type='RATE', res_type='RES', rebin_fact_model=False, rbn_type='log', yscale=None):
        '''
        spec_type: 'RATE' (cts/s/kev), 'FLUX' (ph/s/cm2/kev), 'EFLUX' (ph/s/cm2), 'EEFLUX' (kev/s/cm2), 'ERG' (erg/s/cm2)

        res_type: 'RES' ( (data-model)/sigma ), 'REDCHI2' (delta chi2)

        rebin_fact_model (bool): allows to use finner energy grid to plot model

        rbn_type: type of re-binning ('log' or 'lin')
        '''
        show_residuals = 1 if res_type else 0 # used both as bool and int
        if rebin_fact_model: df_spec_model = self.rebin_spec_fact(rebin_fact_model, rbn_type)
        else: df_spec_model = self.df_model
        df_spec_model_fit = df_spec_model[(df_spec_model.E>e_min_fit)&(df_spec_model.E<e_max_fit)].copy()
        df_spec_compton_fit = self.df_spec_fit

        spec_type_param = spec_dico[spec_type]
        fig, axes = plt.subplots(1+show_residuals, 1, figsize=(9,7), height_ratios=[2,1]*show_residuals+[1]*(1-show_residuals), squeeze=False) # squeeze=0 makes it an array
        if spec_type=='RATE': 
            axes[0,0].plot('E', spec_type, 'r', label='Best fit', data=df_spec_model_fit)
            axes[0,0].errorbar(x='E', xerr='E_ERR', y=spec_type, yerr=spec_type+'_ERR', fmt='k.', label='Data spectrum', data=df_spec_compton_fit)
        else:
            axes[0,0].plot(df_spec_model_fit.E, spec_type_param[1] * df_spec_model_fit['FLUX'] * (df_spec_model_fit.E**spec_type_param[2]), 'r', label='Best fit')
            axes[0,0].errorbar(x = df_spec_compton_fit.E, xerr = df_spec_compton_fit.E_ERR, \
                            y = spec_type_param[1] * df_spec_compton_fit['FLUX'] * (df_spec_compton_fit['E']**spec_type_param[2]), \
                            yerr = spec_type_param[1] * df_spec_compton_fit['FLUX_ERR'] * (df_spec_compton_fit['E']**spec_type_param[2]),\
                            fmt='k.', label='Data spectrum')
        axes[0,0].set_yscale('log');axes[0,0].legend();axes[0,0].set_ylabel(spec_dico[spec_type][0]);axes[0,0].set_xscale('log')
        if yscale:
            axes[0,0].set_ylim(yscale[0],yscale[1])
        if show_residuals:
            axes[0,0].set_xticklabels([])
            axes[0,0].set_xticklabels([],minor=True) # remove energy label for minor and major ticks
            axes[1,0].errorbar(x='E', xerr='E_ERR', y=res_type, yerr=res_type+'_ERR', fmt='k.', data=df_spec_compton_fit)
            axes[1,0].axhline(res_dico[res_type][0], color='grey', linestyle='--')
            axes[1,0].set_ylabel(res_dico[res_type][1]);axes[1,0].set_xscale('log')
        # to show the energies on (some) minor ticks, and with the full numbers instead of power of 10
        formatter = LogFormatter(labelOnlyBase=False)
        axes[show_residuals,0].xaxis.set_minor_formatter(formatter)
        axes[show_residuals,0].xaxis.set_major_formatter(formatter)
        axes[show_residuals,0].set_xlabel('E (keV)')
        return axes


