import logging
import math
import warnings
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

IJD_start_MJD = 51544

def mjd_to_isot(x):
    '''convert MJD values into plot-able ISO dates'''
    isot=Time(x, format='mjd') # charge MJD dates with astropy
    isot.format='isot' # convert to ISO (YYYY-MM-DD)
    return np.datetime64(isot.value) # convert to plot-able dates

class SPIResult:
    def __init__(self, fit_path, result_file="results.spimodfit.fits", residuals_file="residuals.fits", uplim_proba=.9):
        self.uplim_proba = uplim_proba
        self.fit_path = fit_path
        # result file with count-rates
        self.result_file = result_file
        self.result_path = f"{fit_path}/{result_file}"
        self.hdul_result = fits.open(self.result_path)

        self.ener_df = Table(self.hdul_result["SPI.-EBDS-SET"].data).to_pandas()
        self.sources_by_eb = {}
        self.bkg_by_eb = {}
        self.df_sources = pd.DataFrame()
        self.df_bkg = pd.DataFrame()
        print("Building result data frames...")
        self._build_result_df()

        # residual file (if it exists)
        self.residuals_file= residuals_file
        if residuals_file in os.listdir(fit_path):
            self.residuals_path = f"{fit_path}/{residuals_file}"
            print('Building residual array...')
            self._build_residuals()
        else:
            self.residuals_path = None
            print(f'No residual file ({residuals_file}) found in {fit_path}.')
        

    @staticmethod
    def _strip_value(v):
        if isinstance(v, bytes):
            return v.decode(errors="ignore").strip()
        if isinstance(v, str):
            return v.strip()
        return v

    @staticmethod
    def _safe_scalar_columns(table):
        return [name for name in table.colnames if len(table[name].shape) <= 1]


    def _build_result_df(self):
        """Combine all energy extensions of the result FITS file into source/background DataFrames."""
        n_energy_bins = len(self.ener_df)

        all_sources = []
        all_bkg = []

        for eb in range(n_energy_bins):
            eb_ext = 3 + eb

            # Silence all warnings/messages while reading this extension.
            fits_card_logger = logging.getLogger("astropy.io.fits.card")
            prev_level = fits_card_logger.level
            fits_card_logger.setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ext_data = self.hdul_result[eb_ext].data
                table = Table(ext_data)
                scalar_cols = self._safe_scalar_columns(table)
                df = table[scalar_cols].to_df("pandas")
            fits_card_logger.setLevel(prev_level)

            for col in ["PAR_TYPE", "PAR_ID"]:
                if col in df.columns:
                    df[col] = df[col].apply(self._strip_value)

            df = df.rename(columns={"PAR_ID": "NAME", "TSTART_PTG": "IJD_START", "TSTOP_PTG":"IJD_STOP"})
            # df = df.rename(columns={"PAR_ID": "NAME", "TSTART": "IJD_START", "TSTOP":"IJD_STOP"})
            df['MJD_START'] = df.IJD_START + IJD_start_MJD
            df['MJD_STOP'] = df.IJD_STOP + IJD_start_MJD
            df['ISOT_START'] = df.MJD_START.apply(mjd_to_isot)
            df['ISOT_STOP'] = df.MJD_STOP.apply(mjd_to_isot)
            for date_type in ['IJD', 'MJD', 'ISOT']:
                df[f'{date_type}_ERR'] = (df[f'{date_type}_STOP'] - df[f'{date_type}_START'])/2
                df[f'{date_type}_MID'] = df[f'{date_type}_START'] + df[f'{date_type}_ERR']
            
            df["ENERGY_BIN"] = eb
            df["CHANNEL"] = self.ener_df.loc[eb, "CHANNEL"]
            df["E_MIN"] = self.ener_df.loc[eb, "E_MIN"]
            df["E_MAX"] = self.ener_df.loc[eb, "E_MAX"]
            df['E_ERR'] = (df[f'E_MAX'] - df['E_MIN'])/2
            df['E_MID'] = (df[f'E_MAX'] + df['E_MIN'])/2

            df['uplim'] = (df.FLUX_ML - df.FLUX_ERR_ML < 0.)
            # Gaussian upper-limit approximation
            uplim_factor = stats.norm.ppf(self.uplim_proba)
            df[f'FLUX_UPLIM_{self.uplim_proba:g}'] = df.apply(
                lambda x:x['FLUX_ML'] + uplim_factor * x['FLUX_ERR_ML'], axis=1
                )
            

            df_sources = df[df["PAR_TYPE"] == "Point source"].copy()
            df_bkg = df[df["PAR_TYPE"] == "Background model"].copy()

            self.sources_by_eb[eb] = df_sources
            self.bkg_by_eb[eb] = df_bkg

            if not df_sources.empty:
                all_sources.append(df_sources)
            if not df_bkg.empty:
                all_bkg.append(df_bkg)

        self.df_sources = pd.concat(all_sources, ignore_index=True)
        self.df_bkg = pd.concat(all_bkg, ignore_index=True)

    def _build_residuals(self, res_ext_name= 'SPI.-MAXL-RES', res_col_name='RESIDUE', res_err_col_name='STAT_ERR'):
        '''
        Each row is indexed as [point * Ndet + det] and contains one value per energy bin.
        '''
        self.hdul_residuals = fits.open(self.residuals_path)
        self.res_header= self.hdul_residuals[res_ext_name].header
        self.Ndetectors = self.res_header['DET_NUM']
        self.Nscw = self.res_header['ISOC_NUM']
        self.Npoint = self.res_header['PT_NUM']
        self.res_tab = Table(self.hdul_residuals[res_ext_name].data)

        expected_rows = self.Npoint * self.Ndetectors
        if len(self.res_tab) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} residual rows for {self.Npoint} points and "
                f"{self.Ndetectors} detectors, found {len(self.res_tab)}."
            )
        
        residuals = np.asarray(self.res_tab[res_col_name], dtype=float)
        residual_errors = np.asarray(self.res_tab[res_err_col_name], dtype=float)
        n_energy_bins = residuals.shape[-1]
        self.residuals = residuals.reshape(self.Npoint, self.Ndetectors, n_energy_bins)
        self.residual_errors = residual_errors.reshape(self.Npoint, self.Ndetectors, n_energy_bins)
        self.chi = np.divide(
            self.residuals,
            self.residual_errors,
            out=np.full_like(self.residuals, np.nan),
            where=self.residual_errors != 0,
        )
        # self.chi_per_point_energy = np.nansum(self.chi, axis=1)
        print(f'Shape of residual array {self.chi.shape}')

    def get_sources(self, energy_bin=None):
        if energy_bin is None:
            return self.df_sources.copy()
        return self.sources_by_eb.get(energy_bin, pd.DataFrame()).copy()

    def get_background(self, energy_bin=None):
        if energy_bin is None:
            return self.df_bkg.copy()
        return self.bkg_by_eb.get(energy_bin, pd.DataFrame()).copy()

    def select_pointing(self):
        """
        remove bad pointing based on criteria:
        if 1 det has |chi_dp|>det_thresh (default 7) -> rm point
        else, if std(chi_p) > std_point_thresh (default 2) -> rm point
        """
        pass
    
    def plot_source_lightcurves(
        self,
        energy_bin=0,
        ncols=2,
        date_type="MJD",
        figsize_per_panel=(4.5, 3.2),
        share_dates=False,
        sharey=False,
        show_uplim=True,
        show_rel_flux=False
    ):
        date_type = str(date_type).upper()
        df = self.get_sources(energy_bin=energy_bin)
        if df.empty:
            raise ValueError(f"No source results found for energy bin {energy_bin}.")
            
        src_names = sorted(df["NAME"].dropna().unique())
        nsrc = len(src_names)
        if nsrc == 0:
            raise ValueError("No sources available to plot.")

        x_min = None
        x_max = None
        x_min_plot = None
        x_max_plot = None
        if share_dates:
            x_min = df[f"{date_type}_START"].min()
            x_max = df[f"{date_type}_STOP"].max()
            x_dtype = df[f"{date_type}_MID"].dtype
            if np.issubdtype(x_dtype, np.datetime64):
                span = x_max - x_min
                pad = span * 0.03 if span > pd.Timedelta(0) else pd.Timedelta(days=1)
            else:
                span = x_max - x_min
                pad = span * 0.03 if span > 0 else max(abs(float(x_min)), 1.0) * 0.03
            x_min_plot = x_min - pad
            x_max_plot = x_max + pad

        nrows = math.ceil(nsrc / ncols)
        fig_w = figsize_per_panel[0] * ncols
        fig_h = figsize_per_panel[1] * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=sharey)

        axes = np.atleast_1d(axes).ravel()

        for i, src in enumerate(src_names):
            ax = axes[i]
            sdf = df[df["NAME"] == src].copy().sort_values("IJD_START")
            if show_uplim:
                ax.errorbar(date_type+'_MID', 'FLUX_ML', xerr=date_type+'_ERR', yerr='FLUX_ERR_ML', fmt="ko", markersize=4,
                            data=sdf[~sdf.uplim])
                ax.errorbar(date_type+'_MID', xerr=date_type+'_ERR', y=f'FLUX_UPLIM_{self.uplim_proba:g}', fmt='kv', markersize=6,
                            label=f'Upper-limits ({self.uplim_proba*100:g}%)', data=sdf[sdf.uplim], uplims=True)
                
            else:
                ax.errorbar(date_type+'_MID', 'FLUX_ML', xerr=date_type+'_ERR', yerr='FLUX_ERR_ML', fmt="ko", markersize=3, data=sdf)

            if share_dates:
                    ax.set_xlim(x_min_plot, x_max_plot)

            ax.set_title(str(src))
            if i%ncols ==0: ax.set_ylabel("Flux")
            if i//nrows == nrows-1: ax.set_xlabel(date_type)
            
            if str(date_type).upper() in ["IJD", "MJD"]:
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
                ax.tick_params(axis='x', labelrotation=30)
                for lbl in ax.get_xticklabels():
                    lbl.set_ha('right')

            ax.grid(alpha=0.3)

            if show_rel_flux:
                mean_flux = sdf["FLUX_ML"].mean()
                ax.axhline(mean_flux, color="green", linestyle="--")
                ax2 = ax.twinx()
                ylim = ax.get_ylim()
                ax2.set_ylim((ylim[0] - mean_flux) / mean_flux * 100, (ylim[1] - mean_flux) / mean_flux * 100)
                if i % ncols == ncols - 1 or i == nsrc - 1:
                    ax2.set_ylabel("Rel. dev. from mean (%)")

        for j in range(nsrc, len(axes)):
            axes[j].axis("off")

        e_min = df["E_MIN"].iloc[0]
        e_max = df["E_MAX"].iloc[0]
        fig.suptitle(f"Source Light-Curves -  {e_min:g}-{e_max:g} keV (bin {energy_bin})", y=1.02)
        fig.tight_layout()
        return axes

    def plot_energybin_lightcurves_for_source(
        self,
        source_name,
        ncols=2,
        date_type="MJD",
        figsize_per_panel=(4.5, 3.2),
        energy_bins=None,
        sharey=False, show_uplim=True, show_rel_flux=False
    ):
        """Plot one light curve per energy bin for a single source."""
        sname = str(source_name).strip()
        sdf_all = self.df_sources[self.df_sources["NAME"] == sname].copy()
        if sdf_all.empty:
            raise ValueError(f"No source results found for source '{source_name}'.")
        if energy_bins is None:
            energy_bins = sorted(sdf_all["ENERGY_BIN"].dropna().astype(int).unique())
        neb = len(energy_bins)
        if neb == 0:
            raise ValueError(f"No energy bins found for source '{source_name}'.")

        nrows = math.ceil(neb / ncols)
        fig_w = figsize_per_panel[0] * ncols
        fig_h = figsize_per_panel[1] * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=sharey)
        axes = np.atleast_1d(axes).ravel()

        for i, eb in enumerate(energy_bins):

            ax = axes[i]
            sdf = sdf_all[sdf_all["ENERGY_BIN"] == eb].copy().sort_values("IJD_START")
            e_min = sdf["E_MIN"].iloc[0]
            e_max = sdf["E_MAX"].iloc[0]
            if show_uplim:
                ax.errorbar(date_type+'_MID', xerr=date_type+'_ERR', y='FLUX_ML',  yerr='FLUX_ERR_ML', fmt="ko", markersize=4,
                            data=sdf[~sdf.uplim], label=f"{e_min:g}-{e_max:g} keV (Bin {eb})")
                ax.errorbar(date_type+'_MID', xerr=date_type+'_ERR', y=f'FLUX_UPLIM_{self.uplim_proba:g}', fmt='kv', markersize=6,
                            label=f'Upper-limits ({self.uplim_proba*100:g}%)', data=sdf[sdf.uplim], uplims=True)
                
            else:
                ax.errorbar(date_type+'_MID', 'FLUX_ML', xerr=date_type+'_ERR', yerr='FLUX_ERR_ML', fmt="ko", markersize=3,
                            data=sdf, label=f"{e_min:g}-{e_max:g} keV (Bin {eb})")

            if i % ncols == 0:
                ax.set_ylabel("Flux")
            if i // ncols == nrows - 1:
                ax.set_xlabel(date_type)
                ax.tick_params(axis='x', labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)

            if str(date_type).upper() in ["IJD", "MJD"]:
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
                ax.tick_params(axis='x', labelrotation=30)
                for lbl in ax.get_xticklabels():
                    lbl.set_ha('right')

            ax.grid(alpha=0.3)
            ax.legend(loc='best')

            if show_rel_flux:
                mean_flux = sdf["FLUX_ML"].mean()
                ax.axhline(mean_flux, color="green", linestyle="--")
                ax2 = ax.twinx()
                ylim = ax.get_ylim()
                ax2.set_ylim((ylim[0] - mean_flux) / mean_flux * 100, (ylim[1] - mean_flux) / mean_flux * 100)
                if i % ncols == ncols - 1 or i == neb - 1:
                    ax2.set_ylabel("Rel. dev. from mean (%)")

        for j in range(neb, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Light Curves by Energy Bin - {sname}", y=1.02)
        fig.tight_layout()
        return axes

    def plot_chi_per_point_per_det(self, energy_bins=None, n_sigma=3, ncols=1, figsize=(8, 5),
                                   show_det=False, use_same_y_lim=False, show_hist=False, n_hist_bins=30):
        """Plot chi by energy bin, optionally showing a separate series per detector."""
        if not hasattr(self, "chi"):
            raise ValueError("Residual chi values are unavailable; load a residuals file first.")
        if n_sigma <= 0:
            raise ValueError("n_sigma must be positive.")

        chi_values = self.chi
        if energy_bins is None:
            energy_bins = range(chi_values.shape[2])
        else:
            energy_bins = list(energy_bins)

        show_side_histogram = show_hist and ncols == 1
        ncols = min(ncols, len(energy_bins))
        nrows = math.ceil(len(energy_bins) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize[0], figsize[1] * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        finite_chi = np.abs(chi_values[:, :, list(energy_bins)])
        finite_chi = finite_chi[np.isfinite(finite_chi)]
        global_y_lim = max(float(finite_chi.max()) if finite_chi.size else 0, n_sigma) * 1.05

        Nparam= self.res_header['PARS_FIT']
        for ax, energy_bin in zip(axes, energy_bins):
            e_min = self.ener_df.loc[energy_bin, "E_MIN"]
            e_max = self.ener_df.loc[energy_bin, "E_MAX"]
            flat_chi = chi_values[:, :, energy_bin].reshape(-1)

            if show_side_histogram:
                finite_flat_chi = flat_chi[np.isfinite(flat_chi)]
                hist_ax = make_axes_locatable(ax).append_axes("right", size="30%", pad=0.) # pad=0.08
                histogram, bin_edges, _ = hist_ax.hist(
                    finite_flat_chi, orientation="horizontal", alpha=0.7, label="Residuals", bins=n_hist_bins,
                )
                bin_width = np.diff(bin_edges).mean()
                gaussian_y = np.linspace(bin_edges[0], bin_edges[-1], 200)
                gaussian_counts = stats.norm.pdf(gaussian_y) * finite_flat_chi.size * bin_width
                hist_ax.plot(gaussian_counts, gaussian_y, 'g--', label="N(0, 1)")
                hist_ax.axhline(n_sigma, color="red", linestyle="--") # , label=f"{n_sigma:g}-"+r"$\sigma$"
                hist_ax.axhline(-n_sigma, color="red", linestyle="--")
                hist_ax.axhline(0., color="black", linestyle=":")
                # hist_ax.set_xlabel("Count")
                hist_ax.set_xticks([])
                hist_ax.set_yticks([])
                hist_ax.legend(loc="best")

            # recompute chi2 and reduced chi2
            Ndata = (~np.isnan(flat_chi)).sum()
            Ndof = Ndata - Nparam
            chi2 = np.nansum(self.chi[:,:,energy_bin].flatten()**2)
            chi2_red = chi2/Ndof

            ax.set_title(f"{e_min:g}-{e_max:g} keV (bin {energy_bin})")
            label_plt= rf"$\mu$= {np.nanmean(flat_chi):.2e}, ($\sigma$ - 1)={np.nanstd(flat_chi)-1:.2e}"+"\n"
            label_plt += rf"$\chi^2_r$= {chi2_red:.2f} ({Ndof} dof)"
            if show_det:
                for detector in range(self.Ndetectors):
                    if np.isnan(chi_values[:, detector, :]).all():
                        continue
                    ax.plot(
                        chi_values[:, detector, energy_bin], ".",
                    )
            else:
                ax.plot(
                    flat_chi, ".", label=label_plt,
                )
            ax.axhline(n_sigma, color="red", linestyle="--", label=f"{n_sigma:g}-"+r"$\sigma$") 
            ax.axhline(-n_sigma, color="red", linestyle="--")
            ax.axhline(0., color="black", linestyle=":")
            if use_same_y_lim:
                ax.set_ylim(-global_y_lim, global_y_lim)
                if show_side_histogram:
                    hist_ax.set_ylim(-global_y_lim, global_y_lim)

            ax.set_xlabel("Index (point * Ndet + det)")
            ax.set_ylabel(r"$\Delta \chi$")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")

        for ax in axes[len(energy_bins):]:
            ax.set_visible(False)

        fig.tight_layout()
        return axes



    def close(self):
        self.hdul_result.close()
        