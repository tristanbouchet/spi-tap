import logging
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

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
    def __init__(self, fit_path, result_file="results.spimodfit.fits", uplim_proba=.9):
        self.fit_path = fit_path
        self.result_file = result_file
        self.uplim_proba = uplim_proba
        self.result_path = f"{fit_path}/{result_file}"
        self.hdul = fits.open(self.result_path)

        self.ener_df = Table(self.hdul["SPI.-EBDS-SET"].data).to_pandas()
        self.sources_by_eb = {}
        self.bkg_by_eb = {}
        self.df_sources = pd.DataFrame()
        self.df_bkg = pd.DataFrame()
        print("Building result data frames...")
        self._build_dataframes()

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

    def _build_dataframes(self):
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
                ext_data = self.hdul[eb_ext].data
                table = Table(ext_data)
                scalar_cols = self._safe_scalar_columns(table)
                df = table[scalar_cols].to_df("pandas")
            fits_card_logger.setLevel(prev_level)

            for col in ["PAR_TYPE", "PAR_ID"]:
                if col in df.columns:
                    df[col] = df[col].apply(self._strip_value)

            df = df.rename(columns={"PAR_ID": "NAME", "TSTART": "IJD_START", "TSTOP":"IJD_STOP"})
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

    def get_sources(self, energy_bin=None):
        if energy_bin is None:
            return self.df_sources.copy()
        return self.sources_by_eb.get(energy_bin, pd.DataFrame()).copy()

    def get_background(self, energy_bin=None):
        if energy_bin is None:
            return self.df_bkg.copy()
        return self.bkg_by_eb.get(energy_bin, pd.DataFrame()).copy()

    
    def plot_source_lightcurves(
        self,
        energy_bin=0,
        ncols=2,
        date_type="MJD",
        figsize_per_panel=(4.5, 3.2),
        share_dates=False,
        sharey=False,
        show_uplim=True
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
        sharey=False,
        show_uplim=True
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

        for j in range(neb, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Light Curves by Energy Bin - {sname}", y=1.02)
        fig.tight_layout()
        return axes

    def close(self):
        self.hdul.close()
        