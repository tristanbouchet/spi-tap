"""
SPI-TAP — SPI Transient Analysis Pipeline
Runs the entire SPI pipeline for a point source, given dates, energy bins and source variability.
Can be called directly for quick interactive session, or imported for automatized analysis.

Note that ALL the absolute paths to config files or commands are contained in: config.txt

Main analysis steps:
- select data with dates, position and angle
- prepare data with spiselectscw and energy bounds
- create background model with obs_background.py module
- select sources in FOV from catalog
- select variability parameters for sources and bkg
- run spimodfit and generate spectral response

TO DO:
method to import run parameters (dates, energies, etc...) from txt file
"""

print('Loading spi_obs module...\n')

import pandas as pd
import numpy as np
import sys
import os
import subprocess
import shutil
import time
import glob
import pickle
import socket
from datetime import datetime

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from .obs_background import ScwTracerDB, ObsBkg, LiveTimeRev

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
WHITE = "\033[37m"
RESET = "\033[0m"
BOLD = "\033[1m"

IJD_start_MJD = 51544

def mjd_to_isot(x):
    '''convert MJD values into plot-able ISO dates'''
    isot=Time(x, format='mjd') # charge MJD dates with astropy
    isot.format='isot' # convert to ISO (YYYY-MM-DD)
    return np.datetime64(isot.value) # convert to plot-able dates

class ObsSPI:
    """Pipeline for SPI observation analysis"""
    
    EVT_BIN_SIZE = {'SE':.5, 'PSD':.5, 'HE':1.}
    
    def __init__(self, main_dir, initial_dir='.', initial_env=None, config_file='config.txt',
                 gnrl_cat_ext = 'GNRL-REFR-CAT', bg_idx_filename = 'output_bgmodel_conti_sep_idx.fits.gz',
                 spiselect_par_tpl_file = 'spiselectscw.template.par', spimodfit_par_tpl_file='spimodfit.template.par',
                 spimodfit_result_file = 'results.spimodfit.fits',
                 testrun=False):
        
        # Run attributes
        self.testrun = testrun
        self.full_name = None
        self.ra = None
        self.dec = None
        self.lat_src_coord = None
        self.long_src_coord = None
        # date
        self.date_start = None
        self.date_end = None
        self.off_angle = None
        self.date_dir = None
        # date selection result
        self.df_select = None
        self.unique_revs = None
        self.N_point_select = None
        self.N_unique_revs = None
        # energies
        self.evt_type = None
        self.binning_type = None
        self.e_channels = None
        self.e_bin_size = None
        self.e_channels_bounds = None
        self.e_channels_scales = None
        self.Nchan = None
        self.ener_dir = None

        # Path and directories
        self.main_dir = main_dir
        self.initial_dir = initial_dir
        self.initial_env = initial_env
        self.data_dir = None
        self.scw_db_path = None
        self.gnrl_cat_path = None
        self.spi_cat_path = None
        self.all_revs_path = None
        self.bkg_db_dir = None
        self.import_path_config(config_file)
        # TO DO: need to figure out a better way to access template files...
        self.templates_dir = f'{initial_dir}/templates_par'
        # default file names
        self.gnrl_cat_ext =  gnrl_cat_ext
        self.bg_idx_filename = bg_idx_filename
        self.spiselect_par_tpl_file =  spiselect_par_tpl_file
        self.spimodfit_par_tpl_file = spimodfit_par_tpl_file
        self.spimodfit_result_file = spimodfit_result_file
        
        self.scw_tracer_db = ScwTracerDB(self.scw_db_path)
        self.obs_bkg = None
        self.bkg_dict = None

        # import all revs perigee times
        self.import_rev_nodes()

        # default query values
        self.update_default_dico = True
        self.initial_default_query={
            'src_dir' : '',
            'full_name' : 'Crab',
            'date_start' : '2003-01-01', 'date_end' : '2025-01-01',
            'off_angle' : '10',
            'evt_type' : 'SE',
            'binning_type' : 'log',
            # TO DO: additional keywords for PSD
            'e_channels' : '20. 400.',
            'N_chan' : '20',
            'e_channels_bounds' : '20. 50. 200. 400.',
            'e_channels_scales' : '2 -20 2',
            'zenith_angle':'10', 'src_sel':'10',
            'main_src_var_n':'d', 'main_src_var_unit':'d', 'main_src_var_type':'i',
            'src_var_n' : '0', 'src_var_unit' : 'd', 'src_var_type' : 'n', # 'src_max_angle' : '10.',
            'bkg_var_n' : '1', 'bkg_var_unit' : 'd', 'bkg_var_type' : 'i'
        }
        self.initial_default_spimodfit={k: self.initial_default_query[k] 
            for k in ['src_var_n', 'src_var_unit', 'src_var_type', 'bkg_var_n', 'bkg_var_unit', 'bkg_var_type']
        }
        # make shallow copy of initial dico, modified and saved throughout the queries
        self.default_query_dico = self.initial_default_query.copy()

        # if recorded, last query values used as defaults
        self.import_last_query()
    
    def import_path_config(self, config_file='config.txt'):
        """use the config file to set path attributes
        this allows user to change paths without changing this code
        """
        print('Loading config paths...')
        with open(f'{self.main_dir}/{config_file}', 'r') as f_conf:
            for line in f_conf:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse variable_name, value
                if ',' in line:
                    variable_name, value = line.split(',', 1)
                    variable_name = variable_name.strip()
                    value = value.strip()
                    # Set as attribute
                    setattr(self, variable_name, value)
                    print(f'{variable_name} = {value}')

    def import_rev_nodes(self):
        '''import the perigee time (~ start of revolution) for all revs'''
        if self.all_revs_path is None:
            return
        hdul= fits.open(self.all_revs_path)
        self.MJDREF = hdul[1].header['MJDREF']
        self.rev_dates_df = pd.DataFrame({'REV':hdul[1].data['REVOLUTION'],
                      'IJD_START':hdul[1].data['TIME_PERIGEE'].astype(np.float64)
                      })
        self.rev_dates_df['MJD_START'] = self.MJDREF + self.rev_dates.IJD_START
        self.rev_dates_df['DATE_START'] = self.rev_dates_df.MJD_START.apply(mjd_to_isot)
        return self.rev_dates_df

    ########## Query tools ##########

    def import_last_query(self):
        if os.path.isfile(f'{self.main_dir}/last_run.pkl'):
            print('Loading last values as defaults...')
            with open(f'{self.main_dir}/last_run.pkl', 'rb') as f_last:
                self.default_query_dico = pickle.load(f_last)
        else:
            print(f'No last_run file found in {self.main_dir}.')
    
    def save_current_query(self):
        with open(f'{self.main_dir}/last_run.pkl', "wb") as f_last:
            pickle.dump(self.default_query_dico, f_last)
    
    def reset_default_query(self):
        with open(f'{self.main_dir}/last_run.pkl', "wb") as f_init:
            pickle.dump(self.initial_default_query, f_init)

    def query(self, message, default=None, default_key=None, expected_type=None):
        """ask for input. if empty, use default value.
        default_key can also be used instead of default to access value in default dico
        TO DO: add a while loop to check for type (ex: if input cannot be turned into int, as for input again...)
        """

        # if testrun, use default without asking for input.
        if self.testrun:
            value = ''
        else:
            # use the default value from the default_query_dico
            if default_key is not None:
                if default_key in self.default_query_dico:
                    default = self.default_query_dico[default_key]
                else:
                    default= ''
            value = input(f'{message} [default: {default}]\n')

        if not value:
            value = default
            print(f'using default: {value}\n')

        if self.update_default_dico:
            self.default_query_dico[default_key] = value
            self.save_current_query()
        return value

    ########## General tools ##########

    def radec_to_galactic_180range(self, ra, dec):
        """Convert RA/Dec to Galactic coordinates
        use SPI convention with longitude in [-180, 180]° range"""
        source_coord_radec = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        source_coord_galactic = source_coord_radec.galactic
        long_src_coord = source_coord_galactic.l.deg
        if long_src_coord >= 180:
            long_src_coord -= 360
        lat_src_coord = source_coord_galactic.b.deg
        return lat_src_coord, long_src_coord
    
    def add_dir_layer(self, dir_name, overwrite_dir=None):
        """create a new directory. if it already exists, ask whether to remove it.
        if overwrite_dir is not None, skip the interactive input.
        """
        all_dir_list = os.listdir('.')
        write_dir = True
        if dir_name in all_dir_list:
            print(f'Date directory {dir_name} already exists.\n')
            if overwrite_dir is None:
                overwrite_dir = self.query('Do you wish to over-write? (y/n)', 'n')
            if overwrite_dir is True or overwrite_dir == 'y':
                shutil.rmtree(dir_name, ignore_errors=True)
            else:
                write_dir = False
        if write_dir:
            print('Creating new directory...')
            os.mkdir(f'{dir_name}')
            print(f'Directory {dir_name} successfully created.')
        return write_dir
    
    def run_spi_cmd(self, CMD, logfile = 'spiselectscw.log', isolate_env=True):
        """run a command outside python within its own isolated env
        used for: spiselectscw, spimodfit, rmfgen
        """
        if isolate_env:
            if self.initial_env is not None:
                cmd_env = self.initial_env.copy()
            else:
                cmd_env = os.environ.copy()
        else:
            cmd_env = None

        try:
            with open(logfile, 'w') as log:
                log.write(f"{socket.gethostname()} {datetime.now()}\n")
                log.write(f"Command: {CMD}\n")
                log.flush()
                
                process = subprocess.Popen(CMD, stdout=log, stderr=subprocess.STDOUT, text=True,
                                           env=cmd_env)
                # show cycling dots to indicate the command is running
                n_dots= 3
                dot_frames = ['.'*i for i in range(0, n_dots+1)]
                i=0
                while process.poll() is None:
                    print(f'\rRunning command{dot_frames[i % (n_dots+1)]:<{n_dots+1}}', end='', flush=True)
                    i += 1
                    time.sleep(0.5)
                
                log.write(f"\n{socket.gethostname()} {datetime.now()}\n")
            
            print(f"\nCommand completed with exit status: {process.returncode}")
            print(f"Output logged to {logfile}")
            return process.returncode
        
        except Exception as e:
            print(f"Error running command: {e}")
            return False
        
    ########## Source ##########

    def setup_source(self, src_dir, full_name=None, ra=None, dec=None):
        """Setup source directory with provided source information"""
        all_dir_list = os.listdir(self.main_dir)
        self.src_dir = src_dir
        if src_dir in all_dir_list:
            print(f'Directory {src_dir} already exists.\n')
            if full_name is None:
                try:
                    with open(f'{self.main_dir}/{src_dir}/source_info.txt','r') as f_src:
                        full_name = f_src.readline().rstrip('\n')
                        ra = np.float32(f_src.readline().rstrip('\n'))
                        dec = np.float32(f_src.readline().rstrip('\n'))

                    print('Source full name found: '+full_name)
                except FileNotFoundError:
                    print('No source_info.txt file found in directory!')
        else:
            print('Creating new source directory...')
            os.mkdir(f'{self.main_dir}/{src_dir}')
        
        os.chdir(f'{self.main_dir}/{src_dir}')
        
        if full_name is not None and ra is not None and dec is not None:
            self.full_name = full_name
            self.ra = np.float32(ra)
            self.dec = np.float32(dec)

            if not os.path.isfile('source_info.txt'):
                with open('source_info.txt', 'w') as f:
                    f.write(self.full_name+'\n')
                    f.write(str(self.ra)+'\n')
                    f.write(str(self.dec)+'\n')
        
        if self.ra is not None:
            self.lat_src_coord, self.long_src_coord = self.radec_to_galactic_180range(self.ra, self.dec)
            print(f'Galactic coordinates: l={self.long_src_coord:.2f}° b={self.lat_src_coord:.2f}°')
        self.source_coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg)

    def setup_source_interactive(self):
        """Interactive setup of source directory and information"""
        full_name = None
        ra = None
        dec = None
        src_dir = self.query('Short-hand name for directory?', default_key='src_dir')

        all_dir_list = os.listdir(self.main_dir)
        if src_dir in all_dir_list:
            try:
                with open(f'{self.main_dir}/{src_dir}/source_info.txt','r') as f_src:
                    full_name = f_src.readline().rstrip('\n')
                    ra = np.float32(f_src.readline().rstrip('\n'))
                    dec = np.float32(f_src.readline().rstrip('\n'))
                print(f'Source full name found: {full_name} ')
            except FileNotFoundError:
                pass
        
        if full_name is None:
            hdul = fits.open(self.gnrl_cat_path)
            gnrl_cat_table = hdul[self.gnrl_cat_ext].data
            full_name = self.query('* Full name of the source?\n', default_key ='full_name')
            gnrl_cat_select = gnrl_cat_table[gnrl_cat_table['NAME'] == full_name]
            if len(gnrl_cat_select) > 0:
                print('Source found in general catalog.')
                source_info = gnrl_cat_select[0]
                ra, dec = source_info['RA_OBJ'], source_info['DEC_OBJ']
                print(f'Coordinates RA={ra:.2f}° Dec={dec:.2f}°')
            else:
                print(full_name,' was not found in cat!')
                print('* Enter RA/DEC coordinate directly\n')
                ra = np.float32(input('RA?'))
                dec = np.float32(input('Dec?'))
        
        self.setup_source(src_dir, full_name, ra, dec)

    ########## Select obs dates ##########

    def select_observations(self, date_start, date_end, off_angle, overwrite_dir=None):
        """Select observations within time range and off-axis angle"""
        self.date_start = date_start
        self.date_end = date_end
        self.off_angle = off_angle
        self.date_dir = f'{date_start}_{date_end}_{off_angle:.0f}deg'
        
        new_date_dir = self.add_dir_layer(self.date_dir, overwrite_dir)
        os.chdir(self.date_dir)
        
        if os.path.isfile('df_select.csv'):
            print('Importing df_select in directory...')
            # TO DO: import REV as string
            self.df_select = pd.read_csv('df_select.csv')
        else:
            self.df_select = self.scw_tracer_db.df_scw.loc[(self.scw_tracer_db.df_scw.DateStart < date_end) & (self.scw_tracer_db.df_scw.DateEnd > date_start)]
            print(f'Searching for pointings within {off_angle}° of source...')
            # source_coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg)
            self.df_select['skycoord'] = self.df_select.apply(lambda x:SkyCoord(ra=x.RA_SCX*u.deg, dec=x.DEC_SCX*u.deg), axis=1)
            self.df_select['dist_angle'] = self.df_select['skycoord'].apply(lambda x:self.source_coord.separation(x).value)
            self.df_select = self.df_select[self.df_select.dist_angle < off_angle]
            print('Saving df...')
            self.df_select.to_csv('df_select.csv')
        
        self.N_point_select = len(self.df_select)
        self.unique_revs = self.df_select.REV.unique()
        self.N_unique_revs = len(self.unique_revs)
        print(f"{GREEN}Found {self.N_point_select} pointings, for {self.N_unique_revs} unique revolutions.{RESET}")
        if self.N_unique_revs < 30:
            print(self.unique_revs)
        print(f"Total exposure: {self.df_select.TElapse.sum()/1000:,.1f} ks")
    
    def select_observations_interactive(self):
        """Interactive selection of observations with date and off-axis angle"""

        # check for previous date dir
        date_dir_list = [d for d in glob.glob("*_*_*deg") if os.path.isdir(d)]
        if len(date_dir_list) > 0:
            print(f'* Found the following date directories:')
            for i, d_dir in enumerate(date_dir_list):
                print(f'{i} - {d_dir}')
            d_dir_input = self.query(f'Choose directory (#) or create new (n)', 'n')
            if d_dir_input != 'n':
                self.date_dir = date_dir_list[int(d_dir_input)]
                date_start, date_end, off_angle_str = self.date_dir.split('_')
                off_angle = np.float32(off_angle_str.strip('deg'))
                self.select_observations(date_start, date_end, off_angle)
                # self.select_observations(date_start, date_end, off_angle, overwrite_dir=False)
                return

        date_start = self.query('Starting date (YYYY-MM-DD)', default_key='date_start')
        date_end = self.query('End date (YYYY-MM-DD)', default_key='date_end')
        off_angle_query = self.query('* Off-axis angle of pointings (in deg)?', default_key='off_angle')
        off_angle = np.float32(off_angle_query)

        self.select_observations(date_start, date_end, off_angle)

    
    ########## Select energies ##########

    def write_energies_txt(self):
        """extra energies txt file with easy access"""
        print('Writing energies.txt...')
        with open('energies.txt', 'w') as f_ener:
            f_ener.write(self.evt_type+'\n')
            [f_ener.write(f'{e} ') for e in self.e_channels]
            f_ener.write('\n')
            [f_ener.write(f'{e} ') for e in self.e_bin_size]
            f_ener.write('\n')
    
    def setup_energies(self, evt_type, binning_type, emin=None, emax=None, nbins=None, 
                       e_channels_bounds=None, e_channels_scales=None):
        """Setup energy binning with provided parameters"""
        
        energy_loaded = False
        # ener_dir_list = glob.glob('*bins_*')
        ener_dir_list = [d for d in glob.glob("*_*_*_*") if os.path.isdir(d)]
        
        if len(ener_dir_list) > 0 and emin is None and emax is None:
            print(f'* Found the following energy directories:')
            for i, e_dir in enumerate(ener_dir_list):
                print(f'{i} - {e_dir}')
        
        if self.ener_dir and os.path.isfile(f'{self.ener_dir}/energies.txt'):
            print('Found energies file, loading energies...')
            with open(f'{self.ener_dir}/energies.txt', 'r') as f_ener:
                self.evt_type = f_ener.readline().strip()
                self.e_channels = np.array(f_ener.readline().split(), dtype=np.float32)
                self.e_bin_size = np.array(f_ener.readline().split(), dtype=np.float32)
            bin_size = self.EVT_BIN_SIZE[self.evt_type]
            self.Nchan = len(self.e_bin_size)
            energy_loaded = True
            # os.chdir(self.ener_dir)
        
        if not energy_loaded:
            self.evt_type = evt_type
            bin_size = self.EVT_BIN_SIZE[self.evt_type]
            self.binning_type = binning_type
            
            if binning_type == 'perso':
                self.e_channels_bounds = np.array(e_channels_bounds, dtype=np.float32)
                self.e_channels_scales = np.array(e_channels_scales, dtype=np.int32)
                self.e_channels = self.rebin_enerby_bounds(self.e_channels_bounds, self.e_channels_scales)
                
            
            elif binning_type == 'lin' or binning_type == 'log':
                if binning_type == 'log':
                    self.e_channels = np.logspace(np.log10(emin), np.log10(emax), nbins+1, dtype=np.float32)
                else:
                    self.e_channels = np.linspace(emin, emax, nbins+1, dtype=np.float32)
            else:
                raise ValueError(f"Binning type '{binning_type}' unknown.")
            
            self.e_channels = self.e_channels - self.e_channels%bin_size
            self.e_bin_size = np.diff(self.e_channels)
            self.Nchan = len(self.e_channels) - 1
            
            self.ener_dir = f'{self.e_channels[0]:.0f}_{self.e_channels[-1]:.0f}_{self.Nchan}{self.binning_type}_{self.evt_type}'
        
        print(f'Energy range: {self.e_channels[0]:.1f} - {self.e_channels[-1]:.1f} keV')
        print('Energy bins:', self.e_channels)
    
    def setup_energies_interactive(self):
        """Interactive setup of energy binning
        TO DO: automatic double (SE+PSD) or triple (SE+PSD+HE) analysis
        """

        # check for previous energy dir
        ener_dir_list = [d for d in glob.glob("*_*_*_*") if os.path.isdir(d)]
        # ener_dir_list = glob.glob('*bins_*')
        if len(ener_dir_list) > 0:
            print(f'* Found the following energy directories:')
            for i, e_dir in enumerate(ener_dir_list):
                print(f'{i} - {e_dir}')
            ener_dir_input = self.query(f'Choose directory (#) or create new (n)', 'n')
            if ener_dir_input != 'n':
                self.ener_dir = ener_dir_list[int(ener_dir_input)]
                self.setup_energies(None, None)
                return
        
        # create new energy dir from scratch
        evt_input = self.query("Event type? (SE/PSD/HE)", default_key='evt_type')
        self.evt_type = evt_input
        binning_type = self.query("Use lin/log/perso?", default_key='binning_type')
        
        if binning_type == 'perso':
            e_channels_input = self.query('Choose energy bounds (separated by spaces)', default_key='e_channels_bounds')
            e_channels_bounds = e_channels_input.split()
            e_channels_scales_input = self.query('Choose number of bins between each bound, where negative values = log scaling',
                                        default_key='e_channels_scales')
                                        # '1 '*(len(e_channels_bounds) - 1) )
            e_channels_scales = e_channels_scales_input.split()
            self.setup_energies(self.evt_type, binning_type, e_channels_bounds=e_channels_bounds, 
                              e_channels_scales=e_channels_scales)
        
        else:
            e_channels_input = self.query('Chose Emin and Emax (separated by spaces)', default_key='e_channels')
            emin, emax = map(float, e_channels_input.split())
            nbins = int(self.query('Choose number of bins', default_key='N_chan'))
            self.setup_energies(self.evt_type, binning_type, emin=emin, emax=emax, nbins=nbins)
    

    def rebin_enerby_bounds(self, e_channels_bounds, e_channels_scales, dtype=np.float16):
        """Generate e_channels from bounds and scales (like OSA soft)
        scales: positive = linear bins, negative = logarithmic bins
        """
        e_channels = []
        for i in range(len(e_channels_bounds) - 1):
            emin = e_channels_bounds[i]
            emax = e_channels_bounds[i + 1]
            scale = e_channels_scales[i]
            
            if scale > 0:
                e_bin = np.linspace(emin, emax, scale + 1, dtype=np.float32)
            else:
                e_bin = np.logspace(np.log10(emin), np.log10(emax), -scale + 1, dtype=np.float32)
            
            if i == len(e_channels_bounds) - 2:
                e_channels.extend(e_bin)
            else:
                e_channels.extend(e_bin[:-1])

        e_channels = np.array(e_channels, dtype=dtype)
        print(f"Generated {len(e_channels)-1} energy bins edges from {len(e_channels_bounds)} bounds")
        return e_channels
    
    ########## SPISELECTSCW ##########

    def run_spiselectscw(self, run_id):
        """Execute the spiselectscw command following the submit-spiselectscw.sh script."""
        
        # CMD = "/data1/ipp_afs_mirror/integral/software/local/spiselectscw/4.02/amd64_sles11_g++/spiselectscw"
        parfile_path = f'spiselectscw.{run_id}.par'
        
        if not os.path.isfile(parfile_path):
            print(f"Parameter file {parfile_path} doesn't exist. Exit!")
            return False
        
        os.environ['CFITSIO_INCLUDE_FILES'] = self.cfitsio_templates_dir
        os.environ['PFILES'] = '.'
        
        if not os.path.isdir(run_id):
            os.mkdir(run_id)
        
        os.chdir(run_id)
        
        if not os.path.isdir('spi'):
            os.mkdir('spi')
        
        if not os.path.isfile('spi_off_det.fits'):
            os.symlink(self.spi_off_det, 'spi_off_det.fits')
        
        if not os.path.isfile('spi_gnrl_bti.fits'):
            os.symlink(self.spi_gnrl_bti, 'spi_gnrl_bti.fits')
        
        shutil.copy2(f'../{parfile_path}', 'spiselectscw.par')
        
        create_scw_file = None
        scw_file = None
        
        try:
            with open('spiselectscw.par', 'r') as f:
                for line in f:
                    if 'create_scw_file' in line and ',' in line:
                        create_scw_file = int(line.split(',')[3].strip())
                    if line.startswith('scw_file') and '"' in line:
                        scw_file = line.split('"')[1]
        except Exception as e:
            print(f"Warning: Could not parse parameter file: {e}")
        
        if create_scw_file == 0 and scw_file and not os.path.isfile(scw_file):
            os.symlink('../scw.fits', 'scw.fits')
        
        print(f"Launching spiselectscw (RUN_ID: {run_id})...")
        err_code = self.run_spi_cmd(self.spiselectscw_cmd, isolate_env=False)
        return err_code
        

    def make_spiselectscw_par(self, skip_spalready_exists=False):
        """create spiselectscw parameter file"""
        skip_spiselectscw = False
        if os.path.isdir(self.ener_dir):
            if skip_spalready_exists:
                skip_spiselectscw = True
            else:
                raise FileExistsError(f"{self.ener_dir} directory already exists. Set skip_spalready_exists=True to skip.")
        
        if not skip_spiselectscw:
            print('Creating spiselectscw parameter file...')
            e_bins_str = ''
            e_size_str = ''
            for i in range(self.Nchan):
                e1, e2 = self.e_channels[i], self.e_channels[i+1]
                e_bins_str += f'{e1:.1f}-{e2:.1f}, '
                e_size_str += f'{e2-e1:.1f}, '
            # spiselectscw wants revs to be written as int, without leading 0s
            revs_str = ','.join(map(lambda x:str(int(x)), self.unique_revs))
            
            spiselect_par_str = f""" ### User input for {self.full_name}
fits_revolutions_list,s,h,"{revs_str}",,,"Prepared revolutions database to read"
revolutions_cond_value,s,h,"{revs_str}",,,"Selection ranges for revolutions (must* be same as fits_revolution_list)"
select_PtgX_masks_num,i,h,1,0,,"Number of masks for X pointing"
select_PtgX_masks_globrev,i,h,0,0,1,"global reverse of masks result for X pointing"
select_PtgX_masks_name_1,s,h,"Ellipse",,,"Name of the model of the mask 1 for X pointing"
select_PtgX_masks_chi_list_1,r,h,{self.long_src_coord:.3f},-180,180,"chi center of the mask 1 for X pointing"
select_PtgX_masks_psi_list_1,r,h,{self.lat_src_coord:.3f},-90,90,"psi center of the mask 1 for X pointing"
select_PtgX_masks_angle_list_1,r,h,0,,,"rotation center of the mask 1 for X pointing"
select_PtgX_masks_coordsys_1,s,h,"GALACTIC",,,"coordinate system of the mask 1 for X pointing"
select_PtgX_masks_rev_1,i,h,0,0,1,"Reverts the effect of the mask 1 for X pointing"
select_PtgX_masks_npar_1,i,h,4,0,,"Number of parameters passed for the mask 1 for X pointing"
select_PtgX_masks_par_1,s,h,"0 0 {self.off_angle} {self.off_angle}",,,"Parameters passed for the mask 1 for X pointing"
rev_std_proc_pref,s,h,"{self.data_dir}/",,,"Location of revolutions std process - Prefix"
energy_bins,s,h,"{e_bins_str[:-2]} keV",,,"Energy bins selection"
energy_rebin,s,h,"{e_size_str[:-2]} keV",,,"Energy rebinning (must match bins)"
out_expo_map_dol,s,h,"expo.fits",,,"Name of the output exposure map. None if left blank"
    """
            with open(f'{self.templates_dir}/{self.spiselect_par_tpl_file}', 'r') as f_tpl:
                tpl_content = f_tpl.read()
            
            combined_content = spiselect_par_str + 5*'\n' + tpl_content
            output_par_file = f'spiselectscw.{self.ener_dir}.par'
            with open(output_par_file, 'w') as f_out:
                f_out.write(combined_content)
            
            print(f'Parameter file created: {output_par_file}')
            self.spiselectscw_return_code = self.run_spiselectscw(self.ener_dir)
            # create dir to store fov catalogs
            os.mkdir('cat')

            if not os.path.isfile('energies.txt'):
                self.write_energies_txt()

            if self.spiselectscw_return_code != 0:
                print(f'spiselectscw failed to complete (Error code {self.spiselectscw_return_code}).')
                print(f'Check log file {os.getcwd()}/spiselectscw.log')
                os.chdir(self.initial_dir)
                raise AssertionError()
            
        else:
            os.chdir(self.ener_dir)
    
    
    def make_spiselectscw_par_interactive(self):
        """create spiselectscw parameter file
        (with interactive input)
        """
        if os.path.isdir(self.ener_dir):
            rm_ener_dir = self.query(f"{self.ener_dir} dir already exists. Remove it? (y/n)", "n")
            if rm_ener_dir == 'y':
                shutil.rmtree(self.ener_dir, ignore_errors=True)
                # os.rmdir(self.ener_dir)
            else:
                self.make_spiselectscw_par(skip_spalready_exists=True)
                return
        
        self.make_spiselectscw_par(skip_spalready_exists=False)
    
    ########## Background ##########

    def process_background(self):
        """Process background for the observation using python background module
        """
        
        self.ener_path = f'{self.main_dir}/{self.src_dir}/{self.date_dir}/{self.ener_dir}'
        # check if bkg dir was already created
        bg_path_list = glob.glob(f'{self.ener_path}/spi/bg-*')

        if len(bg_path_list) > 0:
            print('Background directory found.')
            self.bkg_dir = bg_path_list[0]

        else:
            print('Preparing background dir...')
            scw_db_path = 'scw.fits.gz'
            self.obs_bkg = ObsBkg(self.ener_path, self.evt_type)
            livetime_rev = LiveTimeRev(self.bkg_db_dir+'/det_livetime_rev.fits', self.evt_type)
            scw_tracer_db = ScwTracerDB(scw_db_path)
            self.obs_bkg.load_tracer(scw_tracer_db)
            self.obs_bkg.normalize_tracer(livetime_rev)
            self.obs_bkg.init_rev_bkg_list(livetime_rev, self.bkg_db_dir)
            self.bkg_dict = self.obs_bkg.calc_bkg()
            self.obs_bkg.write_output_bkg()
            self.bkg_dir = self.obs_bkg.output_dir

    
    ########## Sources selection ##########

    def find_nearby(self, zenith_angle=10):
        '''find all sources within zenith angle + max off-axis angle'''
        self.zenith_angle = zenith_angle
        self.max_angle = zenith_angle + self.off_angle
        assert self.spi_cat_path is not None, 'No catalog loaded! check config.txt file.'
        self.cat = CatSPI(self.spi_cat_path)
        self.nearby_df = self.cat.find_nearby_sources(self.source_coord, self.max_angle)
        return self.nearby_df
    
    def select_brightest(self, src_sel = 15):
        '''select brightest sources'''
        if type(src_sel) == int:
            if src_sel <= 0:
                nearby_src_list = self.nearby_df.NAME.tolist()
            else:
                nearby_src_list = self.nearby_df.iloc[:src_sel].NAME.tolist()
        elif type(src_sel) == list:
            nearby_src_list = self.nearby_df.iloc[src_sel].NAME.tolist()
        self.cat.select_src(nearby_src_list)
        return nearby_src_list
    
    def select_sources_interactive(self):
        zenith_query = self.query(f'Max angle of sources for all pointings?', default_key='zenith_angle')
        zenith_angle = float(zenith_query)
        self.find_nearby(zenith_angle)
        print('Found the following sources:\n', )
        print(self.nearby_df.to_string(float_format="{:.1f}".format))

        src_sel_query = self.query('Give number to select N brightest (0 to select all), or list (ex: [0, 1, 3]).',
              default_key='src_sel')
        if '[' in src_sel_query:
            src_sel = [int(x) for x in src_sel_query[1:-1].split()]
        else:
            src_sel = int(src_sel_query)
        nearby_src_list = self.select_brightest(src_sel)
        print(f'Sources selected:', nearby_src_list)


    ########## Main source variability ##########

    VAR_UNIT_CONVERSION = {
        'pi':'constant, pointings, increments', 'di':'constant, days, increments',
        'pn':'constant, pointings, nodes', 'dn':'constant, days, nodes',
        'ri':'constant, days, nodes'
        }

    @staticmethod
    def convert_var_n_array(n, array_length=8):
        arr= np.zeros(array_length, dtype=np.int8)
        arr[0] = n
        return arr
        
    def select_src_var(self, main_src_var_unit, main_src_var_n, main_src_var_type):
        '''
        modify the variability in the catalog using keywords: VAR_MODL, VAR_NPAR, VAR_PARS (array)
        variability unit can be p(ointing) or d(ays) and is converted using a dico into correct catalog value
        increment number (main_src_var_n) is converted into an array for VAR_PARS
        for simplicity, VAR_NPAR is always 1
        '''
        self.all_src_dico= {}
        if main_src_var_type == 'i':
            npar = int(1)
            par_array = self.convert_var_n_array(main_src_var_n, self.cat.new_table['VAR_PARS'].shape[1])
        elif main_src_var_type == 'n':
            np.fromstring(main_src_var_n, dtype=int, sep=' ')
        elif main_src_var_type == 'r':
            pass
        else:
            raise NotImplementedError(main_src_var_type)


        self.all_src_dico[self.full_name] = {
            'VAR_MODL':self.VAR_UNIT_CONVERSION[main_src_var_unit+main_src_var_type],
            'VAR_NPAR':npar,
            'VAR_PARS':par_array
                                        }
        self.cat.update_src_params(self.all_src_dico)
    
    def select_src_var_interactive(self):
        print(f'*** Main source variability ({self.full_name}) ***')
        main_src_var_type = self.query('Variability type? (i=increments, n=nodes)', default_key='main_src_var_type')
        main_src_var_unit = self.query('Variability unit? (p=pointing, d=day, r=revolution)', default_key='main_src_var_unit')
        main_src_var_n = self.query('Variability number? (int)', default_key='main_src_var_n')
        self.select_src_var(main_src_var_unit, main_src_var_n, main_src_var_type)


    ########## Flux (spimodfit) ##########

    def run_spimodfit(self, run_id, clobber=True):
        """Execute the spimodfit command following the submit-spimodfit_v3.2_ga05us.sh script
        use clobber=None for interactive sessions
        other True/False to remove directory
        """
        
        # CMD = "/data1/ipp_afs_mirror/integral/software/local/spimodfit/3.2/amd64_sles11_g++/spimodfit"
        parfile_path = f'spimodfit.{run_id}.par'
        
        if not os.path.isfile(parfile_path):
            print(f"Parameter file {parfile_path} does not exist.")
            return False
        
        subdir = run_id
        if os.path.exists(subdir):
            if clobber is None: # for interactive sessions
                clobber_query= self.query(f'Directory {subdir} exists. Remove it? (y/n)', 'y')
                if clobber_query =='y': clobber = True
                else: clobber = False
            if clobber:
                print(f"Removing existing directory {subdir}...")
                shutil.rmtree(subdir, ignore_errors=True)
            else:
                print(f"Directory {subdir} exists. Please use clobber=True if you want to overwrite it")
                return False
        
        os.mkdir(subdir)
        os.chdir(subdir)
        
        shutil.copy2(f'../{parfile_path}', 'spimodfit.par')
        
        os.environ['CFITSIO_INCLUDE_FILES'] = self.cfitsio_templates_dir
        os.environ['PFILES'] = '.'
        
        print(f"Launching spimodfit (RUN_ID: {run_id})...")
        err_code = self.run_spi_cmd(self.spimodfit_cmd, logfile='spimodfit.log', isolate_env=False)

        self.spec_path = f'{self.main_dir}/{self.src_dir}/{self.date_dir}/{self.ener_dir}/{self.fit_dir}'
        return err_code
        
    
    def make_spimodfit_par(self, src_var_n=0, src_var_unit='d', src_var_type='n', src_max_angle=None,
                           bkg_var_n=1, bkg_var_unit='d', bkg_var_type='i', fov_cat_path=None, overwrite_fov_cat= True,
                           skip_spimodfit_exists=False):
        """create spimodfit parameter file"""

        self.run_path = f'{self.main_dir}/{self.src_dir}/{self.date_dir}/{self.ener_dir}'
        print(f'Current spimodfit run directory: {self.run_path}')

        skip_spimodfit = False
        if os.path.isdir(self.ener_dir):
            if skip_spimodfit_exists:
                skip_spimodfit = True
            else:
                raise FileExistsError(f"{self.ener_dir} directory already exists. Set skip_spimodfit_exists=True to skip.")
        
        if not skip_spimodfit:
            # path to the fit directory where results are stored
            self.fit_dir = f'fit_{src_var_n}{src_var_unit}{src_var_type}_{bkg_var_n}{bkg_var_unit}{bkg_var_type}'

            # saving mini catalog
            if fov_cat_path is None:
                self.fov_cat_path = f'{self.run_path}/cat/fov_cat_{src_var_n}{src_var_unit}{src_var_type}_{bkg_var_n}{bkg_var_unit}{bkg_var_type}.fits'
            else:
                self.fov_cat_path = fov_cat_path
            self.cat.save_cat(self.fov_cat_path, overwrite=overwrite_fov_cat)
            print(f'FOV catalog saved at {self.fov_cat_path}')

            # print('! Reverting to original cat for testing !')
            # self.fov_cat_path= self.spi_cat_path
            print('Creating spimodfit parameter file...')
            if src_max_angle is None:
                src_max_angle = self.zenith_angle
            
            # File paths, energy binning and catalog

            spimodfit_par_str = f"""### User input for {self.full_name}
# ----------- file paths -----------
counts_input_file,s,h,"{self.run_path}/spi/evts_det_spec.fits.gz",,," input count file"
pointing_input_file,s,h,"{self.run_path}/spi/pointing.fits.gz",,," input pointing file"
ebounds_input_file,s,h, "{self.run_path}/spi/energy_boundaries.fits.gz",,," input energy bounds file"
deadtime-dol,s,h,"{self.run_path}/spi/dead_time.fits.gz",,,"DTI deadtime/livetime input file"
gti-dol,s,h,"{self.run_path}/spi/gti.fits.gz",,,"GTI input file"
background_input_file,s,h,"{self.bkg_dir}/{self.bg_idx_filename}" ,,,"input background file"

# ----------- energy re-binning (none for standard analysis) -----------
first_energy_bin,i,h,1,1,10000,"First selected bin"
last_energy_bin,i,h,{self.Nchan},1,10000,"Last selected bin"
m_energy_rebin,i,h,1,1,100,"number of bins per rebinned energy"
energy_range_min,i,h,1,,,"minimum energy range sequence number as in ebounds file: 1,2,3..."
energy_range_max,i,h,{self.Nchan},,,"maximum energy range sequence number as in ebounds file: 1,2,3..."

# ----------- catalog -----------
source-cat-dol,s,h,"{self.fov_cat_path}",,,"input catalogue of sources "

# ----------- source variability parameters -----------
source_parameters_fit,i,h,1,0,1,"Sources fit parameter 1=yes" 
"""
            
            # Source variability parameter

            if src_var_unit == 'r':

                spimodfit_par_str += f"""
source_var_coef,s,h,"&{self.all_revs_path}[1] col=TIME_PERIGEE d n, 1435.41635 1659.46 3337.5 3799.66740 d n",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)"
sources_zenith_angle,r,h,{src_max_angle},0,," Sources maximum zenithal angle"

    """
            else:
                spimodfit_par_str += f"""# VARIATION PER REVOLUTION
source_var_coef,s,h,"{src_var_n} {src_var_unit} {src_var_type}",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes")
sources_zenith_angle,r,h,{src_max_angle},0,," Sources maximum zenithal angle"

    """
            
            # Background variability parameter

            spimodfit_par_str += """
# ----------- background variability parameters -----------
collect_background_models,i,h,0,0,1,"Collect background components into one model (0/1)"
"""
            if bkg_var_unit=='r':
                spimodfit_par_str += f"""
# OPTION 1a: FIT BACKGROUND ONCE PER REVOLUTION AND DETECTOR FAILURE
# CAN ALSO REPLACE all_revs.fits WITH EVERY N REVOLUTIONS WHERE N=0.003125 TO 30
# CHECK /data1/ipp_afs_mirror/integral/shared_analysis/cookbook/revolutions/ FOR AVAILABILITY
background_var_coef_01,s,h,"&{self.all_revs_path}[1] col=TIME_PERIGEE d n, 1435.41635 1659.46 3337.5 3799.66740 d n",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)"
background_var_coef_02,s,h,"&{self.all_revs_path}[1] col=TIME_PERIGEE d n, 1435.41635 1659.46 3337.5 3799.66740 d n",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)"
"""
            else:
                spimodfit_par_str += f"""
# OPTION 1b: FIT BACKGROUND ONCE PER POINTING/DAYS
background_var_coef_01,s,h,"{bkg_var_n} {bkg_var_unit} {bkg_var_type}",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)"
background_var_coef_02,s,h,"{bkg_var_n} {bkg_var_unit} {bkg_var_type}",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)"
"""

            with open(f'{self.templates_dir}/{self.spimodfit_par_tpl_file}', 'r') as f_tpl:
                tpl_content = f_tpl.read()
            
            combined_content = spimodfit_par_str + 5*'\n' + tpl_content
            output_par_file = f'spimodfit.{self.fit_dir}.par'
            with open(output_par_file, 'w') as f_out:
                f_out.write(combined_content)
            
            print(f'Parameter file created: {output_par_file}')
            self.spimodfit_return_code = self.run_spimodfit(self.fit_dir, clobber=None)

            if self.spimodfit_return_code != 0:
                print(f'spimodfit failed to complete (Error code {self.spimodfit_return_code}).')
                print(f'Check log file {os.getcwd()}/spimodfit.log')
                os.chdir(self.initial_dir)
                raise AssertionError()
            else:
                N_src_spec= len(glob.glob('spectra_*'))
                print(f'spimodfit ran successfully. {N_src_spec} spectra created.')
    
    
    def make_spimodfit_par_interactive(self):
        """create spiselectscw parameter file
        (with interactive input)
        """
        # print('*** Selecting spimodfit parameters')
        print('Initial default spimodfit parameters:\n', self.initial_default_spimodfit)
        std_spimodfit_query= self.query('Use standard spimodfit analysis? (y/n)', default='n')
        if std_spimodfit_query == 'y':
            self.make_spimodfit_par(**self.initial_default_spimodfit)

        else:
            print('*** Other sources variability ***\n')
            src_var_unit = self.query('Variability unit? (p=pointing, d=day, r=revolution)', default_key='src_var_unit')
            src_var_n = self.query('Variability number? (int)', default_key='src_var_n')
            src_var_type = self.query('Variability type? (i(ncrements)/n(nodes))', default_key='src_var_type')

            print('*** Background variability ***\n')
            bkg_var_unit = self.query('Variability unit? (p=pointing, d=day, r=revolution)', default_key='bkg_var_unit')
            bkg_var_n = self.query('Variability number? (int)', default_key='bkg_var_n')
            bkg_var_type = self.query('Variability type? (i(ncrements)/n(nodes))', default_key='bkg_var_type')

            self.make_spimodfit_par(src_var_n=src_var_n, src_var_unit=src_var_unit, src_var_type=src_var_type, src_max_angle=None,
                           bkg_var_n=bkg_var_n, bkg_var_unit=bkg_var_unit, bkg_var_type=bkg_var_type
                           )
    
    def analyze_spimodfit(self, sigma_threshold = 4, verbose=True):
        '''TO DO: make df that contains all spimodfit info'''
        with open('spimodfit.log', 'r') as f_fit:
            spimodfit_log = f_fit.readlines()
            pearson_chi2_log = [l.split('=')[-1] for l in spimodfit_log if "Corresponding Pearson's chi2 stat / dof" in l]
            pearson_chi2_list = np.array([float(p.split('/dof')[0]) for p in pearson_chi2_log])
            dof_list = np.array([int(p.split(' ')[2][1:]) for p in pearson_chi2_log])

            chi2_threshold = sigma_threshold * np.sqrt(2/dof_list)
            channels_above_thresh = np.where(np.abs(pearson_chi2_list - 1) > chi2_threshold)[0]
            if verbose:
                print(f"Pearson's chi2 stat / dof for each energy bin (threshold = {sigma_threshold} sigma)")
                print(f'{'bin':<3}  {'E range (keV)':<15}   {'chi2 red./dof'}')
                for i in range(len(pearson_chi2_list)):
                    if np.abs(pearson_chi2_list[i] - 1) > chi2_threshold[i]:
                        print(f'{BOLD}{RED}{i:<3}: {self.e_channels[i]:<6} - {self.e_channels[i+1]:<6} : {pearson_chi2_list[i]:.2f} ({dof_list[i]} dof) (thres= ± {chi2_threshold[i]:.3f}){RESET}')
                        # print(f'{BOLD}{RED}{i}: {self.e_channels[i]} - {self.e_channels[i+1]} : {pearson_chi2_list[i]:.2f}{RESET}')
                    else:
                        print(f'{i:<3}: {self.e_channels[i]:<6} - {self.e_channels[i+1]:<6} : {pearson_chi2_list[i]:.2f} ({dof_list[i]} dof) (thres= ± {chi2_threshold[i]:.3f})')
                        # print(f'{i}: {self.e_channels[i]} - {self.e_channels[i+1]} : {pearson_chi2_list[i]:.2f}')
            if len(channels_above_thresh) > 0:
                print(f'\n! Warning ! {len(channels_above_thresh)} energy bins have a chi2 above the set threshold.')
                print(f'Channels: {channels_above_thresh.tolist()}')
                print('Re-running spimodfit or ignoring these channels is recommanded.')

            
    ########## Response generation (spirmfgen) ##########

    def generate_response(self, clobber='n'):
        """Generate response matrix using spirmf command"""
        
        outRMFfile = "rmf_spi"
        
        # Create symbolic links to templates
        print("Creating symbolic links to templates...")
        try:
            for tpl_file in os.listdir(self.cfitsio_templates_dir):
                if tpl_file.endswith('.tpl'):
                    src = os.path.join(self.cfitsio_templates_dir, tpl_file)
                    dst = os.path.join('.', tpl_file)
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
        except Exception as e:
            print(f"Warning: Could not create template symlinks: {e}")
        
        print(f"Launching spirmf to generate response matrix...")
        CMD = [
                    f"{self.rmfgen_cmd}",
                    f"rw-grp-dol={self.inRMFfile}[GROUPING]",
                    f"ebounds-dol={self.spimodfit_result_file}[SPI.-EBDS-SET]",
                    f"outfile={outRMFfile}",
                    "single=y", "update=n", "phafile=", f"clobber={clobber}", "chatter=2"
                ]
        
        err_code = self.run_spi_cmd(CMD, logfile='spirmf.log')
        
        # Clean up template files
        print("Cleaning up template files...")
        for tpl_file in os.listdir('.'):
            if tpl_file.endswith('.tpl'):
                try:
                    os.remove(tpl_file)
                except OSError:
                    pass

        if err_code == 0:
            self.add_rmf_keywords(outRMFfile=outRMFfile)
            self.fix_spec_extensions(outRMFfile=outRMFfile)

        return err_code
    
    def add_rmf_keywords(self, outRMFfile='rmf_spi'):
        '''add the RMF name to the RESPFILE keyword
        this allows XSpec to load RMF automatically
        '''
        print('Adding SPE keyword to rmf.fits ...')
        with fits.open(f"{outRMFfile}.rmf.fits", mode="update", memmap=True) as hdul:
            hdul["GROUPING"].header["ISDCLEVL"] = "SPE"
            hdul["SPI.-RMF.-RSP"].header["ISDCLEVL"] = "SPE"
            hdul["SPI.-EBDS-SET"].header["ISDCLEVL"] = "SPE"
            hdul.flush()

        print('Adding RMF keyword to spectra FITS...')
        spec_path_list = glob.glob(f'{self.spec_path}/spectra_*')
        if len(spec_path_list)>0:
            for sfile in spec_path_list:
                with fits.open(sfile, mode="update", memmap=True) as hdul:
                    hdul["SPI.-PHA1-SPE"].header["RESPFILE"] = f"{outRMFfile}.rmf.fits"
                    hdul.flush()
        else:
            print(f'No spectra found in {self.spec_path}. Could no write RESPFILE keyword.')
    
    def fix_spec_extensions(self, outRMFfile='rmf_spi'):
        print('Modifying spectra and RMF extension for 3ML compatibility')
        # Update second extension (extension 2) -> EXTNAME = 'MATRIX'
        with fits.open(f"{outRMFfile}.rmf.fits", mode="update", memmap=True) as hdul:
            hdul[2].header["EXTNAME"] = "MATRIX"
            hdul[3].header["EXTNAME"] = "EBOUNDS"

        # Update second extension (extension 2) -> EXTNAME = 'SPECTRUM'
        spec_path_list = glob.glob(f'{self.spec_path}/spectra_*')
        if len(spec_path_list)>0:
            for sfile in spec_path_list:
                with fits.open(sfile, mode="update", memmap=True) as hdul:
                    hdul[2].header["EXTNAME"] = "SPECTRUM"


class CatSPI:
    def __init__(self, cat_path=None, cat_ext='SPI.-SRCL-CAT', uint=True):
        self.cat_path = cat_path
        self.cat_ext = cat_ext
        self.hdul = fits.open(cat_path, uint=uint)
        cat_hdu = self.hdul[cat_ext]
        self.header = cat_hdu.header
        self.table = cat_hdu.data
        self.new_table = None

    @staticmethod
    def _norm_name(x):
        return x.decode("utf-8", errors="ignore").strip() if isinstance(x, (bytes, np.bytes_)) else str(x).strip()

    @staticmethod
    def _fits_str(value, sample):
        if isinstance(sample, (bytes, np.bytes_)):
            return value.encode("utf-8")
        return value

    def find_nearby_sources(self, target, max_angle=20, flux_col = 'FLUX_CRAB_LB', flux_element= 0,
                     ra_colname = "RA_OBJ", dec_colname = "DEC_OBJ"):
        '''find all sources within max_angle sorted by their flux'''

        # find closest sources
        src_coords = SkyCoord(self.table[ra_colname] * u.deg, self.table[dec_colname] * u.deg, frame="icrs")
        sep_deg = target.separation(src_coords).deg
        mask = sep_deg <= max_angle
        sel_idx = np.where(mask)[0]

        has_flux_col = flux_col in self.table.names
        if has_flux_col:
            # Flux value used for sorting (vector column element or scalar column)
            flux_data = self.table[flux_col]
            flux_val = flux_data[sel_idx, flux_element]

            # Sort by flux element
            order = np.argsort(flux_val)[::-1]
            sel_idx = sel_idx[order]
            flux_val = flux_val[order]
        else:
            # If flux is unavailable, keep deterministic ordering by source separation.
            order = np.argsort(sep_deg[sel_idx])
            sel_idx = sel_idx[order]

        sources_table = pd.DataFrame({
            "NAME": [self._norm_name(v) for v in self.table["NAME"][sel_idx]],
            ra_colname: self.table[ra_colname][sel_idx],
            dec_colname: self.table[dec_colname][sel_idx],
            "SEP_DEG": sep_deg[sel_idx],
        })
        # if has_flux_col:
        #     sources_table[f"{flux_col}[{flux_element}]"] = flux_val
        print(f'Found {len(sources_table)} sources in catalog within {max_angle}° of target sources (RA={target.ra.value:.2f}°, Dec={target.dec.value:.2f}°)')
        sources_table.reset_index(drop=True, inplace=True)
        if has_flux_col:
            sources_table[f"Flux (mCrab)"] = flux_val * 1e3
        return sources_table
    

    def select_src(self, list_src, save_dir=None, save_name='nearby_cat.fits'):
        '''make a new catalog restricted to a list of sources'''
        cat_names = np.array([self._norm_name(n) for n in self.table["NAME"]])
        self.new_table = self.table[np.isin(cat_names, list_src)]

        if save_dir is not None:
            # copy HDUL and change cat table to save to new FITS
            self.new_hdul = self.hdul.copy()
            self.new_hdul[self.cat_ext].data = self.new_table
            # new_hdus = [new_cat_hdu if hdu.name == "SPI.-SRCL-CAT" else hdu.copy() for hdu in self.hdul]
            new_cat_path = f"{new_cat_path}/{save_name}"
            fits.HDUList(self.new_hdul).writeto(new_cat_path, overwrite=True)
            print(f'New cat written at {new_cat_path}')
        return self.new_table

    def update_src_params(self, all_src_dico: dict):
        '''
        update many parameters of many sources
        all_src_dico should contain a dico for each src name with parameter name and value inside 
        '''
        if self.new_table is None:
            self.new_table = self.table.copy()

        for src in all_src_dico.keys():
            src_dico = all_src_dico[src]
            if src in self.new_table.NAME:
                for par_key in src_dico.keys():
                    self.new_table[par_key][self.new_table.NAME == src] = src_dico[par_key]
            else:
                print(f'Cannot change parameters of {src} (not found in new cat table).')
    
    def save_cat(self, new_cat_path, overwrite=True):
        '''create a new catalog using the modified new_table'''
        # some columns to remove
        flux_col_list = ['FLUX_LB', 'FLUX_CRAB_LB']

        self.new_hdul = self.hdul.copy()
        self.new_hdul[self.cat_ext].data = self.new_table
        tab = Table(self.new_hdul[self.cat_ext].data)
        # I suspect some extra columns can make spimodfit crash
        for flux_col in flux_col_list:
            if flux_col in tab.columns:
                tab.remove_columns(flux_col)
        self.new_hdul[self.cat_ext] = fits.BinTableHDU(tab, header=self.new_hdul[self.cat_ext].header)
        self.new_hdul.writeto(new_cat_path, overwrite=overwrite)

    def add_src(self, src_dico, confusion_angle=1, ref_src_name = 'Crab'):
        '''
        add new source to catalog.
        check whether there is already a source within the confusion_angle first.
        for parameters not found in the src_dico, a reference row with ref_src_name is used.
        '''
        pass

