"""
SPI-TAP — SPI Transient Analysis Pipeline
Runs the entire SPI pipeline for a point source, given dates, energy bins and source variability.
Can be called directly for quick interactive session, or imported for automatized analysis. 

Main analysis steps:
- select data with dates, position and angle
- prepare data with spiselectscw and energy bounds
- create background model with obs_background.py module
- select variability parameters for source and bkg
(WIP
- run model fitting with spimodfit
- create source spectra with response)

"""

import pandas as pd
import numpy as np
import sys
import os
import subprocess
import shutil
import glob
import pickle
import socket
from datetime import datetime

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

from spibackground.obs_background import ScwTracerDB, ObsBkg, LiveTimeRev


class ObsSPI:
    """Pipeline for SPI observation analysis"""
    
    EVT_BIN_SIZE = {'SE':.5, 'PSD':.5, 'HE':1.}
    
    def __init__(self, main_dir, initial_dir=None, config_file='config.txt',
                 gnrl_cat_ext = 'GNRL-REFR-CAT', bg_idx_filename = 'output_bgmodel_conti_sep_idx.fits.gz',
                 spiselect_par_tpl_file = 'spiselectscw.template.par', spimodfit_par_tpl_file='spimodfit.template.par',
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
        self.data_dir = None
        self.scw_db_path = None
        self.gnrl_cat_path = None
        self.spi_cat_path = None
        self.all_revs_path = None
        self.templates_dir = None
        self.bkg_db_dir = None
        self.import_path_config(config_file)
        # default file names
        self.gnrl_cat_ext =  gnrl_cat_ext
        self.bg_idx_filename = bg_idx_filename
        self.spiselect_par_tpl_file =  spiselect_par_tpl_file
        self.spimodfit_par_tpl_file = spimodfit_par_tpl_file
        
        self.scw_tracer_db = ScwTracerDB(self.scw_db_path)
        self.obs_bkg = None
        self.livetime_rev = None
        self.bkg_dict = None

        # default query values
        self.update_default_dico = True
        self.initial_default_query={
            'src_dir' : '',
            'full_name' : 'Crab',
            'date_start' : '2003-01-01', 'date_end' : '2025-01-01', 'off_angle' : '15',
            'evt_type' : 'SE',
            'binning_type' : 'log',
            # TO DO: additional keywords for PSD
            'e_channels' : '20. 400.',
            'N_chan' : '20',
            'e_channels_bounds' : '20. 50. 200. 400.',
            'e_channels_scales' : '2 -20 2',
            'src_var_n' : '0', 'src_var_unit' : 'd', 'src_var_type' : 'n', 'src_max_angle' : '20.',
            'bkg_var_n' : '1', 'bkg_var_unit' : 'd', 'bkg_var_type' : 'i'
        }
        self.initial_default_spimodfit={k: self.initial_default_query[k] 
            for k in ['src_var_n', 'src_var_unit', 'src_var_type', 'src_max_angle', 'bkg_var_n', 'bkg_var_unit', 'bkg_var_type']
        }
        # make shallow copy of initial dico, modified and saved throughout the queries
        self.default_query_dico = self.initial_default_query.copy()

        # if recorded, last query values used as defaults
        self.import_last_query()
    
    def import_path_config(self, config_file='config.txt'):
        """use the config file to set path attributes
        this allows user to change paths without changing this code
        """
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
                    print(f'Loaded config: {variable_name} = {value}')
            

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

    def query(self, message, default=None, default_key=None):
        """ask for input. if empty, use default value.
        default_key can also be used instead of default to access value in default dico
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
            self.df_select = pd.read_csv('df_select.csv')
        else:
            self.df_select = self.scw_tracer_db.df_scw.loc[(self.scw_tracer_db.df_scw.DateStart < date_end) & (self.scw_tracer_db.df_scw.DateEnd > date_start)]
            print(f'Searching for pointings within {off_angle}° of source...')
            source_coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg)
            self.df_select['skycoord'] = self.df_select.apply(lambda x:SkyCoord(ra=x.RA_SCX*u.deg, dec=x.DEC_SCX*u.deg), axis=1)
            self.df_select['dist_angle'] = self.df_select['skycoord'].apply(lambda x:source_coord.separation(x).value)
            self.df_select = self.df_select[self.df_select.dist_angle < off_angle]
            print('Saving df...')
            self.df_select.to_csv('df_select.csv')
        
        self.N_point_select = len(self.df_select)
        self.unique_revs = self.df_select.REV.unique()
        self.N_unique_revs = len(self.unique_revs)
        print(f"Found {self.N_point_select} pointings, for {self.N_unique_revs} unique revolutions.")
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
        
        CMD = "/data1/ipp_afs_mirror/integral/software/local/spiselectscw/4.02/amd64_sles11_g++/spiselectscw"
        CFITSIO_TEMPLATES = "/data1/ipp_afs_mirror/integral/software/osa/osa-10.0/linux64_sw-10.0/templates"
        parfile_path = f'spiselectscw.{run_id}.par'
        
        if not os.path.isfile(parfile_path):
            print(f"Parameter file {parfile_path} doesn't exist. Exit!")
            return False
        
        os.environ['CFITSIO_INCLUDE_FILES'] = CFITSIO_TEMPLATES
        os.environ['PFILES'] = '.'
        
        if not os.path.isdir(run_id):
            os.mkdir(run_id)
        
        os.chdir(run_id)
        
        if not os.path.isdir('spi'):
            os.mkdir('spi')
        
        if not os.path.isfile('spi_off_det.fits'):
            os.symlink('/data1/ipp_afs_mirror/integral/software/local/spiselectscw/current/spi_off_det.fits',
                       'spi_off_det.fits')
        
        if not os.path.isfile('spi_gnrl_bti.fits'):
            os.symlink('/data1/ipp_afs_mirror/integral/data/ic/spi/lim/spi_gnrl_bti_0005.fits',
                       'spi_gnrl_bti.fits')
        
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
        
        logfile = 'spiselectscw.log'
        print(f"Running spiselectscw (RUN_ID: {run_id})...")
        
        try:
            with open(logfile, 'w') as log:
                log.write(f"{datetime.now()}\n")
                log.flush()
                
                result = subprocess.run(CMD, stdout=log, stderr=subprocess.STDOUT, text=True)
                
                log.write(f"\n{datetime.now()}\n")
            
            print(f"Command completed with exit status: {result.returncode}")
            print(f"Output logged to {logfile}")
            return result.returncode
            
        except Exception as e:
            print(f"Error running command: {e}")
            return False

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

            if not os.path.isfile('energies.txt'):
                self.write_energies_txt()

            if self.spiselectscw_return_code != 0:
                print(f'spiselectscw failed to complete (Error code {self.spiselectscw_return_code}).')
                print(f'Check log file {os.getcwd()}/spiselectscw.log')
                os.chdir(self.initial_dir)
                raise AssertionError()
    
    
    def make_spiselectscw_par_interactive(self):
        """create spiselectscw parameter file
        (with interactive input)
        """
        if os.path.isdir(self.ener_dir):
            rm_ener_dir = self.query(f"{self.ener_dir} dir already exists. Remove it? (y/n)", "y")
            if rm_ener_dir == 'y':
                shutil.rmtree(self.ener_dir, ignore_errors=True)
                # os.rmdir(self.ener_dir)
            else:
                self.make_spiselectscw_par(skip_spalready_exists=True)
                return
        
        self.make_spiselectscw_par(skip_spalready_exists=False)
    
    ########## Background ##########

    def process_background(self):
        """Process background for the observation using python background module"""
        
        scw_db_path = 'scw.fits.gz'
        self.obs_bkg = ObsBkg(f'{self.main_dir}/{self.src_dir}/{self.date_dir}/{self.ener_dir}', self.evt_type)
        self.livetime_rev = LiveTimeRev(self.bkg_db_dir+'/det_livetime_rev.fits', self.evt_type)
        scw_tracer_db = ScwTracerDB(scw_db_path)
        self.obs_bkg.load_tracer(scw_tracer_db)
        self.obs_bkg.normalize_tracer(self.livetime_rev)
        self.obs_bkg.init_rev_bkg_list(self.livetime_rev, self.bkg_db_dir)
        self.bkg_dict = self.obs_bkg.calc_bkg()
        self.obs_bkg.write_output_bkg()

        self.bkg_dir = self.obs_bkg.output_dir
    
    ########## Flux (spimodfit) ##########

    def run_spimodfit(self, run_id, clobber=False):
        """Execute the spimodfit command following the submit-spimodfit_v3.2_ga05us.sh script
        """
        
        CMD = "/data1/ipp_afs_mirror/integral/software/local/spimodfit/3.2/amd64_sles11_g++/spimodfit"
        CFITSIO_TEMPLATES = "/data1/ipp_afs_mirror/integral/software/osa/osa-10.0/linux64_sw-10.0/templates"
        parfile_path = f'spimodfit.{run_id}.par'
        
        if not os.path.isfile(parfile_path):
            print(f"Parameter file {parfile_path} does not exist.")
            return False
        
        subdir = run_id
        if os.path.exists(subdir):
            if clobber:
                print(f"Removing existing directory {subdir}...")
                shutil.rmtree(subdir, ignore_errors=True)
            else:
                print(f"Directory {subdir} exists. Please use clobber=True if you want to overwrite it")
                return False
        
        os.mkdir(subdir)
        os.chdir(subdir)
        
        shutil.copy2(f'../{parfile_path}', 'spimodfit.par')
        
        os.environ['CFITSIO_INCLUDE_FILES'] = CFITSIO_TEMPLATES
        os.environ['PFILES'] = '.'
        
        logfile = 'spimodfit.log'
        print(f"Running spimodfit (RUN_ID: {run_id})...")
        
        try:
            with open(logfile, 'w') as log:
                log.write(f"{socket.gethostname()} {datetime.now()}\n")
                log.write(f"Command: {CMD}\n")
                log.flush()
                
                result = subprocess.run(CMD, stdout=log, stderr=subprocess.STDOUT, text=True)
                
                log.write(f"\n{socket.gethostname()} {datetime.now()}\n")
            
            print(f"Command completed with exit status: {result.returncode}")
            print(f"Output logged to {logfile}")
            return result.returncode
            
        except Exception as e:
            print(f"Error running command: {e}")
            return False

    
    def make_spimodfit_par(self, src_var_n=0, src_var_unit='d', src_var_type='n', src_max_angle=20.,
                           bkg_var_n=1, bkg_var_unit='d', bkg_var_type='i',
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
            print('Creating spimodfit parameter file...')
            
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
source-cat-dol,s,h,"{self.spi_cat_path}",,,"input catalogue of sources "

# ----------- source variability parameters -----------
source_parameters_fit,i,h,1,0,1,"Sources fit parameter 1=yes" 
"""
            
            # Source variability parameter

            if src_var_unit == 'rev':

                spimodfit_par_str += f"""
source_var_coef,s,h,"{src_var_n} {src_var_unit} {src_var_type}",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes")
sources_zenith_angle,r,h,{src_max_angle},0,," Sources maximum zenithal angle"

    """
            else:
                spimodfit_par_str += f"""# VARIATION PER REVOLUTION
source_var_coef,s,h,"&{self.all_revs_path}[1] col=TIME_PERIGEE d n, 1435.41635 1659.46 3337.5 3799.66740 d n",,,"Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)"
sources_zenith_angle,r,h,{src_max_angle},0,," Sources maximum zenithal angle"

    """
            
            # Background variability parameter

            spimodfit_par_str += """
# ----------- background variability parameters -----------
collect_background_models,i,h,0,0,1,"Collect background components into one model (0/1)"
"""
            if bkg_var_unit=='rev':
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
            self.fit_dir = f'fit_{src_var_n}{src_var_unit}{src_var_type}_{bkg_var_n}{bkg_var_unit}{bkg_var_type}'
            output_par_file = f'spimodfit.{self.fit_dir}.par'
            with open(output_par_file, 'w') as f_out:
                f_out.write(combined_content)
            
            print(f'Parameter file created: {output_par_file}')
            self.spimodfit_return_code = self.run_spimodfit(self.fit_dir)

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
        print('*** Selecting spimodfit parameters')
        print('Initial default spimodfit parameters:\n', self.initial_default_spimodfit)
        std_spimodfit_query= self.query('Use standard spimodfit analysis? (y/n)', default='n')
        if std_spimodfit_query == 'y':
            self.make_spimodfit_par(**self.initial_default_spimodfit)

        else:
            src_max_angle = self.query('Max off-angle for catalog sources selection', default_key='src_max_angle')
            print('* Source variability: Time variability definition : number + rev(olution)/d(ays)/p(pointings) + i(ncrements)/n(nodes)')
            src_var_unit = self.query('Variability unit? (p=pointing, d=day, rev=revolution)', default_key='src_var_unit')
            src_var_n = self.query('Variability number? (0 or odd)', default_key='src_var_n')
            src_var_type = self.query('Variability type? (i(ncrements)/n(nodes))', default_key='src_var_type')

            print('* Background variability: Time variability definition : d(ays)/p(pointings) + i(ncrements)/n(nodes)')
            bkg_var_unit = self.query('Variability unit? (p=pointing, d=day, rev=revolution)', default_key='bkg_var_unit')
            bkg_var_n = self.query('Variability number? (>=0)', default_key='bkg_var_n')
            bkg_var_type = self.query('Variability type? (i(ncrements)/n(nodes))', default_key='bkg_var_type')

            self.make_spimodfit_par(src_var_n=src_var_n, src_var_unit=src_var_unit, src_var_type=src_var_type, src_max_angle=src_max_angle,
                           bkg_var_n=bkg_var_n, bkg_var_unit=bkg_var_unit, bkg_var_type=bkg_var_type
                           )
    
    ########## Response generation (spirmfgen) ##########
    def generate_response(self):
        pass

# def main():
if __name__ == '__main__':

    """Main pipeline execution with user interaction"""

    initial_dir= os.getcwd()
    print('*** SPI Observation Pipeline ***\n')
    
    obs = ObsSPI(
        # those paths should be changed:
        main_dir = initial_dir+'/obs',
        # main_dir = '/home/tbouchet/SPI_SOURCES/obs',
        initial_dir=initial_dir,
        testrun=False
        )
    
    print(f'Main observation directory is {obs.main_dir}\n')

    print('Tip: default values can be selected by simply pressing enter.\n')
    print('\n=== Source Selection ===\n')
    obs.setup_source_interactive()
    
    print('\n=== Observation Selection ===\n')
    obs.select_observations_interactive()
    
    print('\n=== Energy Binning ===\n')
    obs.setup_energies_interactive()
    
    print('\n=== Prepare data set with spiselectscw ===\n')
    obs.make_spiselectscw_par_interactive()
    
    print('\n=== Background Processing ===\n')
    obs.process_background()
    
    print('\n=== Flux extraction (spimodfit) ===\n')
    obs.make_spimodfit_par_interactive()

    print('\n=== Response matrix generation (spirmfgen) ===\n')
    obs.generate_response()
    # print('\n*** Pipeline completed ***\n')

    # back to original place
    os.chdir(initial_dir)

# if __name__ == '__main__':
#     main()
