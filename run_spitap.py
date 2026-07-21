"""
Main pipeline execution with user interaction
"""

import spitap.spi_obs as st
import os

if __name__ == '__main__':

    initial_dir= os.getcwd()
    initial_env = os.environ.copy()

    # TO DO: make better logo when launching pipeline
    print('***************************************')
    print('*** SPI Transient Analysis Pipeline ***\n')
    print('***************************************')
    
    obs = st.ObsSPI(
        # main_dir = initial_dir+'/obs',
        main_dir = '/home/tbouchet/SPI_SOURCES/obs',
        initial_dir=initial_dir,
        initial_env=initial_env,
        testrun=False
        )
    
    print(f'\nMain observation directory:\n{obs.main_dir}\n')

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
    print('\n=== Results from spimodfit ===\n')
    obs.analyze_spimodfit(verbose=True)

    print('\n=== Response matrix generation (spirmfgen) ===\n')
    obs.generate_response()

    print('\n****** Pipeline completed ******\n')
    print(f'Spectra available at {os.getcwd()}')

    # go back to initial dir and reset env
    os.environ.clear()
    os.environ.update(initial_env)
    os.chdir(initial_dir)
    

