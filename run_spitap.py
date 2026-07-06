"""
Main pipeline execution with user interaction
"""

import spitap.spi_obs as st
import os

if __name__ == '__main__':

    initial_dir= os.getcwd()
    print('*** SPI Observation Pipeline ***\n')
    
    obs = st.ObsSPI(
        # main_dir = initial_dir+'/obs',
        main_dir = '/home/tbouchet/SPI_SOURCES/obs',
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

