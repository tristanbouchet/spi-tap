"""
Main pipeline execution with user interaction
"""

import argparse
import os
import spitap.spi_obs as st

# for code development/debugging
import importlib
importlib.reload(st)

RED = "\033[31m"
RESET = "\033[0m"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the SPI transient analysis pipeline')
    parser.add_argument(
        '--obsdir',
        default=os.getcwd(),
        help='Main observation directory (defaults to the current working directory)'
    )
    args = parser.parse_args()

    main_dir = args.obsdir
    initial_dir = os.getcwd()
    initial_env = os.environ.copy()

    # TO DO: make better logo when launching pipeline
    print('*********************************************')
    print('****** SPI Transient Analysis Pipeline ******')
    print('*********************************************\n')
    
    obs = st.ObsSPI(
        # main_dir = initial_dir+'/obs',
        main_dir = main_dir,
        initial_dir=initial_dir,
        initial_env=initial_env,
        testrun=False
        )
    
    print(f'\nMain observation directory:\n{obs.main_dir}\n')

    print('Tip: default values can be selected by only pressing enter.\n')
    try:
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

        print('\n=== Select FOV sources ===\n')
        obs.select_sources_interactive()
        
        print('\n=== Flux extraction (spimodfit) ===\n')
        obs.select_src_var_interactive()
        obs.make_spimodfit_par_interactive()

        print('\n=== Results from spimodfit ===\n')
        obs.analyze_spimodfit(verbose=True)

        print('\n=== Response matrix generation (spirmfgen) ===\n')
        obs.generate_response()
        
        print('')
        print('********************************')
        print('****** Pipeline completed ******')
        print('********************************\n')
        print(f'Spectra available at {os.getcwd()}')

    except Exception as e:
        print(f'{RED}An error occured during the pipeline !{RESET}')
        print(e)

    print('\n*** Reverting to original directory and env ***')
    # go back to initial dir and reset env
    os.environ.clear()
    os.environ.update(initial_env)
    os.chdir(initial_dir)
    

