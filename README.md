SPI-TAP — SPI Transient Analysis Pipeline

Runs the entire SPI pipeline for a point source, given dates, energy bins and source variability.
Can be called directly for quick interactive session, or imported for automatized analysis. 

# Requirements

The pipeline requires the following libraries:

numpy scipy pandas astropy ipython

# Main analysis steps
- select data with dates, position and angle
- prepare data with spiselectscw and energy bounds
- create background model with obs_background.py module
- select variability parameters for source and bkg
- run model fitting with spimodfit
- create source spectra with response (WIP)

# Quick interactive session

First, install the required libraries, preferentially in a clean conda environment,

```console
conda init
conda create -n "spitap" python=3.13
conda activate spitap
conda install ipython numpy scipy pandas astropy
```

Once activated, clone the spi-tap repo with,

```console
git clone https://github.com/tristanbouchet/spi-tap.git
cd spi-tap
```

The obs directory will contain by default all your analysis results. In it, there should be a config.txt file that contains the main paths to use for analysis, pointing to various locations in the ga05us server. You can check that they are correct and/or modify them (for instance the spi_cat_path).

If you do not wish to install spi-tap, you can simply run the main script. Using an interactive python session is recommanded for de-bugging:

```console
ipython
run run_spitap
```
and let the prompts guide you.

# Full install (WIP)

Another way is to install the pipeline in your environment so that it can be called or imported from anywhere. It also avoids wrong git manip that could over-write your previous analysis. The main drawback is that you need to re-install it whenever you want to modify the Python script.

First, install the pipeline, preferably in your dedicated conda environment
```console
python setup.py bdist_wheel
pip install dist/spitap-1.0-py3-none-any.whl
```

Next, you should create a main analysis directory in your location of choice. Avoid creating it in the spi-tap repo directory (i.e. where the original spi_obs.py script is).

```console
cp -r <your_spitap_dir>/obs <your_new_analysis_dir>
cd <your_new_analysis_dir>
```

Now you can run the interactive analysis with
```console
run_spi_tap
```

For automatized analysis, you can import spi-tap into your own script or jupyter-notebook with
```console
import spitap
```

# Spectra analysis

After running the pipeline for a source, they can be copied locally and analyzed there (using e.g. XSpec). Otherwise, a jupyter notebook is provided which shows an exemple of spectral analysis using the 3ML library. To install it, unset some env variables and use pip:

```console
unsetenv HEADAS PFILES LHEASOFT
pip install astromodels threeml
```
