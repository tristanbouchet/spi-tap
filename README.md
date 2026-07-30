SPI-TAP — SPI Transient Analysis Pipeline

This code allows to run the INTEGRAL/SPI analysis on the AG Siegert ga05us server in Würzburg.

It runs the entire SPI pipeline for a point source, given dates, energy bins and source variability.
It can be called directly for quick interactive session, or imported for automatized analysis.

# Requirements

The pipeline requires the following libraries:

numpy scipy pandas astropy ipython

# Main analysis steps
- Select data with dates, position and angle
- Prepare data with spiselectscw and energy bounds
- Create background model
- Select variability parameters for sources and bkg
- Run model fitting with spimodfit and create responses
- Analyze results (spectral fit, residuals, light-curves)

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

The obsdir argument gives the path to the analysis directory with all your results. In it, there should be a config.txt file that contains the main paths to use for analysis, pointing to various locations in the ga05us server. You can check that they are correct and/or modify them (for instance the spi_cat_path). An example of such directory is given in the repo, called "obs".

If you do not wish to install spi-tap, you can simply run the main script. Using an interactive python session is recommanded for de-bugging:

```console
ipython
run run_spitap --obsdir <path_to_analysis>
```
and let the prompts guide you.

For automatized runs, use the auto_spitab.ipynb notebook, where all the run parameters can be stored in a dictionary.

<!-- # Full install (WIP)

Another way is to install the pipeline in your environment so that it can be called or imported from anywhere. It also avoids wrong git manip that could over-write your previous analysis. The main drawback is that you need to re-install it whenever you want to modify the Python script.

First, install the pipeline, preferably in your dedicated conda environment
```console
python setup.py bdist_wheel
pip install dist/spitap-1.0-py3-none-any.whl
```

For quick spectral fitting, also install lmfit (https://github.com/lmfit/lmfit-py)
```console
conda install -c conda-forge lmfit
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
``` -->

# Spectra analysis

After running the pipeline, a simple analysis using the spec_fit module (included in SPI-TAP) based on the lmfit library allows to fit the spectrum with simple models (powerlaw, cut-off exponential, ...). An example is shown in the fit_spectra.ipynb.

For a more detailed analysis, fit_spectra_3ML.ipynb contains an exemple of spectral analysis using the 3ML library. To install it, unset some env variables and use pip:

```console
unsetenv HEADAS PFILES LHEASOFT
pip install astromodels threeml
```
