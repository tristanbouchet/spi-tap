"""Prepare a spimodfit-compatible source catalog from the LB catalog.

Builds/modifies the "SPI.-SRCL-CAT" extension of a spimodfit catalog by adding
flux columns from the LB catalog (L. Bouchet+2008) and 
variability flags (pointings or days), then writes the result to a new FITS file.
"""
import argparse

import numpy as np
from astropy.io import fits

# sources that require pointing variability below 50 keV (Bouchet+2008)

DEFAULT_VAR_SRC_DICO = {
    ('pointings', 1):
    [
    'A0535+32', 'Vela X-1', 'GX 301-2', '4U 1700-377', 'Sco X-1', 'Aql X-1', 'GRS 1915+105',
    'Cyg X-1', 'Cyg X-3', 'SWIFT J1753.5-0127', 'IGR J17464-3213', 'V0332+53',
    ],
    ('days', 1): [],
}



def _norm_name(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


class SMFCat:
    """Wraps a spimodfit source catalog, allowing incremental edits and saving at any point."""

    CAT_EXT = "SPI.-SRCL-CAT"
    LB_EXT = "SPICAT"

    def __init__(self, smf_cat_path, lb_cat_path):
        self.smf_cat_path = smf_cat_path
        self.lb_cat_path = lb_cat_path

        # spimodfit compatible catalog
        self.smf_cat_hdul = fits.open(smf_cat_path)
        # original LB catalog with flux sources (L. Bouchet+2008)
        self.lb_cat_hdul = fits.open(lb_cat_path, uint=False)
        self.lb_cat_table = self.lb_cat_hdul[self.LB_EXT].data

        self.cat_hdu = self.smf_cat_hdul[self.CAT_EXT]
        self.matched = None

    @property
    def cat_data(self):
        return self.cat_hdu.data

    def add_flux(self, crab_name="Crab"):
        """Add FLUX_LB and FLUX_CRAB_LB columns, normalizing FLUX_CRAB_LB by the Crab flux."""
        lb_names = np.array([_norm_name(n) for n in self.lb_cat_table["NAME"]])
        smf_names = np.array([_norm_name(n) for n in self.cat_data["NAME"]])

        lb_flux_col = self.lb_cat_hdul[self.LB_EXT].columns["FLUX"]
        flux_shape = self.lb_cat_table["FLUX"][0].shape
        flux_dtype = self.lb_cat_table["FLUX"].dtype
        zero_flux = np.zeros(flux_shape, dtype=flux_dtype)

        flux_by_name = {name: flux for name, flux in zip(lb_names, self.lb_cat_table["FLUX"])}
        smf_flux = np.array([flux_by_name.get(name, zero_flux) for name in smf_names], dtype=flux_dtype)
        self.matched = np.array([name in flux_by_name for name in smf_names])

        crab_name_norm = _norm_name(crab_name)
        crab_idx = np.where(lb_names == crab_name_norm)[0]
        if len(crab_idx) == 0:
            raise ValueError(f"Reference source not found in LB catalog: {crab_name}")
        crab_flux = self.lb_cat_table["FLUX"][int(crab_idx[0])].astype(np.float64)

        # Normalize SMF flux by Crab flux (element-wise)
        smf_flux_crab = np.divide(
            smf_flux.astype(np.float64),
            crab_flux,
            out=np.full_like(smf_flux, np.nan, dtype=np.float64),
            where=crab_flux != 0,
        )

        flux_crab_col = fits.Column(
            name="FLUX_CRAB_LB",
            format=lb_flux_col.format,
            dim=lb_flux_col.dim,
            array=smf_flux_crab,
        )
        new_flux_col = fits.Column(
            name="FLUX_LB", format=lb_flux_col.format,
            unit=lb_flux_col.unit, dim=lb_flux_col.dim, array=smf_flux,
        )

        base_cols = [col for col in self.cat_hdu.columns]
        self.cat_hdu = fits.BinTableHDU.from_columns(
            base_cols[:7] + [new_flux_col, flux_crab_col] + base_cols[7:],
            name=self.CAT_EXT,
        )
        return self

    def add_variability(self, var_src_dico=DEFAULT_VAR_SRC_DICO, new_var_pars_len=2 ** 11):
        """Resize VAR_PARS to new_var_pars_len and flag sources with the requested variability.

        var_src_dico maps a (var_type, var_n) tuple, e.g. ("pointings", 1), to the list of
        source names that should use that "constant, {var_type}, increments" variability model.
        """
        cat_data = self.cat_data
        cat_names = np.array([_norm_name(n) for n in cat_data["NAME"]])

        new_var_pars_table = np.zeros(
            shape=(cat_data["VAR_PARS"].shape[0], new_var_pars_len),
            dtype=cat_data["VAR_PARS"].dtype,
        )
        var_pars_col = self.cat_hdu.columns["VAR_PARS"]
        new_var_pars_col = fits.Column(
            name="VAR_PARS",
            format=f"{new_var_pars_len}{var_pars_col.format.lstrip('0123456789')}",
            unit=var_pars_col.unit,
            dim=f"({new_var_pars_len})",
            array=new_var_pars_table,
        )
        self.cat_hdu = fits.BinTableHDU.from_columns(
            [new_var_pars_col if col.name == "VAR_PARS" else col for col in self.cat_hdu.columns],
            name=self.CAT_EXT,
        )
        cat_data = self.cat_data

        def _fits_str(value, sample):
            if isinstance(sample, (bytes, np.bytes_)):
                return value.encode("utf-8")
            return value

        for (var_type, var_n), src_list in var_src_dico.items():
            if not src_list:
                continue
            is_var = np.isin(cat_names, list(set(src_list)))
            cat_data["VAR_MODL"][is_var] = _fits_str(
                f"constant, {var_type}, increments", cat_data["VAR_MODL"][0],
            )
            cat_data["VAR_PARS"][is_var, 0] = var_n
            cat_data["VAR_NPAR"][is_var] = int(1)  # always 1 for increments
        return self

    def add_new_sources(self, new_src_list, ref_src_name="Crab"):
        """Append new sources, copying unspecified column values from ref_src_name."""
        cat_hdu = self.cat_hdu
        cat_data = cat_hdu.data
        col_names = list(cat_data.names)
        col_name_set = set(col_names)

        ref_name_norm = _norm_name(ref_src_name)
        all_names_norm = np.array([_norm_name(v) for v in cat_data["NAME"]])
        ref_idx = np.where(all_names_norm == ref_name_norm)[0]
        if len(ref_idx) == 0:
            raise ValueError(f"Reference source not found: {ref_src_name}")
        ref_i = int(ref_idx[0])

        n_old = len(cat_data)
        n_new = len(new_src_list)
        merged_hdu = fits.BinTableHDU.from_columns(cat_hdu.columns, nrows=n_old + n_new, name=self.CAT_EXT)

        for c in col_names:
            merged_hdu.data[c][:n_old] = cat_data[c]

        added_names = []
        for i, src_dico in enumerate(new_src_list):
            dst = n_old + i

            for c in col_names:
                merged_hdu.data[c][dst] = cat_data[c][ref_i]

            unknown_keys = [k for k in src_dico.keys() if k not in col_name_set]
            if unknown_keys:
                raise KeyError(f"Unknown columns in source dict: {unknown_keys}")

            for k, v in src_dico.items():
                merged_hdu.data[k][dst] = v

            added_names.append(src_dico.get("NAME", _norm_name(cat_data["NAME"][ref_i])))

        self.cat_hdu = merged_hdu
        print(f"Added sources: {n_new}")
        print(np.array(added_names))
        return self

    def save_current_cat(self, out_path):
        """Write the catalog in its current state (whichever steps have been applied) to out_path."""
        new_hdus = [self.cat_hdu if hdu.name == self.CAT_EXT else hdu.copy() for hdu in self.smf_cat_hdul]
        fits.HDUList(new_hdus).writeto(out_path, overwrite=True)

        print(f"Written: {out_path}")
        if self.matched is not None:
            smf_names = np.array([_norm_name(n) for n in self.cat_data["NAME"]])
            print(f"Matched sources: {self.matched.sum()}/{len(smf_names)}")
            print("Included sources:")
            print(smf_names[self.matched])
            print("Missing sources (FLUX=0):")
            print(smf_names[~self.matched])
        return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smf-cat", default="/home/tbouchet/cat/spi_cat_LB_spimodfit_corr.fits",
                         help="Path to the spimodfit-compatible catalog FITS file.")
    parser.add_argument("--lb-cat", default="/home/tbouchet/cat/spi_cat_LB_corr.fits",
                         help="Path to the LB catalog FITS file (source of FLUX values).")
    parser.add_argument("--out", default="/home/tbouchet/spi-tap/cat/spi_cat_LB_var_point.fits",
                         help="Output path for the resulting catalog.")
    parser.add_argument("--crab-name", default="Crab", help="Reference source name used to normalize flux.")
    parser.add_argument("--var-pars-len", type=int, default=2 ** 11, help="New VAR_PARS array size.")
    parser.add_argument("--skip-variability", action="store_true", help="Skip the add_variability step.")
    parser.add_argument("--skip-flux", action="store_true", help="Skip the add_flux step.")
    args = parser.parse_args()

    smf_cat = SMFCat(args.smf_cat, args.lb_cat)
    if not args.skip_flux:
        smf_cat.add_flux(crab_name=args.crab_name)
    if not args.skip_variability:
        smf_cat.add_variability(new_var_pars_len=args.var_pars_len)
    smf_cat.save_current_cat(args.out)


if __name__ == "__main__":
    main()
