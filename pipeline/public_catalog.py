"""
public_catalog.py

Format the catalog nicely and make it fit for public consumption.

Takes the following command line arguments:
- Name to save the output catalog as
- All the catalogs for individual galaxies
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
from astropy import units as u

import utils

# Get the input arguments
output_table = Path(sys.argv[1])
image_band = sys.argv[2]
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[3:]] # current_catalogs (for whatever band you are running)
for catalog in catalogs:
    catalog["Band"] = image_band

data_PHANGS = Path("data_PHANGS")

prev_bands = []
for galaxy in data_PHANGS.iterdir():
    if galaxy.name == ".DS_Store":
        continue
    galaxy_name = galaxy.name
    
    # Hubble band search below
    for band in ["f438w", "f275w", "f336w", "f555w", "f814w"]:
        band_dir = data_PHANGS/Path(f"{galaxy_name}/size/{band}")
        if band_dir.exists():
            band_catalog = band_dir/Path("final_catalog_final_30_pixels_psf_my_stars_15_pixels_2x_oversampled.txt")
            band_table = table.Table.read(band_catalog, format="ascii.ecsv")
            band_table["Band"] = band
            prev_bands.append(band_table)

# go through problems directory for old band data
prob_dir = Path("problems")
for band in prob_dir.iterdir():
    if band.name == image_band:
        cur_band_dir = prob_dir/Path(f"{band.name}")
        
        for galaxy in cur_band_dir.iterdir():
            if galaxy.name == ".DS_Store":
                continue
            galaxy_name = galaxy.name
    
            # Hubble band search below
            for old_band in ["f438w", "f275w", "f336w", "f555w", "f814w"]:
                band_dir = cur_band_dir/Path(f"{galaxy_name}/size/{old_band}")
                if band_dir.exists():
                    band_catalog = band_dir/Path("final_catalog_final_30_pixels_psf_my_stars_15_pixels_2x_oversampled.txt")
                    band_table = table.Table.read(band_catalog, format="ascii.ecsv")
                    band_table["Band"] = old_band
                    prev_bands.append(band_table)
        

# ======================================================================================
#
# handling class
#
# ======================================================================================
"""
# this is ugly, sorry
for cat in catalogs:
    # manually do the sorting of classes depending on the field.
    # `pipeline/format_catalogs.py` was referenced to get this right.
    assert len(np.unique(cat["field"])) == 1
    field = cat["field"][0]

    # set dummy column to make sure the string is long enough
    cat["morphology_class_source"] = " " * 20

    if field == "ngc5194-ngc5195-mosaic":
        # Here we use the class_mode_human, but then supplement it with the ML
        # classification for ones that weren't classified by humans
        mask_ml = cat["class_mode_human"] == 0
        mask_h = ~mask_ml

        # then fill in appropriately. First use dummy for class, will fill in
        cat["morphology_class"] = -99
        cat["morphology_class"][mask_h] = cat["class_mode_human"][mask_h]
        cat["morphology_class"][mask_ml] = cat["class_ml"][mask_ml]
        cat["morphology_class_source"][mask_h] = "human_mode"
        cat["morphology_class_source"][mask_ml] = "ml"

        # check my work
        mask_ml_check = cat["morphology_class_source"] == "ml"
        assert np.array_equal(np.unique(cat["class_mode_human"][mask_ml_check]), [0])

    # note that NGC4449 is another case where we originally selected ML clusters, but
    # now it only has human selected clusters. We threw out the ML clusters in derived
    # properties. Double check this
    elif field == "ngc4449":
        for c in np.unique(cat["class_whitmore"]):
            assert c in [1, 2]
        cat.rename_column("class_whitmore", "morphology_class")
        cat["morphology_class_source"] = "human_mode"

    #elif field == "ngc1566":
        # Here we simply use the hybrid method, as the documentation says it is the one
        # to use
        #cat.rename_column("class_hybrid", "morphology_class")
        #cat["morphology_class_source"] = "hybrid"

    else:  # normal catalogs
        if "PHANGS_CLUSTER_CLASS_HUMAN" in cat.colnames:
            class_col = "PHANGS_CLUSTER_CLASS_HUMAN" #"class_mode_human"
        #elif "class_linden_whitmore" in cat.colnames:
            #class_col = "class_linden_whitmore"
        #elif "class_whitmore" in cat.colnames:
            #class_col = "class_whitmore"
        #else:
            #raise ValueError(f"No class found for {field}")

        cat.rename_column(class_col, "morphology_class")
        cat["morphology_class_source"] = "human_mode"



# validate what I've done with the classes
assert np.array_equal(np.unique(catalog["morphology_class"]), [1, 2])
#assert np.array_equal(
    #np.unique(catalog["morphology_class_source"]),
    #["human_mode", "hybrid", "ml"],
#)
"""

# then stack them together in one master catalog
cur_catalog = table.vstack(catalogs, join_type="inner")
prev_catalog = table.vstack(prev_bands, join_type="inner")
catalog = table.vstack([cur_catalog, prev_catalog], join_type="inner")
# ======================================================================================
#
# galaxy data
#
# ======================================================================================
# then get the stellar mass, SFR, and galaxy distance
home_dir = Path(__file__).parent.parent
galaxy_table = table.Table.read(
    home_dir / "pipeline" / "lee_etal_2022_table_1.txt",
    format="ascii.commented_header"
)
# read the Calzetti table
gal_mass = dict()
gal_sfr = dict()
gal_type = dict()
gal_ra = dict()
gal_dec = dict()
for row in galaxy_table:
    name = row["galaxy"].lower()
    gal_mass[name] = 10**row["log_M"]
    gal_sfr[name] = row["SFR_tot"]
    gal_type[name] = row["T_type"]
    gal_ra[name] = row["RA"]
    gal_dec[name] = row["Dec"]

# set dummy quantities
catalog["galaxy_distance_mpc"] = -99.9
catalog["galaxy_distance_mpc_err"] = -99.9
catalog["galaxy_stellar_mass"] = -99.9
catalog["galaxy_sfr"] = -99.9
catalog["galaxy_type"] = -99.9
catalog["galactocentric_distance_arcseconds"] = -99.9
catalog["galactocentric_distance_pc"] = -99.9
catalog["galactocentric_distance_pc_err"] = -99.9

def distance_in_degrees(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * u.degree
    
def arcsec_to_pc_with_errors(arcsec, dist, dist_err):
    """
    :param arcsec: Size in arcseconds
    :param dist: Distance to galaxy
    :param dist_err: Error in distance to galaxy
    :return: Galactocentric radius and radius error in parsecs
    """
    radians = (arcsec * u.arcsec).to("radian").value
    parsecs = radians * dist.to("pc").value

    # then the fractional error is added in quadrature to get the resulting
    # fractional error, assuming the distance error is symmetric
    frac_err_dist = dist_err / dist

    #frac_err_arcsec_up = arcsec_error_up / arcsec
    #frac_err_arcsec_down = arcsec_error_down / arcsec

    #frac_err_tot_up = np.sqrt(frac_err_dist ** 2 + frac_err_arcsec_up ** 2)
    #frac_err_tot_down = np.sqrt(frac_err_dist ** 2 + frac_err_arcsec_down ** 2)

    err_pc = parsecs * frac_err_dist

    return parsecs, err_pc

# then add these quantities for all columns
for row in catalog:
    # get the field and galaxy of this cluster
    field = row["field"]
    galaxy = row["galaxy"]

    # then get the needed quantities and store them
    dist = utils.distance(home_dir / "data_PHANGS" / field)
    dist_err = utils.distance_error(home_dir / "data_PHANGS" / field)
    row["galaxy_distance_mpc"] = dist.to("Mpc").value
    row["galaxy_distance_mpc_err"] = dist_err.to("Mpc").value
    row["galaxy_stellar_mass"] = gal_mass[galaxy]
    row["galaxy_sfr"] = gal_sfr[galaxy]
    row["galaxy_type"] = gal_type[galaxy]
    
    arcsec_dist_center = distance_in_degrees(row["PHANGS_RA"], row["PHANGS_DEC"], gal_ra[galaxy], gal_dec[galaxy]).to(u.arcsec).value
    pc_dist_center, pc_dist_center_err = arcsec_to_pc_with_errors(arcsec_dist_center, dist, dist_err)
    
    row["galactocentric_distance_arcseconds"] = arcsec_dist_center
    row["galactocentric_distance_pc"] = pc_dist_center
    row["galactocentric_distance_pc_err"] = pc_dist_center_err
    
    #row["galaxy_RA"] = gal_ra[galaxy]
    #row["galaxy_DEC"] = gal_dec[galaxy]
    
    
    
    

# get age in years
catalog["SEDfix_age"] *= 1e6
# then calculate specific star formation rate
catalog["galaxy_ssfr"] = catalog["galaxy_sfr"] / catalog["galaxy_stellar_mass"]

# ======================================================================================
#
# a bit of into about the sample
#
# ======================================================================================
n_r = np.sum(catalog["reliable_radius"])
n_rm = np.sum(np.logical_and(catalog["reliable_mass"], catalog["reliable_radius"]))
print(f"{len(catalog)} total clusters")
print(f"{n_r} clusters have reliable radii")
print(f"{n_rm} of those have reliable mass")

print(f"{len(np.unique(catalog['field']))} different fields")
# for field in np.unique(catalog["field"]):
#     print(f"\t- {field}")
print(f"{len(np.unique(catalog['galaxy']))} different galaxies")
# for gal in np.unique(catalog["galaxy"]):
#     print(f"\t- {gal}")

# ======================================================================================
#
# Formatting the table
#
# ======================================================================================
# delete a few quantities that I calculated for debugging or testing that are not needed
catalog.remove_columns(
    [
        "estimated_local_background_diff_sigma",
        "fit_rms",
        "x_pix_snapshot_oversampled",
        "y_pix_snapshot_oversampled",
        "x_pix_snapshot_oversampled_e-",
        "x_pix_snapshot_oversampled_e+",
        "y_pix_snapshot_oversampled_e-",
        "y_pix_snapshot_oversampled_e+",
        #"PHANGS_F275W_VEGA_TOT", #"mag_F275W",
        #"PHANGS_F275W_VEGA_TOT_ERR", #"photoerr_F275W",
        #"PHANGS_F336W_VEGA_TOT", #"mag_F336W",
        #"PHANGS_F336W_VEGA_TOT_ERR", #"photoerr_F336W",
        #"PHANGS_F438W_VEGA_TOT",
        #"PHANGS_F438W_VEGA_TOT_ERR",
        #"PHANGS_F555W_VEGA_TOT",
        #"PHANGS_F555W_VEGA_TOT_ERR",
        #"PHANGS_F814W_VEGA_TOT", #"mag_F814W",
        #"PHANGS_F814W_VEGA_TOT_ERR", #"photoerr_F814W",
        #"PHANGS_CI", #"CI",
        #"PHANGS_EBV_MINCHISQ", #"E(B-V)",
        #"PHANGS_EBV_MINCHISQ_ERR",
        #"E(B-V)_max",
        #"E(B-V)_min",
        #"PHANGS_F275W_mJy_TOT", #"chi_2_F265W",
        #"PHANGS_F275W_mJy_TOT_ERR",
        #"PHANGS_F336W_mJy_TOT", #"chi_2_F336W",
        #"PHANGS_F336W_mJy_TOT_ERR",
        #"PHANGS_F438W_mJy_TOT",
        #"PHANGS_F438W_mJy_TOT_ERR",
        #"PHANGS_F555W_mJy_TOT",
        #"PHANGS_F555W_mJy_TOT_ERR",
        #"PHANGS_F814W_mJy_TOT", #"chi_2_F814W",
        #"PHANGS_F814W_mJy_TOT_ERR",
        #"PHANGS_REDUCED_MINCHISQ", #"chi_2_reduced",
        #"PHANGS_CLUSTER_CLASS_ML_VGG",
        #"PHANGS_CLUSTER_CLASS_ML_VGG_QUAL",
        #"INDEX",
        #"NO_DETECTION_FLAG",
        #"N_filters",
        #"Q_probability",
        "log_luminosity",
        "log_luminosity_e-",
        "log_luminosity_e+",
        "dx_from_snap_center",
        "dy_from_snap_center",
        "pixel_scale",
    ]
)

# rename a few columns
catalog.rename_column("profile_diff_reff", "fit_quality_metric")
#catalog.rename_column("x_pix_single", "x_legus")
#catalog.rename_column("y_pix_single", "y_legus")

# set the order for the leftover columns
"""
new_col_order = [
    "field",
    "ID_PHANGS_CLUSTERS", #"ID",
    "galaxy",
    "galaxy_distance_mpc",
    "galaxy_distance_mpc_err",
    "galaxy_stellar_mass",
    "galaxy_sfr",
    "galaxy_ssfr",
    "PHANGS_RA", #"RA",
    "PHANGS_DEC", #"Dec",
    "PHANGS_X", #"x_legus",
    "PHANGS_Y", #"y_legus",
    "morphology_class",
    "morphology_class_source",
    "PHANGS_AGE_MINCHISQ", #"age_yr",
    #"age_yr_min",
    #"age_yr_max",
    "PHANGS_AGE_MINCHISQ_ERR",
    "PHANGS_MASS_MINCHISQ", #"mass_msun",
    #"mass_msun_min",
    #"mass_msun_max",
    "PHANGS_MASS_MINCHISQ_ERR",
    "x",
    "x_e-",
    "x_e+",
    "y",
    "y_e-",
    "y_e+",
    "mu_0",
    "mu_0_e-",
    "mu_0_e+",
    "scale_radius_pixels",
    "scale_radius_pixels_e-",
    "scale_radius_pixels_e+",
    "axis_ratio",
    "axis_ratio_e-",
    "axis_ratio_e+",
    "position_angle",
    "position_angle_e-",
    "position_angle_e+",
    "power_law_slope",
    "power_law_slope_e-",
    "power_law_slope_e+",
    "local_background",
    "local_background_e-",
    "local_background_e+",
    "num_bootstrap_iterations",
    "radius_fit_failure",
    "fit_quality_metric",
    "reliable_radius",
    "reliable_mass",
    "r_eff_pixels",
    "r_eff_pixels_e-",
    "r_eff_pixels_e+",
    "r_eff_arcsec",
    "r_eff_arcsec_e-",
    "r_eff_arcsec_e+",
    "r_eff_pc",
    "r_eff_pc_e-",
    "r_eff_pc_e+",
    #"crossing_time_yr",
    #"crossing_time_log_err",
    #"density",
    #"density_log_err",
    #"surface_density",
    #"surface_density_log_err",
]

# check that I got all columns listed
#for i, name in enumerate(catalog.colnames):
    #if name not in new_col_order:
        #print(name)
#print(len(new_col_order), len(catalog.colnames))
assert len(new_col_order) == len(catalog.colnames)
assert sorted(new_col_order) == sorted(catalog.colnames)

# then apply this order
catalog = catalog[new_col_order]
"""

# ======================================================================================
#
# Catalog validation
#
# ======================================================================================
print(catalog)
# validate that there aren't nans or infinities where they don't belong
for col in catalog.colnames:
    try:
        assert np.sum(np.isnan(catalog[col])) == 0
        assert np.sum(np.isinf(catalog[col])) == 0
        assert np.sum(np.isneginf(catalog[col])) == 0
    except TypeError:  # can't check if strings are nans
        continue
    except AssertionError:
        # density and crossing time have nans where the mass is bad
        #assert "density" in col or "crossing_time" in col
        # check that there are no nans where the mass is good
        mask_good_mass = catalog["reliable_mass"]
        #print(np.sum(np.isnan(catalog[col][mask_good_mass])))
        #print(catalog[col,"SEDfix_mass"][np.where(np.isnan(catalog[col][mask_good_mass]))])
        # assert np.sum(np.isnan(catalog[col][mask_good_mass])) == 0
        assert np.sum(np.isinf(catalog[col][mask_good_mass])) == 0
        assert np.sum(np.isneginf(catalog[col][mask_good_mass])) == 0

# validate that all clusters from the same galaxy have the same galaxy properties
for gal in np.unique(catalog["galaxy"]):
    mask = catalog["galaxy"] == gal
    assert len(np.unique(catalog["galaxy_distance_mpc"][mask])) == 1
    assert len(np.unique(catalog["galaxy_distance_mpc_err"][mask])) == 1
    assert len(np.unique(catalog["galaxy_stellar_mass"][mask])) == 1
    assert len(np.unique(catalog["galaxy_sfr"][mask])) == 1
    assert len(np.unique(catalog["galaxy_ssfr"][mask])) == 1
# double check the field distances too, since those are the same per field
for field in np.unique(catalog["field"]):
    mask = catalog["field"] == field
    assert len(np.unique(catalog["galaxy_distance_mpc"][mask])) == 1
    assert len(np.unique(catalog["galaxy_distance_mpc_err"][mask])) == 1

# verify that errors are always non-negative
for col in catalog.colnames:
    if col.endswith("_e-") or col.endswith("_e+"):
        assert np.min(catalog[col]) >= 0

# ======================================================================================
#
# write the catalog!
#
# ======================================================================================
catalog.write(output_table, format="ascii.ecsv")
