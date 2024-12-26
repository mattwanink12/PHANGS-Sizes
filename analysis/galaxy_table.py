"""
galaxy_table.py - Create a table showing the characteristics of the various galaxies
and the clusters in them

This takes the following parameters
- Path where this text file will be saved
- Oversampling factor for the psf
- The width of the psf snapshot
- The source of the psf stars
- All the completed catalogs
"""
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize
from astropy.io import fits
from astropy import table
from astropy import units as u

# ======================================================================================
#
# Load basic parameters and the catalogs
#
# ======================================================================================
output_name = Path(sys.argv[1])
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])
psf_source = sys.argv[4]
catalog = table.Table.read(sys.argv[5], format="ascii.ecsv")

# get some derived properties to use later
home_dir = Path(sys.argv[5]).parent
fields = np.unique(catalog["field"])
catalog["pixel_scale"] = catalog["r_eff_arcsec"] / catalog["r_eff_pixels"]
for item in catalog["pixel_scale"]:
    assert np.isclose(item, 39.62e-3, atol=0, rtol=1e-2)

# ======================================================================================
#
# significant figures in the distances table
#
# ======================================================================================
# each galaxy has a different number, so I need to be careful with how I present.
# for galaxies in Sabbi directly, I use the same sig figs as that paper lists. For
# galaxies with the mean I use 2 decimal places
distances_decimal_places = {
    "ic4247": (2, 1),
    "ic559": (1, 1),
    "ngc1313-e": (2, 2),
    "ngc1313-w": (2, 2),
    "ngc1433": (1, 1),
    "ngc1566": (1, 1),
    "ngc1705": (2, 2),
    "ngc3344": (1, 1),
    "ngc3351": (1, 1),
    "ngc3738": (2, 2),
    "ngc4242": (1, 1),
    "ngc4395-n": (2, 2),
    "ngc4395-s": (2, 2),
    "ngc4449": (2, 2),
    "ngc45": (1, 1),
    "ngc4656": (1, 1),
    "ngc5194-ngc5195-mosaic": (2, 2),
    "ngc5238": (2, 2),
    "ngc5253": (2, 2),
    "ngc5474": (1, 1),
    "ngc5477": (1, 1),
    "ngc628-c": (1, 1),
    "ngc628-e": (1, 1),
    "ngc6503": (1, 1),
    "ngc7793-e": (2, 2),
    "ngc7793-w": (2, 2),
    "ugc1249": (1, 1),
    "ugc4305": (2, 2),
    "ugc4459": (2, 2),
    "ugc5139": (2, 2),
    "ugc685": (2, 2),
    "ugc695": (1, 1),
    "ugc7408": (1, 1),
    "ugca281": (2, 2),
}

# ======================================================================================
#
# Functions to calculate the psf effective radius
#
# ======================================================================================
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_psf_reff(field_dir, distance_mpc, pixel_scale):
    # pixel scale is in arcseconds per pixel
    psf_name = (
        f"psf_"
        f"{psf_source}_stars_"
        f"{psf_width}_pixels_"
        f"{oversampling_factor}x_oversampled.fits"
    )

    psf = fits.open(field_dir / "size" / psf_name)["PRIMARY"].data
    # the center is the brightest pixel of the image
    x_cen, y_cen = np.unravel_index(np.argmax(psf), psf.shape)
    total = np.sum(psf)
    half_light = total / 2.0
    # then go through all the pixel values to determine the distance from the center.
    # Since we're using the center pixel (on an integer value), different pixels
    # will have identical distances. Add all the pixels at the same distance together
    unique_values = dict()
    for x in range(psf.shape[1]):
        for y in range(psf.shape[0]):
            # need to include the oversampling factor in the distance
            radius = distance(x, y, x_cen, y_cen) / oversampling_factor
            if radius not in unique_values:
                unique_values[radius] = 0
            unique_values[radius] += psf[y][x]

    # Then sort the radii
    radii, values = [], []
    for r, v in unique_values.items():
        radii.append(r)
        values.append(v)
    idxs_sort = np.argsort(radii)
    sorted_radii = np.array(radii)[idxs_sort]
    sorted_values = np.array(values)[idxs_sort]

    # then turn this into cumulative light
    cumulative_light = np.cumsum(sorted_values)
    # then to find where we cross 1/2, create an interpolation object
    cum_interp = interpolate.interp1d(x=sorted_radii, y=cumulative_light, kind="linear")
    # find where it reaches half
    def to_minimize(r_half):
        return abs(cum_interp(r_half) - half_light)

    psf_size_pixels = optimize.minimize(to_minimize, x0=1.5, bounds=[[0, 5]]).x[0]

    # then convert to pc.
    psf_size_radians = (psf_size_pixels * pixel_scale * u.arcsec).to("radian").value
    return psf_size_radians * distance_mpc * 1e6


# ======================================================================================
#
# Then we print this all as a nicely formatted latex table
#
# ======================================================================================
def format_field_name(raw_name):
    name = raw_name.lower()
    # check one edge case
    if name == "ngc5194-ngc5195-mosaic":
        return "NGC 5194/NGC 5195"
    # then format names nicely for the table
    if "ngc" in name:
        return name.replace("ngc", "NGC ")
    elif "ugca" in name:
        return name.replace("ugca", "UGCA ")
    elif "ugc" in name:
        return name.replace("ugc", "UGC ")
    elif "ic" in name:
        return name.replace("ic", "IC ")
    elif "eso" in name:
        return name.replace("eso", "ESO ")
    else:
        raise ValueError(f"{name} wasn't handled properly.")


def get_iqr_string(cat):
    mask = cat["reliable_radius"]
    r_eff = cat["r_eff_pc"][mask]
    values = np.percentile(r_eff, [25, 50, 75])
    return f"{values[0]:.2f} -- {values[1]:.2f} -- {values[2]:.2f}"


def write_field(out_file, field, galaxy=None):
    cat = catalog[catalog["field"] == field]
    # for NGC5194/NGC5195 we'll want to split the field. Otherwise don't do this
    if galaxy is not None:
        cat = cat[cat["galaxy"] == galaxy]
        name = format_field_name(galaxy)
    else:
        name = format_field_name(field)

    n = len(cat)
    # row is any row from that galaxy or field. Galaxy data is the same
    row = cat[0]
    mass_str = f"{np.log10(row['galaxy_stellar_mass']):.2f}"
    ssfr_str = f"{np.log10(row['galaxy_ssfr']):.2f}"
    dist = row["galaxy_distance_mpc"]
    dist_err = row["galaxy_distance_mpc_err"]
    dist_decimals, err_decimals = distances_decimal_places[field]
    dist_str = f"{dist:.{dist_decimals}f} $\pm$ {dist_err:.{err_decimals}f}"
    psf_size = measure_psf_reff(home_dir / "data" / field, dist, row["pixel_scale"])
    iqr_str = get_iqr_string(cat)

    out_file.write(
        f"\t\t{name} & "
        f"{n} & "
        f"{mass_str} & "
        f"{ssfr_str} & "
        f"{dist_str} & "
        f"{psf_size:.2f} & "
        f"{iqr_str} "
        f"\\\\ \n"
    )


# saving for the fancy multirow, in case I ever want to do that again.
# out_file.write(
#     f"\t\tNGC 5194 & "
#     f"{gal_data['NGC 5194']['n']} & "
#     f"{gal_data['NGC 5194']['mass_str']} & "
#     f"{gal_data['NGC 5194']['ssfr_str']} & "
#     # "\multirow{2}{*}{" + dist_str + "} & "
#     # "\multirow{2}{*}{" + f"{this_psf_size:.2f}" + "} & "
#     f"{dist_str} & "
#     f"{this_psf_size:.2f} & "
#     f"{gal_data['NGC 5194']['iqr_str']} "
#     f"\\\\ \n"
# )
#
# out_file.write(
#     f"\t\tNGC 5195 & "
#     f"{gal_data['NGC 5195']['n']} & "
#     f"{gal_data['NGC 5195']['mass_str']} & "
#     f"{gal_data['NGC 5195']['ssfr_str']} & "
#     # "& "
#     # "& "
#     f"{dist_str} & "
#     f"{this_psf_size:.2f} & "
#     f"{gal_data['NGC 5195']['iqr_str']} "
#     f"\\\\ \n"
# )


with open(output_name, "w") as out_file:
    out_file.write("\t\\begin{tabular}{lrccccc}\n")
    out_file.write("\t\t\\toprule\n")
    out_file.write(
        "\t\tLEGUS Field & "
        "N & "
        "log $M_\star$ ($\Msun$) & "
        "log sSFR (yr$^{-1}$) & "
        "Distance (Mpc) & "
        "PSF size (pc) & "
        "Cluster $\\reff$: 25--50--75th percentiles \\\\ \n"
    )
    out_file.write("\t\t\midrule\n")
    for field in fields:
        # NGC 5194 and 5195 are in the same field, so they need to be handled separately
        # in the table.
        if field == "ngc5194-ngc5195-mosaic":
            write_field(out_file, field, "ngc5194")
            write_field(out_file, field, "ngc5195")
        else:
            write_field(out_file, field)

    out_file.write("\t\t\midrule\n")
    # get the total values
    total_n = len(catalog)
    total_iqr = get_iqr_string(catalog)
    out_file.write(f"\t\tTotal & {total_n} {'& -- '*4} & {total_iqr} \\\\ \n")

    out_file.write("\t\t\\bottomrule\n")
    out_file.write("\t\end{tabular}\n")
