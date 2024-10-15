from pathlib import Path
import sys
import numpy as np
from astropy.table import Table, join
from astropy.io.fits import getdata

def find_catalogs(home_dir):
    """
    Find the name of the base catalogs names. We need a function for this because we
    don't know whether it has ACS and WFC3 in the filename of just one.

    :param home_dir: Directory to search for catalogs
    :type home_dir: Path
    :return: A boolean to say whether the LEGUS catalog was found, a string representing the name of the LEGUS catalog, and a string representing the name of the PHANGS catalog.
    :rtype: tuple
    """
    galaxy_name = home_dir.name
    
    # Reformat some names to conform to the length structure of other galaxy names
    if galaxy_name in ["ngc628c", "ngc628e", "ngc685"]:
        galaxy_name = galaxy_name[0:3] + "0" + galaxy_name[3:]
    
    cluster_dir = home_dir/Path("cluster")
    legusSet = False
    phangsSet = False
    for item in cluster_dir.iterdir():
        if not item.is_file():
            continue
            
        filename = item.name
        # see if it starts and ends with what the catalog should be. We don't know what
        # instruments make up the catalog data, so we leave that segment of the name out
        # EX: PHANGS_HST_cluster_catalog_dr_4_cat_release_2_ngc1433_ml_class12_addlegusmatch.fits
        if filename.startswith("PHANGS_HST_cluster_catalog_dr_4_cat_release_2_") and filename.endswith(
            f"{galaxy_name}_human_class12_addlegusmatch.fits"
        ):
            legusCatalog = item
            legusSet = True

            
        elif filename.startswith("hlsp_phangs-cat_hst") and filename.endswith(f"{galaxy_name}_multi_v1_sed-machine-cluster-class12.fits"):
            phangsCatalog = item
            phangsSet = True
            
            
    if (phangsSet):
        if (legusSet):
            return legusSet, legusCatalog, phangsCatalog
        
        return legusSet, "", phangsCatalog
            
            
    # if we got here, we have an error.
    raise FileNotFoundError(f"No catalog found in {home_dir}")

# Start by getting the catalog files
final_catalog = Path(sys.argv[1])
home_dir = final_catalog.parent.parent
legus_exists, legus_catalog_name, phangs_catalog_name = find_catalogs(home_dir)

# Read PHANGS catalog and narrow down only to list of columns we care about
# If you would prefer to keep more columns for your analysis, edit the column selection line to the slice of your choosing
phangs_catalog = Table.read(phangs_catalog_name)
phangs_catalog = phangs_catalog['ID_PHANGS_CLUSTER', 'PHANGS_X', 'PHANGS_Y', 'PHANGS_RA', 'PHANGS_DEC', 'PHANGS_AGE_MINCHISQ', 'PHANGS_AGE_MINCHISQ_ERR', 'PHANGS_MASS_MINCHISQ', 'PHANGS_MASS_MINCHISQ_ERR', 'PHANGS_REDUCEDCHISQ_MINCHISQ', 'SEDfix_age', 'SEDfix_mass', 'SEDfix_age_limlo', 'SEDfix_mass_limlo', 'SEDfix_age_limhi', 'SEDfix_mass_limhi']

# Join LEGUS data if the LEGUS catalog exists, otherwise just fill extra LEGUS columns with 0s
if (legus_exists):
    legus_catalog = Table.read(legus_catalog_name)
    legus_catalog.rename_column("ID_PHANGS_CLUSTERS_v1p2", "ID_PHANGS_CLUSTER")
    catalog = join(phangs_catalog, legus_catalog["legus_best_mass", "legus_best_age", "legus_id", "ID_PHANGS_CLUSTER", "legus_q_prob"], join_type='left')
else:
    catalog = phangs_catalog
    for col in ["legus_best_mass", "legus_best_age", "legus_id", "legus_q_prob"]:
        catalog[col] = np.zeros(len(phangs_catalog))



# PHANGS catalogs use python indexing so no subtraction is necessary but this naming scheme is used throughout so we keep it
catalog['x'] = catalog['PHANGS_X']  #- 1  Not NECESSARY IN PHANGS
catalog['y'] = catalog['PHANGS_Y']  #- 1

# then write this catalog to the desired output file. Astropy recommends ECSV
catalog.write(final_catalog, format="ascii.ecsv", overwrite=True)
