# ======================================================================================
#
# Outline
#
# ======================================================================================
# This makefile handles the full data analysis pipeline to measure the radii of star
# clusters in LEGUS galaxies. The code makes some assumptions about where the data is.
# In this repository directory, there should be a `data` directory. Within that
# directory, there should be a subdirectory for each field. Within those directories
# are the HST images and cluster catalogs. The names of those files are the same as
# they are when downloaded from the LEGUS website. I make no modifications. Then the
# code should be able to handle the rest.
# I do have a separate pipeline to do a slightly different fitting method on the two
# galaxies done by Ryon et al. 2017. You may want to remove those. To do this, set
# the ryon_dirs attribute below to be empty.
# Much of this makefile consists of setup, but I'll add more clarifying comments on
# what the pipeline does in the `Rules` section below.
#
# ======================================================================================
#
# basic setup
#
# ======================================================================================
data_home = $(shell pwd)/data_PHANGS #$(shell pwd)/data
# This directory should have nothing but directories with data
# We'll do this complicated line that just gets all directories inside data_home
all_data_dirs = $(sort $(dir $(wildcard $(shell pwd)/data_PHANGS/*/))) #$(data_home)
data_dirs = $(filter-out %artificial/, $(all_data_dirs))
ryon_dirs = $(filter %ngc1313-e/ %ngc1313-w/ %ngc628-e/ %ngc628-c/, $(all_data_dirs))
artificial_dir = $(filter %artificial/, $(all_data_dirs))

# ======================================================================================
#
# Configuration variables
#
# ======================================================================================
# psf type can either be "my" or "legus"
psf_type = my
psf_pixel_size = 15
psf_oversampling_factor = 2
fit_region_size = 30
run_name = final
image_band = f555w

# ======================================================================================
#
# Python scripts
#
# ======================================================================================
pipeline_dir = ./pipeline/
analysis_dir = ./analysis/
docs_dir = ./docs/
mass_radius_dir = $(analysis_dir)mass_radius_relation/

# pipeline
catalog_script = $(pipeline_dir)format_catalogs.py
v1_star_list_script = $(pipeline_dir)preliminary_star_list.py
psf_star_list_script = $(pipeline_dir)select_psf_stars.py
psf_creation_script = $(pipeline_dir)make_psf.py
psf_comparison_script = $(analysis_dir)psf_compare.py
psf_demo_image_script = $(analysis_dir)psf_demo_image.py
sigma_script = $(pipeline_dir)make_sigma_image.py
mask_script = $(pipeline_dir)make_mask_image.py
fitting_script = $(pipeline_dir)fit.py
fit_utils = $(pipeline_dir)fit_utils.py
final_catalog_script = $(pipeline_dir)derived_properties.py
public_catalog_script = $(pipeline_dir)public_catalog.py
# documentation
example_mrr_plot_script = $(docs_dir)example_plot.py
webpage_script = $(docs_dir)generate_webpage.py
# analysis
#comparison_script = $(analysis_dir)ryon_comparison.py
#radius_dist_script = $(analysis_dir)radius_distribution.py
#radius_dist_all_galaxies_script = $(analysis_dir)radius_distribution_all_galaxies.py
#stacked_distribution_script = $(analysis_dir)stacked_distribution.py
#example_plot_script = $(analysis_dir)example_fit.py
#cluster_bound_script = $(analysis_dir)cluster_bound.py
#density_script = $(analysis_dir)density.py
#toy_model_script = $(analysis_dir)age_toy_model.py
fit_quality_script = $(analysis_dir)fit_quality.py
#galaxy_table_script = $(analysis_dir)galaxy_table.py
#experiment_script = ./testing/experiments.py
# mrr analysis
#mass_radius_utils = $(mass_radius_dir)mass_radius_utils.py
#mass_radius_utils_mle_fitting = $(mass_radius_dir)mass_radius_utils_mle_fitting.py
#mass_radius_utils_plotting = $(mass_radius_dir)mass_radius_utils_plotting.py
#mass_radius_utils_external_data = $(mass_radius_dir)mass_radius_utils_external_data.py
#mass_radius_legus_full_script = $(mass_radius_dir)mass_radius_legus_full.py
#mass_radius_legus_young_script = $(mass_radius_dir)mass_radius_legus_young.py
#mass_radius_legus_agesplit_script = $(mass_radius_dir)mass_radius_legus_agesplit.py
#mass_radius_legus_mw_script = $(mass_radius_dir)mass_radius_legus_mw.py
#mass_radius_legus_external_script = $(mass_radius_dir)mass_radius_legus_external.py
#mass_radius_legus_mw_external_script = $(mass_radius_dir)mass_radius_legus_mw_external.py
#mass_radius_final_table_script = $(mass_radius_dir)mass_radius_final_table.py
# artificial clusters
artificial_cluster_catalog_script = $(pipeline_dir)artificial_cluster_catalog.py
artificial_cluster_image_script = $(pipeline_dir)artificial_cluster_image.py
artificial_comparison_script = $(analysis_dir)artificial_comparison.py

# ======================================================================================
#
# Directories to store data
#
# ======================================================================================
my_dirname = size/
my_dirs = $(foreach dir,$(data_dirs),$(dir)$(my_dirname)) #$(my_dirname)
my_dirs_ryon = $(foreach dir,$(ryon_dirs),$(dir)$(my_dirname))
cluster_fit_dirs = $(foreach dir,$(my_dirs),$(dir)cluster_fit_plots)
cluster_plot_dirs = $(foreach dir,$(my_dirs),$(dir)plots)
local_plots_dir = ./outputs_$(run_name)/
mass_size_tables_dir = $(local_plots_dir)sub_fit_tables/
fit_quality_dir = $(local_plots_dir)fit_quality/
all_my_dirs = $(my_dirs) $(cluster_fit_dirs) $(cluster_plot_dirs) $(local_plots_dir) $(mass_size_tables_dir) $(fit_quality_dir)

# ======================================================================================
#
# All the pipeline filenames I'll produce
#
# ======================================================================================
cat = clean_catalog.txt
star_prelim = preliminary_stars.txt
star_psf = psf_stars.txt
psf_legus = psf_legus_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.fits
psf_my = psf_my_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.fits
psf_comp_plot = psf_paper_my_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.png
sigma_image = sigma_electrons.fits
mask = mask_image.fits
fit = cluster_fits_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.h5
final_cat = final_catalog_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.txt
fit_ryon = cluster_fits_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled_ryonlike.h5
final_cat_ryon = final_catalog_$(run_name)_$(fit_region_size)_pixels_psf_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled_ryonlike.txt

# ======================================================================================
#
# Put all these files in each of these directories
#
# ======================================================================================
catalogs = $(foreach dir,$(my_dirs),$(dir)$(cat))
v1_star_lists = $(foreach dir,$(my_dirs),$(dir)$(star_prelim))
psf_star_lists = $(foreach dir,$(my_dirs),$(dir)$(star_psf))
psfs_legus = $(foreach dir,$(my_dirs),$(dir)$(psf_legus))
psfs_my = $(foreach dir,$(my_dirs),$(dir)$(psf_my))
psf_comp_plots = $(foreach dir,$(my_dirs),$(dir)$(psf_comp_plot))
sigma_images = $(foreach dir,$(my_dirs),$(dir)$(sigma_image))
masks = $(foreach dir,$(my_dirs),$(dir)$(mask))
fits = $(foreach dir,$(my_dirs),$(dir)$(fit))
final_cats = $(foreach dir,$(my_dirs),$(dir)$(final_cat))
fits_ryon = $(foreach dir,$(my_dirs_ryon),$(dir)$(fit_ryon))
final_cats_ryon = $(foreach dir,$(my_dirs_ryon),$(dir)$(final_cat_ryon))

# determine which psfs to use for fitting
ifeq ($(psf_type),my)
fit_psf = $(psf_my)
fit_psfs = $(psfs_my)
else ifeq ($(psf_type),legus)
fit_psf = $(psf_legus)
fit_psfs = $(psfs_legus)
else
$(error Bad PSF type!)
endif

# ======================================================================================
#
# Various plots and tables that will be the outputs
#
# ======================================================================================
galaxy_table = $(local_plots_dir)galaxy_table.txt
public_catalog = cluster_sizes_brown_gnedin_21.txt
webpage_template = $(docs_dir)index_template.md
example_mrr_plot = $(docs_dir)example_mrr.png
webpage = $(docs_dir)index.md
psf_demo_image = $(local_plots_dir)psf_demo_$(psf_type)_stars_$(psf_pixel_size)_pixels_$(psf_oversampling_factor)x_oversampled.pdf
comparison_plot = $(local_plots_dir)comparison_plot.pdf
radius_dist_plot = $(local_plots_dir)radius_distribution.pdf
radius_dist_all_galaxies_plot = $(local_plots_dir)radius_distribution_all_galaxies.pdf
stacked_distribution_plot = $(local_plots_dir)stacked_distribution.pdf
crossing_time_plot = $(local_plots_dir)crossing_time.pdf
bound_fraction_plot = $(local_plots_dir)bound_fraction.pdf
density_plot = $(local_plots_dir)density.pdf
density_fits_txt = $(local_plots_dir)density_fits.txt
toy_model_plot = $(local_plots_dir)age_toy_model.pdf
example_fit_plot = $(local_plots_dir)example_fit.pdf
fit_quality_plot = $(fit_quality_dir)fit_quality.png
# lots of mass size versions, all done separately
mass_radius_legus_full_plot = $(local_plots_dir)mass_radius_legus_full.pdf
mass_radius_legus_full_txt = $(mass_size_tables_dir)legus_full_table.txt
mass_radius_legus_young_plot = $(local_plots_dir)mass_radius_legus_young.pdf
mass_radius_legus_young_txt = $(mass_size_tables_dir)legus_young_table.txt
mass_radius_legus_agesplit_plot = $(local_plots_dir)mass_radius_legus_agesplit.pdf
mass_radius_legus_agesplit_txt = $(mass_size_tables_dir)legus_agesplit_table.txt
mass_radius_legus_mw_txt = $(mass_size_tables_dir)legus_mw_table.txt
mass_radius_legus_external_txt = $(mass_size_tables_dir)legus_external_table.txt
mass_radius_legus_mw_external_plot = $(local_plots_dir)mass_radius_legus_mw_external.pdf
mass_radius_legus_mw_external_txt = $(mass_size_tables_dir)legus_mw_external_table.txt
# the mass size tables get combined together into one final table
mass_radius_table = $(local_plots_dir)mass_radius_fits_table.txt
# Also do a comparison of the artificial star tests
artificial_comparison = $(local_plots_dir)artificial_tests.pdf
# then combine everything together
outputs = $(artificial_comparison)
		  #$(psf_comp_plots) $(fit_quality_plot) $(public_catalog)
          #$(psf_demo_image) $(psf_comp_plots) \
          #$(example_fit_plot) \
          #$(galaxy_table) \
          #$(comparison_plot) \
          #$(radius_dist_plot) $(radius_dist_all_galaxies_plot) \
          #$(stacked_distribution_plot) \
          #$(crossing_time_plot) $(bound_fraction_plot) \
          #$(density_plot) $(density_fits_txt) \
          #$(mass_radius_legus_full_plot) $(mass_radius_legus_young_plot) \
          #$(mass_radius_legus_agesplit_plot)  \
          #$(mass_radius_legus_mw_external_plot) $(mass_radius_table) \
          #$(toy_model_plot) \
          #$(artificial_comparison) \
          #$(webpage)


# ======================================================================================
#
#  Rules
#
# ======================================================================================
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html

# Script to move the data from a pipeline run on some band into a band specific folder
# within each galaxy folder, specifically inside whichever folder you have designated
# under my_dirname
move_band_script:
	./move_band $(image_band)

all: $(all_my_dirs) $(outputs)

$(all_my_dirs):
	mkdir $@

# The LEGUS catalogs are in a bit of a clunky format, with a catalog and separate
# readme. I format them into a cleaner format for use in the rest of the analysis.
$(catalogs): $(catalog_script)
	python $(catalog_script) $@

# --------------------------------------------------------------------------------------
#  PSF creation
# --------------------------------------------------------------------------------------
# We need to select the stars to use for the PSF. To start, we automatically make a
# preliminary list of stars, which we'll then give to the user to sort through to
# make the final list.
# The star lists require the catalogs, as we need to exclude anything that's
# one of the clusters from our selection of stars.
# To do this we use SECONDEXPANSION, and turn the star list into a catalog name. We'll
# use SECONDEXPANSION throughout, but only need to define it once
.SECONDEXPANSION:
$(v1_star_lists): %: | $(v1_star_list_script) $$(dir %)$$(cat)
	python $(v1_star_list_script) $@ $(dir $@)$(cat) $(psf_pixel_size)

# Then have the user choose the stars they like to include in the PSF generation.
# note that there's no dependency on the script itself here. That's becuase I
# don't want any unimportant future changes to make me redo all the star
# selection, since it's tedious. If I need to remake these, just delete the
# files
#| $$(dir %)$$(star_prelim) #$(dir $@)$(star_prelim) after $@
$(psf_star_lists): %: | $$(dir %)$$(star_prelim)
	python $(psf_star_list_script) $@ $(dir $@)$(star_prelim) $(psf_pixel_size) $(image_band)

# Then we use these star lists to automatically generate the PSFs.
$(psfs_my): %: | $(psf_creation_script) $$(dir %)$$(star_psf)
	python $(psf_creation_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) my $(image_band)

# I also did a comparison to PSFs from the star lists selected by LEGUS.
$(psfs_legus): %: $(psf_creation_script)
	python $(psf_creation_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) legus

# A plot comparing the psfs
#$$(dir %)$$(psf_legus)
$(psf_comp_plots): %: $$(dir %)$$(psf_my) $$(dir %)$$(star_psf) $(psf_comparison_script)
	python $(psf_comparison_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) $(image_band)

# And the demo image that will go in the paper (Figure 2)
$(psf_demo_image): $(fit_psfs) $(psf_demo_image_script)
	python $(psf_demo_image_script) $@ $(psf_oversampling_factor) $(fit_psfs)

# --------------------------------------------------------------------------------------
#  Fitting process
# --------------------------------------------------------------------------------------
# The first step in the fitting is the creation of the sigma image, which holds the
# pixel uncertainties.
$(sigma_images): $(sigma_script)
	python $(sigma_script) $@ $(image_band)

# Create the masks, where we mask out nearby stars and other clusters. We need the
# sigma image to help determine the uncertainty parameters used to tune the find star
# algorithms. We also need the catalog so we can mask out nearby clusters.
$(masks): %: $(mask_script) $$(dir %)$$(cat) $$(dir %)$$(sigma_image)
	python $(mask_script) $@ $(dir $@)$(cat) $(dir $@)$(sigma_image) $(image_band)

# Then we can actually do the fitting. This rule here does the full fitting method on
# all clusters.
$(fits): %: $(fitting_script) $(fit_utils) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask) $$(dir %)$$(cat)
	python $(fitting_script) $@ $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(dir $@)$(cat) $(fit_region_size) $(image_band)

# We also do a fitting method similar to R17 for a more direct comparison. This is only
# done on NGC 1313 and NGC 628.
#$(fits_ryon): %: $(fitting_script) $(fit_utils) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask) $$(dir %)$$(cat)
	#python $(fitting_script) $@ $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(dir $@)$(cat) $(fit_region_size) ryon_like

# Add the derived properties to these catalogs, such as effective radius and density.
$(final_cats): %: $(final_catalog_script) $(fit_utils) $$(dir %)$$(fit) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask)
	python $(final_catalog_script) $@ $(dir $@)$(fit) $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(fit_region_size) $(image_band)

# We also have a R17 like version of the derived properties that use the same distances
# as used in R17.
#$(final_cats_ryon): %: $(final_catalog_script) $(fit_utils) $$(dir %)$$(fit_ryon) $$(dir %)$$(fit_psf) $$(dir %)$$(sigma_image) $$(dir %)$$(mask)
	#python $(final_catalog_script) $@ $(dir $@)$(fit_ryon) $(dir $@)$(fit_psf) $(psf_oversampling_factor) $(dir $@)$(sigma_image) $(dir $@)$(mask) $(fit_region_size) ryon_like

# --------------------------------------------------------------------------------------
#  fit checking and exmamples
# --------------------------------------------------------------------------------------
# one plot that uses some debugging quantities that we don't want in the public catalog
$(fit_quality_plot): $(fit_quality_script) $(final_cats)
	python $(fit_quality_script) $@ $(run_name) $(final_cats)

# Make an example of the cluster fit process (Figure 3)
$(example_fit_plot): $(public_catalog) $(example_plot_script)
	python $(example_plot_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) $(fit_region_size) $(public_catalog)


# --------------------------------------------------------------------------------------
#  create the public catalog, which is used in the rest of the analysis
# --------------------------------------------------------------------------------------
$(public_catalog): $(public_catalog_script) $(final_cats)
	python $(public_catalog_script) $@ $(image_band) $(final_cats)

# --------------------------------------------------------------------------------------
#  Analysis
# --------------------------------------------------------------------------------------
# Make Table 1 showing galaxy properties
$(galaxy_table): $(public_catalog) $(galaxy_table_script)
	python $(galaxy_table_script) $@ $(psf_oversampling_factor) $(psf_pixel_size) $(psf_type) $(public_catalog)

# Make the comparisons to Ryon+17's results (Figure 9)
$(comparison_plot): $(comparison_script) $(public_catalog) $(final_cats_ryon)
	python $(comparison_script) $@ $(public_catalog) $(final_cats_ryon)

# Plot the distribution of effective radii (Figure 10)
$(radius_dist_plot): $(public_catalog) $(radius_dist_script)
	python $(radius_dist_script) $@ $(public_catalog)

# Also make a separate plot where all galaxies are separate, and do some KS test
# analysis as well.
$(radius_dist_all_galaxies_plot): $(public_catalog) $(radius_dist_all_galaxies_script)
	python $(radius_dist_all_galaxies_script) $@ $(public_catalog)

# Quantify the shape of the stacked radius distribution
$(stacked_distribution_plot): $(public_catalog) $(stacked_distribution_script)
	python $(stacked_distribution_script) $@ $(public_catalog)

# Plot the crossing time compared to age (Figure 14) and the bound fraction (Figure 15)
$(crossing_time_plot) $(bound_fraction_plot) &: $(public_catalog) $(cluster_bound_script) $(mass_radius_utils_plotting)
	python $(cluster_bound_script) $(crossing_time_plot) $(bound_fraction_plot) $(public_catalog)

# Plot the density distributions (Figure 16 and Table 3)
$(density_plot) $(density_fits_txt) &: $(public_catalog) $(density_script) $(mass_radius_utils_plotting)
	python $(density_script) $(density_plot) $(density_fits_txt) $(public_catalog)

# Make the toy model of cluster evolution (Figure 17)
$(toy_model_plot): $(toy_model_script) $(mass_radius_table) $(mass_radius_utils_plotting) $(public_catalog)
	python $(toy_model_script) $@ $(mass_radius_table) $(public_catalog)

# Various mass-radius relation plots. There are several versions here, but they will be
# the versions shown in Table 2 and Figures 11, 12, and 13.
# Need make v4.3 for this to work (can be installed with conda)
$(mass_radius_legus_full_plot) $(mass_radius_legus_full_txt) &: $(public_catalog) $(mass_radius_legus_full_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_full_script) $(mass_radius_legus_full_plot) $(mass_radius_legus_full_txt) $(public_catalog)

$(mass_radius_legus_young_plot) $(mass_radius_legus_young_txt) &: $(public_catalog) $(mass_radius_legus_young_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_young_script) $(mass_radius_legus_young_plot) $(mass_radius_legus_young_txt) $(public_catalog)

$(mass_radius_legus_agesplit_plot) $(mass_radius_legus_agesplit_txt) &: $(public_catalog) $(mass_radius_legus_agesplit_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_plotting)
	python $(mass_radius_legus_agesplit_script) $(mass_radius_legus_agesplit_plot) $(mass_radius_legus_agesplit_txt) $(public_catalog)

$(mass_radius_legus_mw_txt) &: $(public_catalog) $(mass_radius_legus_mw_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_external_data)
	python $(mass_radius_legus_mw_script) $(mass_radius_legus_mw_txt) $(public_catalog)

$(mass_radius_legus_external_txt) &: $(public_catalog) $(mass_radius_legus_external_script) $(mass_radius_utils) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_external_data)
	python $(mass_radius_legus_external_script) $(mass_radius_legus_external_txt) $(public_catalog)

$(mass_radius_legus_mw_external_plot) $(mass_radius_legus_mw_external_txt) &: $(public_catalog) $(mass_radius_legus_mw_external_script) $(mass_radius_utils) $(mass_radius_utils_plotting) $(mass_radius_utils_mle_fitting) $(mass_radius_utils_external_data)
	python $(mass_radius_legus_mw_external_script) $(mass_radius_legus_mw_external_plot) $(mass_radius_legus_mw_external_txt) $(public_catalog)

# combine all the fits into Table 2.
$(mass_radius_table): $(mass_radius_final_table_script) $(mass_radius_legus_full_txt) $(mass_radius_legus_young_txt) $(mass_radius_legus_agesplit_txt) $(mass_radius_legus_mw_txt)  $(mass_radius_legus_external_txt)  $(mass_radius_legus_mw_external_txt)
	python $(mass_radius_final_table_script) $(mass_radius_table) $(mass_radius_legus_full_txt) $(mass_radius_legus_young_txt) $(mass_radius_legus_agesplit_txt) $(mass_radius_legus_mw_txt)  $(mass_radius_legus_external_txt)  $(mass_radius_legus_mw_external_txt)

# --------------------------------------------------------------------------------------
#  Artificial cluster tests
# --------------------------------------------------------------------------------------
# Set up the artificial cluster tests separately. This is a bit clunky as it
# isn't as automated as the normal runs, but that's okay as it's a different
# workflow from those other runs, slightly.
# the artificial image needs the long name so the code can find it
base_field = ngc1087
base_field_catalog = $(shell pwd)/data_PHANGS/$(base_field)/$(my_dirname)$(cat)
artificial_catalog = $(artificial_dir)true_catalog.txt
artificial_image = $(artificial_dir)hlsp_phangs-hst_hst_acs-wfc_artificial_$(image_band)_v1_exp-drc-sci.fits
artificial_psf = $(artificial_dir)$(my_dirname)$(psf_my)
artificial_sigma_image = $(artificial_dir)$(my_dirname)$(sigma_image)
artificial_mask_image = $(artificial_dir)$(my_dirname)$(mask)
artificial_fit = $(artificial_dir)$(my_dirname)$(fit)
artificial_final_cat = $(artificial_dir)$(my_dirname)$(final_cat)

# use the psf from another galaxy
$(artificial_psf): $(psfs_my)
	cp $(shell pwd)/data_PHANGS/$(base_field)/$(my_dirname)$(psf_my) $@

# Make the catalog with the true locations and parameters of the clusters
# This depends on the LEGUS catalog so I can avoid those clusters
$(artificial_catalog): $(artificial_cluster_catalog_script) $(base_field_catalog)
	python $(artificial_cluster_catalog_script) $@ $(base_field) $(base_field_catalog)

# the artificial image with fake clusters
$(artificial_image): $(artificial_cluster_image_script) $(artificial_catalog)
	python $(artificial_cluster_image_script) $@ $(artificial_catalog) $(psf_oversampling_factor) $(fit_region_size) $(artificial_psf) $(base_field) $(image_band)

# sigma image is done with the normal pipeline. It does depend on the original image
$(artificial_sigma_image): $(sigma_script) $(artificial_image)
	python $(sigma_script) $@ $(image_band)

# mask is done automatically
$(artificial_mask_image): $(mask_script) $(artificial_catalog) $(artificial_sigma_image) $(artificial_image)
	python $(mask_script) $@ $(artificial_catalog) $(artificial_sigma_image) $(image_band)

# then we can do the fitting and postprocessing!
$(artificial_fit): %: $(fitting_script) $(fit_utils) $(artificial_psf) $(artificial_sigma_image) $(artificial_mask_image) $(artificial_catalog)
	python $(fitting_script) $@ $(artificial_psf) $(psf_oversampling_factor) $(artificial_sigma_image) $(artificial_mask_image) $(artificial_catalog) $(fit_region_size) $(image_band)

# Add the derived properties to these catalogs
$(artificial_final_cat): $(final_catalog_script) $(fit_utils) $(artificial_fit) $(artificial_psf) $(artificial_sigma_image) $(artificial_mask_image)
	python $(final_catalog_script) $@ $(artificial_fit) $(artificial_psf) $(psf_oversampling_factor) $(artificial_sigma_image) $(artificial_mask_image) $(fit_region_size) $(image_band)

# Then make the plot comparing the results (Figure 8)
$(artificial_comparison): $(artificial_final_cat) $(artificial_comparison_script)
	python $(artificial_comparison_script) $@ $(artificial_final_cat) $(artificial_psf) $(psf_oversampling_factor)

# --------------------------------------------------------------------------------------
#  Documentation
# --------------------------------------------------------------------------------------
# plot to go in the webpage demonstrating the cluster catalogs
$(example_mrr_plot): $(example_mrr_plot_script) $(public_catalog)
	python $(example_mrr_plot_script) $@ $(public_catalog)

# the webpage documenting the cluster catalogs and how to use them.
$(webpage): $(webpage_script) $(public_catalog) $(webpage_template) $(example_mrr_plot)
	python $(webpage_script) $@ $(public_catalog) $(webpage_template) $(example_mrr_plot_script)
