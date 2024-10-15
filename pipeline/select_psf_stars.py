"""
select_psf_stars.py

Select stars to be made into the PSF. This takes the outputs of
`preliminary_star_list.py` and presents these stars to the user for inspection.
"""
import sys
from pathlib import Path

from astropy import table
from astropy import stats
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import betterplotlib as bpl
from astropy.table import Table, vstack
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
)
from PySide2.QtGui import QPixmap, QKeySequence
from PySide2.QtCore import Qt

import utils
import pandas

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output catalog name, which we can use to get the home directory
final_catalog = Path(sys.argv[1]).absolute()
home_dir = final_catalog.parent.parent
# We'll need to get the preliminary catalog too
preliminary_catalog_path = Path(sys.argv[2]).absolute()
psf_width = int(sys.argv[3]) #psf_width = int(sys.argv[3])

# ======================================================================================
#
# Get the data - preliminary catalog and images, plus other setup
#
# ======================================================================================
# read in this catalog
stars_table = table.Table.read(preliminary_catalog_path, format="ascii.ecsv")

# Get isolated sources list
def find_isolated_star_list(home_dir):
    """
    Find the name of the isolated star list if it exists. We need a function for this because we
    don't know whether it has ACS and WFC3 in the filename of just one.

    :param home_dir: Directory to search for catalogs
    :type home_dir: Path
    :return: Path objects pointing to the catalog and readme file.
    :rtype: tuple
    """
    galaxy_name = home_dir.name
    # This handles one strange exception in the galaxy naming structure for isolated star lists
    if galaxy_name == "ngc685":
        galaxy_num = galaxy_name[-3:]
    else:
        galaxy_num = galaxy_name[-4:]

    isolated_dir = home_dir/Path("isolated_objects")
    
    for item in isolated_dir.iterdir():
        if not item.is_file():
            continue
            
        filename = item.name
        # See if it starts and ends with what the isolated catalog should be. There are multiple different conventions depending on the galaxy name.
            
        if galaxy_name == "ngc628c" or galaxy_name == "ngc628e" or galaxy_name[0] == 'i':
            if filename.startswith(f"{galaxy_name}") and filename.endswith(
                f".txt"
            ):
                star_list = item
                return star_list
        
        else:
            if filename.startswith(f"n{galaxy_num}") and filename.endswith(
                f".txt"
            ):
                star_list = item
                return star_list
    # if we got here, we have an error.
    raise FileNotFoundError(f"No star list found in {home_dir}")
    
star_list_name = find_isolated_star_list(home_dir)
# This line handles in line comments for the rows of the isolated sources catalog. Many of the catalogs have them and the Pandas package has one of the best
# ways to get rid of these segments so we first convert to a Pandas dataframe and back to a Astropy table.
pandas_table = pandas.read_table(star_list_name, comment = "#", sep="\s+")
isolated_stars_table = Table.from_pandas(pandas_table)

# Get only the stars from the isolated sources list.
isolated_stars_table = isolated_stars_table[np.where(isolated_stars_table["flag"] == 0)]

# handle one vs zero indexing
isolated_stars_table["x"] -= 1
isolated_stars_table["y"] -= 1

isolated_stars_table = isolated_stars_table["x", "y"]


# Check for Dolphot

if stars_table[0][0] == "No Dolphot file":
    # No Dolphot file exists -> use isolated sources
    write_table = isolated_stars_table
    write_table.write(final_catalog, format="ascii.ecsv")
    
    
else:
    # Get the image in the appropriate band.
    band_select = sys.argv[4] # edit this here to get new data
    bands = utils.get_drc_image(home_dir)
    image_data = bands[band_select][0]

    # get the noise_level, which will be used later
    _, _, noise = stats.sigma_clipped_stats(image_data, sigma=2.0)


    # ======================================================================================
    #
    # Setting up GUI
    #
    # ======================================================================================
    # make the temporary location to store images
    temp_loc = str(Path(__file__).absolute().parent / "temp.png")
    # and add a column to the table indicating which stars are kept
    stars_table["use_for_psf"] = False


    class MainWindow(QMainWindow):
        def __init__(self, starList, imageData, width):
            QMainWindow.__init__(self)

            self.starData = starList
            self.idx = -1
            self.imageData = imageData
            self.plotWidth = width

            # the layout of this will be as follows: There will be an image of the star
            # shown, with it's data on the left side. Below that will be two buttons:
            # yes and no
            vBoxMain = QVBoxLayout()

            # The first thing will be the data and image, which are laid out horizontally
            hBoxImageData = QHBoxLayout()
            self.image = QLabel()
            self.starDataText = QLabel("Image data here\nAttr 1\nAttr2")
            hBoxImageData.addWidget(self.image)
            hBoxImageData.addWidget(self.starDataText)
            hBoxImageData.setAlignment(Qt.AlignTop)
            vBoxMain.addLayout(hBoxImageData)

            # then the buttons at the bottom, which will also be laid our horizontally
            hBoxInput = QHBoxLayout()
            self.acceptButton = QPushButton("Accept")
            self.rejectButton = QPushButton("Reject")
            self.exitButton = QPushButton("Done Selecting Stars")
            # set the tasks that each button will do
            self.acceptButton.clicked.connect(self.accept)
            self.rejectButton.clicked.connect(self.reject)
            self.exitButton.clicked.connect(self.exit)
            # and make keyboard shortcuts
            acceptShortcut = QShortcut(QKeySequence("right"), self.acceptButton)
            rejectShortcut = QShortcut(QKeySequence("left"), self.rejectButton)
            exitShortcut = QShortcut(QKeySequence("d"), self.exitButton)

            acceptShortcut.activated.connect(self.accept)
            rejectShortcut.activated.connect(self.reject)
            exitShortcut.activated.connect(self.exit)

            hBoxInput.addWidget(self.rejectButton)
            hBoxInput.addWidget(self.acceptButton)
            hBoxInput.addWidget(self.exitButton)
            vBoxMain.addLayout(hBoxInput)

            # have to set a dummy widget to act as the central widget
            container = QWidget()
            container.setLayout(vBoxMain)
            self.setCentralWidget(container)
            # self.resize(1000, 1000)

            # add the first star
            self.nextStar()

            # then we can show the widget
            self.show()

        def accept(self):
            self.starData["use_for_psf"][self.idx] = True
            self.nextStar()

        def reject(self):
            self.starData["use_for_psf"][self.idx] = False
            self.nextStar()

        def exit(self):
            QApplication.quit()

        def nextStar(self):
            self.idx += 1
            thisStar = self.starData[self.idx]
            while thisStar["is_cluster"]:
                self.idx += 1
                thisStar = self.starData[self.idx]
            # make the temporary plot
            self.snapshot()
            # then add it to the the GUI
            self.image.setPixmap(QPixmap(temp_loc))
            # update the label
            new_info = (f"Number of Accepted Stars: {np.sum(self.starData['use_for_psf'])}\n"
                        f"Number of Examined Stars: {self.idx}\n\n"
                        f"x: {thisStar['Dolphot_x']:.3f}\n"
                        f"y: {thisStar['Dolphot_y']:.3f}\n")
                #f"Image Directory: \n{home_dir}\n\n"
                #f"Number of Accepted Stars: {np.sum(self.starData['use_for_psf'])}\n"
                #f"Number of Examined Stars: {self.idx}\n\n"
                #f"ID: {thisStar['id']:.3f}\n"
                #f"x: {thisStar['xcentroid']:.3f}\n"
                #f"y: {thisStar['ycentroid']:.3f}\n\n"
                #f"FWHM: {thisStar['fwhm']:.3f}\n"
                #f"Sharpness: {thisStar['sharpness']:.3f}\n"
                #f"Ellipticiy: {thisStar['roundness']:.3f}\n\n"
                #f"Sky: {thisStar['sky']:.3f}\n"
                #f"Flux (unknown units): {thisStar['flux']:.3f}\n"
            #)
            if thisStar["near_star"]:
                new_info += "NEAR ANOTHER STAR\n"
                self.starDataText.setStyleSheet("QLabel { color : firebrick; }")
            if thisStar["near_cluster"]:
                new_info += "NEAR AN IDENTIFIED CLUSTER\n"
                self.starDataText.setStyleSheet("QLabel { color : red; }")
            if not (thisStar["near_star"] or thisStar["near_cluster"]):
                self.starDataText.setStyleSheet("QLabel { color : black; }")

            self.starDataText.setText(new_info)

            self.image.repaint()
            self.starDataText.repaint()

        def snapshot(self):
            thisStar = self.starData[self.idx]
            cen_x = thisStar["Dolphot_x"]
            cen_y = thisStar["Dolphot_y"]
            # get the subset of the data first
            # get the central pixel
            cen_x_pix = int(np.floor(cen_x))
            cen_y_pix = int(np.floor(cen_y))
            # we'll select a larger subset around that central pixel, then change the plot
            # limits to be just in the center, so that the object always appears at the
            # center
            buffer_half_width = int(np.ceil(self.plotWidth / 2) + 3)
            min_x_pix = cen_x_pix - buffer_half_width
            max_x_pix = cen_x_pix + buffer_half_width
            min_y_pix = cen_y_pix - buffer_half_width
            max_y_pix = cen_y_pix + buffer_half_width
            # then get this slice of the data
            snapshot_data = self.imageData[min_y_pix:max_y_pix, min_x_pix:max_x_pix]

            # When showing the plot I want the star to be in the very center. To do this I
            # need to get the values for the border in the new pixel coordinates
            cen_x_new = cen_x - min_x_pix
            cen_y_new = cen_y - min_y_pix
            # then make the plot limits
            min_x_plot = cen_x_new - 0.5 * self.plotWidth
            max_x_plot = cen_x_new + 0.5 * self.plotWidth
            min_y_plot = cen_y_new - 0.5 * self.plotWidth
            max_y_plot = cen_y_new + 0.5 * self.plotWidth

            fig, ax = bpl.subplots(figsize=[6, 5])
            vmax = np.max(snapshot_data)
            vmin = -5 * noise
            linthresh = max(0.01 * vmax, 5 * noise)
            norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=linthresh, base=10)
            im = ax.imshow(snapshot_data, norm=norm, origin="lower")
            ax.set_limits(min_x_plot, max_x_plot, min_y_plot, max_y_plot)
            ax.scatter([cen_x_new], [cen_y_new], marker="x", c=bpl.almost_black, s=20)
            ax.remove_labels("both")
            #ax.remove_spines(["all"])
            fig.colorbar(im, ax=ax)
            fig.savefig(temp_loc, dpi=100, bbox_inches="tight")
            plt.close(fig)


    app = QApplication()

    # The MainWindow class holds all the structure
    window = MainWindow(stars_table, image_data, psf_width)

    # Execute application
    app.exec_()

    stars_table = stars_table[np.where(stars_table["use_for_psf"])]
    
    stars_table.rename_columns(["Dolphot_x", "Dolphot_y"], ["x", "y"])
    stars_table = stars_table["x", "y"]
    
    # if band is not f555w, isolated sources won't be much help from experience, you are free to test this yourself and remove the following from the if statement.
    if band_select == "f555w":
        
        # create final star table for background analysis and psf creation
        stars_table = vstack([stars_table, isolated_stars_table])

        # Remove duplicates
        size = len(stars_table)
        i = 0

        while i < size:
            j = i + 1
            while j < size:
                if (abs(stars_table["x"][i] - stars_table["x"][j]) < 1) and (abs(stars_table["y"][i] - stars_table["y"][j]) < 1):
                    # These are just to check what I removed, feel free to comment them out.
                    print("same star x_coords:", stars_table["x"][i], stars_table["x"][j], i, j)
                    print("same star y_coords:", stars_table["y"][i], stars_table["y"][j], "\n")
                    stars_table.remove_row(j)
                    size -= 1
                j += 1
            i += 1
    
    
    # The table will be modified as we go. We can then grab the rows selected and output
    # the table
    write_table = stars_table
    write_table.write(final_catalog, format="ascii.ecsv")
