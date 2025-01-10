"""
con_ind.py - Creates a KDE displaying the concentration indices for the stars involved 
in PSF creation. This KDE uses the Epanechnikov kernel.
"""

import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

data_dir = Path("data_PHANGS")

mean_cis = []
for galaxy in data_dir.iterdir():
    if galaxy.name == ".DS_Store":
        continue
        
    galaxy_name = galaxy.name
    
    size_home_dir = data_dir / f"{galaxy_name}/size"
    
    cis = np.loadtxt(size_home_dir / "Concentration_Index.txt")
    
    mean = np.mean(cis)
    
    mean_cis.append(mean)
    

# Epanechnikov kernel
def k_E(u):
    k = 0.75*(1.-u**2)
    k[k<0] = 0
    return k

# Kernel Density Estimator
def kde(Xdata, Xgrid, hmin=None, norm=True, kernel=k_E):
    # optimal bandwidth
    IQR = np.percentile(Xdata,75) - np.percentile(Xdata,25)
    h = min(np.std(Xdata), IQR/1.34)/len(Xdata)**0.2
    if hmin is not None and hmin > h:
        print('kde: h = %g hmin = %g' % (h,hmin))
        h = hmin
    k = np.array([ np.sum(kernel((x-Xdata)/h)) for x in Xgrid ])
    if norm:
        k = k/len(Xdata)
    return k/h
    
print(min(mean_cis), max(mean_cis))


grid = np.linspace(0, 3, 10000)

plt.fill(grid, kde(mean_cis, grid, np.std(mean_cis)), alpha=0.3)
plt.title("Concentation Index KDE")
plt.xlabel("CI Values")
plt.ylabel("Normalized Density")
plt.xlim(0.5, 1.5)
plt.ylim(0, 5)

plt.savefig("Con_Ind_Kernel.png")

