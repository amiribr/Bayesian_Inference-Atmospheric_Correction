import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import time
import pandas as pd

kbio = pd.read_csv('mm01_kbio.txt', sep=' ',header=0)

# Create the netCDF file and add dimensions
dataset = Dataset('mm01_kbio.nc', 'w',format='NETCDF4')
dataset.createDimension('nwavelengths', len(kbio.wl))

# Set global attributes
dataset.title = "Morel and Maritorena kbio table"
dataset.date_created = (datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%SZ')).encode('ascii')
dataset.creator_name = "Sean Bailey, NASA/GSFC/OBPG"
dataset.creator_email = "data@oceancolor.gsfc.nasa.gov" 
dataset.creator_url = "https://oceancolor.gsfc.nasa.gov" 
dataset.product_name="mm01_kbuio.nc"

# Create the table data sets
wavelength = dataset.createVariable('wavelength', np.float32, ('nwavelengths',))
kw = dataset.createVariable('kw', np.float32, ('nwavelengths',))
e = dataset.createVariable('e', np.float32, ('nwavelengths',))
X = dataset.createVariable('X', np.float32, ('nwavelengths',))

# Write the data
kw[:] = np.array(kbio.kw)
e[:] = np.array(kbio.e)
X[:] = np.array(kbio.X)
wavelength[:] = np.array(kbio.wl)

# Add some variable attributes
wavelength.units = 'nm'

# Close the file
dataset.close()
