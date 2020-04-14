import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import time
import pandas as pd

mud = pd.read_csv('mud.dat', sep='\t',header=0)
fdat = pd.read_csv('f.dat', sep='\t',header=0)
fpdat = pd.read_csv('fp.dat', sep='\t',header=0)

mud = mud.drop(mud[mud.solz < 0].index)
fdat = fdat.drop(fdat[fdat.solz < 0].index)
fpdat = fpdat.drop(fpdat[fpdat.solz < 0].index)

grdwvl = mud.wl.unique()
grdchl = np.array([0.03,0.1,0.3,1,3,10])
grdsolz = mud.solz.unique()

mudval = np.zeros((len(grdsolz),len(grdwvl),len(grdchl)))
fval = np.zeros((len(grdsolz),len(grdwvl),len(grdchl)))
fpval = np.zeros((len(grdsolz),len(grdwvl),len(grdchl)))

for s in grdsolz:
    x = np.where(grdsolz == s)
    for w in grdwvl:
        y = np.where(grdwvl == w)

        mudval[x,y,:] = mud[(mud.solz == s) & (mud.wl == w)].iloc[0,2:8]
        fval[x,y,:] = fdat[(fdat.solz == s) & (fdat.wl == w)].iloc[0,2:8]
        fpval[x,y,:] = fpdat[(fpdat.solz == s) & (fpdat.wl == w)].iloc[0,2:8]
    
# Create the netCDF file and add dimensions
dataset = Dataset('morel_mud_f_fp.nc', 'w',format='NETCDF4')
dataset.createDimension('nsolz',len(grdsolz))
dataset.createDimension('nwavelengths', len(grdwvl))
dataset.createDimension('nchl', len(grdchl))

# Set global attributes
dataset.title = "Morel f, fp and mud tables"
dataset.date_created = (datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%SZ')).encode('ascii')
dataset.creator_name = "Sean Bailey, NASA/GSFC/OBPG"
dataset.creator_email = "data@oceancolor.gsfc.nasa.gov" 
dataset.creator_url = "https://oceancolor.gsfc.nasa.gov" 
dataset.product_name="morel_mud_f_fp.nc'"
#    dataset.source = 'generate_gas_tables.py'

# Create the table data sets
mudSDS = dataset.createVariable('mud', np.float32, ('nsolz','nwavelengths','nchl'))
fSDS = dataset.createVariable('f', np.float32, ('nsolz','nwavelengths','nchl'))
fpSDS = dataset.createVariable('fp', np.float32, ('nsolz','nwavelengths','nchl'))
solz = dataset.createVariable('solz', np.float32, ('nsolz',))
wavelength = dataset.createVariable('wavelength', np.float32, ('nwavelengths',))
chlorophyll = dataset.createVariable('chlorophyll', np.float32, ('nchl',))

# Write the data
mudSDS[:] = mudval
fSDS[:] = fval
fpSDS[:] = fpval
solz[:] = grdsolz
wavelength[:] = grdwvl
chlorophyll[:] = grdchl

# Add some variable attributes
wavelength.units = 'nm'
chlorophyll.units = 'mg/m3'
solz.units = 'degrees'
mudSDS.description = 'mean cosine for downward flux'


# Close the file
dataset.close()
