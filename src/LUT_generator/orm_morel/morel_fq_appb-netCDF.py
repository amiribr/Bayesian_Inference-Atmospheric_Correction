import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import time

#
# tables 1 and 2 from appendix b of morel et al. (2002)
# wl (columns): 412.5, 442.5, 490, 510, 560, 620, 670
# chl (rows): 0.03, 0.1, 0.3, 1, 3, 10
# used by morel_fq_appb.pro
#

f0 = [[0.297892,0.311742,0.347280,0.359728,0.375008,0.370053,0.372716],
[0.324018,0.328848,0.345755,0.350503,0.358735,0.350497,0.349206],
[0.340239,0.341657,0.350980,0.349334,0.349570,0.334437,0.330755],
[0.351673,0.352505,0.362207,0.359773,0.357388,0.327275,0.320913],
[0.359587,0.360429,0.374357,0.376513,0.383433,0.335178,0.323731],
[0.370570,0.370782,0.389128,0.397841,0.424316,0.362306,0.340204]]

sf = [[0.065801,0.076526,0.095435,0.103901,0.121165,0.134426,0.143912],
[0.095786,0.111534,0.138252,0.147280,0.169210,0.183429,0.195362],
[0.131988,0.154209,0.191203,0.201694,0.225606,0.227381,0.236441],
[0.183170,0.213187,0.261757,0.276009,0.305781,0.284162,0.285455],
[0.239626,0.273898,0.330935,0.349059,0.386471,0.352961,0.343078],
[0.316124,0.351720,0.415626,0.438584,0.482277,0.457638,0.433943]]

q0 = [[3.318220,3.291250,3.245640,3.176500,3.138020,3.116680,3.126530],
[3.375400,3.385700,3.408430,3.359680,3.336890,3.309970,3.332380],
[3.484950,3.529680,3.613410,3.601660,3.606950,3.542300,3.560200],
[3.675060,3.746830,3.868930,3.894090,3.942030,3.815240,3.801910],
[3.913530,3.991380,4.110030,4.141140,4.188650,4.104160,4.053610],
[4.252700,4.313260,4.393950,4.405460,4.368130,4.427190,4.371050]]

sq = [[0.863223,0.976278,1.129290,1.203100,1.296770,1.413010,1.479420],
[1.055510,1.190090,1.382130,1.482910,1.625960,1.775070,1.866290],
[1.302830,1.469930,1.700680,1.827460,2.036100,2.175700,2.269360],
[1.671180,1.877700,2.115910,2.239960,2.479820,2.711180,2.789710],
[2.083950,2.303690,2.506640,2.580090,2.707700,3.141310,3.229040],
[2.625950,2.843750,2.978790,2.986960,2.898750,3.527960,3.717540]]

# Create the netCDF file and add dimensions
dataset = Dataset('morel_fq_appb.nc', 'w',format='NETCDF4')
dataset.createDimension('nchl', 6)
dataset.createDimension('nwavelengths', 7)

# Set global attributes
dataset.title = "Morel f/Q table"
dataset.date_created = (datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%SZ')).encode('ascii')
dataset.creator_name = "Sean Bailey, NASA/GSFC/OBPG"
dataset.creator_email = "data@oceancolor.gsfc.nasa.gov" 
dataset.creator_url = "https://oceancolor.gsfc.nasa.gov" 
dataset.product_name="morel_fq_appb.nc'"
#    dataset.source = 'generate_gas_tables.py'

# Create the table data sets
f0SDS = dataset.createVariable('f0', np.float32, ('nchl','nwavelengths'))
sfSDS = dataset.createVariable('sf', np.float32, ('nchl','nwavelengths'))
q0SDS = dataset.createVariable('q0', np.float32, ('nchl','nwavelengths'))
sqSDS = dataset.createVariable('sq', np.float32, ('nchl','nwavelengths'))
wavelength = dataset.createVariable('wavelength', np.float32, ('nwavelengths',))
chlorophyll = dataset.createVariable('chlorophyll', np.float32, ('nchl',))

# Write the data
f0SDS[:] = f0
sfSDS[:] = sf
q0SDS[:] = q0
sqSDS[:] = sq
wavelength[:] = np.array([412.5, 442.5, 490, 510, 560, 620, 670])
chlorophyll[:] = np.array([0.03, 0.1, 0.3, 1, 3, 10])

# Add some variable attributes
wavelength.units = 'nm'
chlorophyll.units = 'mg/m3'

# Close the file
dataset.close()
