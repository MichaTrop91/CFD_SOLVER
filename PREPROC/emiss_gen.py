# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# This script is used to generate an emission file for the model.
# Please modify the script to change the number and location of sources.

import numpy as np
from netCDF4 import Dataset

#name of grid file
simulation_name = 'play_ground'

#open grid file
grid_file = Dataset(simulation_name + '.nc', 'r')

x_coord = grid_file.variables['x'][:]
y_coord = grid_file.variables['y'][:]
z_coord = grid_file.variables['ufl'][:]

nx = x_coord.size
ny = y_coord.size
nz = z_coord.size


#define sources for tracers
################################

tr1_srcfld = np.zeros([nz, ny, nx])
tr2_srcfld = np.zeros([nz, ny, nx])

tr1_name = 'source1'
tr2_name = 'source2'

# use point source for source1

loc_src1_x = 202.5
loc_src1_y = 252.5
loc_src1_z = 20.5

ind_x = np.argmin(np.absolute(loc_src1_x - x_coord))
ind_y = np.argmin(np.absolute(loc_src1_y - y_coord))
ind_z = np.argmin(np.absolute(loc_src1_z - z_coord))
tr1_srcfld[ind_z, ind_y, ind_x] = 1.0


# use an area source of four grid cells for source2

loc_src2_x = 242.5
loc_src2_y = 132.5
loc_src2_z = 0.5

ind_x = np.argmin(np.absolute(loc_src2_x - x_coord))
ind_y = np.argmin(np.absolute(loc_src2_y - y_coord))
ind_z = np.argmin(np.absolute(loc_src2_z - z_coord))

tr2_srcfld[ind_z, ind_y:ind_y + 2, ind_x:ind_x + 2] = 1.0

#write the file

emiss_file = Dataset(simulation_name + '_emiss.nc', 'w', type='NETCDF4')
zdim = emiss_file.createDimension('ufl', nz)
ydim = emiss_file.createDimension('y', ny)
xdim = emiss_file.createDimension('x', nx)

zvar = emiss_file.createVariable('ufl', np.float,  'ufl')
yvar = emiss_file.createVariable('y', np.float,  'y')
xvar = emiss_file.createVariable('x', np.float,  'x')

tr1var = emiss_file.createVariable(tr1_name, np.float,  ('ufl', 'y', 'x'))
tr2var = emiss_file.createVariable(tr2_name, np.float,  ('ufl', 'y', 'x'))

tr1var.longname = tr1_name
tr2var.longname = tr2_name

zvar[:] = z_coord
yvar[:] = y_coord
xvar[:] = x_coord

tr1var[:] = tr1_srcfld
tr2var[:] = tr2_srcfld

emiss_file.close()
