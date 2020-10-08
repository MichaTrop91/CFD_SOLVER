# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# This script is used to generate boundary files for the model.
# Please modify the script to change the boundary conditions.

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import interp1d


#name of grid file
simulation_name = 'play_ground'

#time units
t_units = 'seconds since 2019-01-01 00:00:00'

#start of simulation
st_time = 0.0

#end of simulation
sim_time = 3600.0

#stride between boundary files
delta_t = 900.0


#open grid file
grid_file = Dataset(simulation_name + '.nc', 'r')

x_coord = grid_file.variables['x'][:] 
y_coord = grid_file.variables['y'][:]
z_coord = grid_file.variables['ufl'][:]

x2_coord = grid_file.variables['x2'][:]
y2_coord = grid_file.variables['y2'][:]
z2_coord = grid_file.variables['uhl'][:]

dx = x_coord[1:] - x_coord[:-1]
dy = y_coord[1:] - y_coord[:-1]
dz = z_coord[1:] - z_coord[:-1]

nx = x_coord.size
ny = y_coord.size
nz = z_coord.size

ntimes = int(sim_time / delta_t) + 2
times = np.linspace(st_time, sim_time + delta_t, ntimes)



#define the physical fields

#wind speed u-component
u = np.empty([ntimes, nz, ny, nx + 1], dtype=np.float64)

#wind speed v-component
v = np.empty([ntimes, nz, ny + 1, nx], dtype=np.float64)

#wind speed w-component
w = np.empty([ntimes, nz + 1, ny, nx], dtype=np.float64)

#turbulent intensity (RMS) u-component
u_rms = np.empty([ntimes, nz, ny, nx + 1], dtype=np.float64)

#turbulent intensity (RMS) v-component
v_rms = np.empty([ntimes, nz, ny + 1, nx], dtype=np.float64)

#turbulent intensity (RMS) w-component
w_rms = np.empty([ntimes, nz + 1, ny, nx], dtype=np.float64)

#air density
rho = np.empty([ntimes, nz, ny, nx], dtype=np.float64)

#potential temperature
theta = np.empty([ntimes, nz, ny, nx], dtype=np.float64)

#specific humidity
qv = np.empty([ntimes, nz, ny, nx], dtype=np.float64)

#surface potential temperature
thsurf = np.empty([ntimes, nz, ny, nx], dtype=np.float64)

#surface specific humdity
qvsurf = np.empty([ntimes, nz, ny, nx], dtype=np.float64)


u[:] = 1.0
v[:] = 0.0
w[:] = 0.0

u_rms[:] = 0.2
v_rms[:] = 0.2
w_rms[:] = 0.2

rho[:] = 1.0

theta[:] = 280.0

qv[:] = 0.0

thsurf[:] = 280.0

qvsurf[:] = 0.0


for n, time in enumerate(times):
    data_new = Dataset(simulation_name + '_bnd_{:06d}.nc'.format(int(time)), 'w', type='NETCDF4')

    timdim = data_new.createDimension('time', 1)
    zdim = data_new.createDimension('ufl', nz)
    ydim = data_new.createDimension('y', ny)
    xdim = data_new.createDimension('x', nx)
    z2dim = data_new.createDimension('uhl', nz + 1)
    y2dim = data_new.createDimension('y2', ny + 1)
    x2dim = data_new.createDimension('x2', nx + 1)

    timvar = data_new.createVariable('time', int, 'time')
    zvar = data_new.createVariable('ufl', np.float,  'ufl')
    yvar = data_new.createVariable('y', np.float,  'y')
    xvar = data_new.createVariable('x', np.float,  'x')
    z2var = data_new.createVariable('uhl', np.float,  'uhl')
    y2var = data_new.createVariable('y2', np.float,  'y2')
    x2var = data_new.createVariable('x2', np.float,  'x2')
    
    u_var = data_new.createVariable('U', np.float, ('time', 'ufl', 'y', 'x2'))
    v_var = data_new.createVariable('V', np.float, ('time', 'ufl', 'y2', 'x'))
    w_var = data_new.createVariable('W', np.float, ('time', 'uhl', 'y', 'x'))
    urms_var = data_new.createVariable('U_rms', np.float, ('time', 'ufl', 'y', 'x2'))
    vrms_var = data_new.createVariable('V_rms', np.float, ('time', 'ufl', 'y2', 'x'))
    wrms_var = data_new.createVariable('W_rms', np.float, ('time', 'uhl', 'y', 'x'))
    Rho_var = data_new.createVariable('Rho', np.float, ('time', 'ufl', 'y', 'x'))
    Th_var = data_new.createVariable('Theta', np.float, ('time', 'ufl', 'y', 'x'))
    QV_var = data_new.createVariable('QV', np.float, ('time', 'ufl', 'y', 'x'))
    Thsrf_var = data_new.createVariable('Th_S', np.float, ('time', 'ufl', 'y', 'x'))
    QVsrf_var = data_new.createVariable('QV_S', np.float, ('time', 'ufl', 'y', 'x'))

    timvar[:] =  time
    timvar.units = t_units
    zvar[:] = z_coord
    yvar[:] = y_coord
    xvar[:] = x_coord
    z2var[:] = z2_coord
    y2var[:] = y2_coord
    x2var[:] = x2_coord

    u_var[:] = u[n]
    v_var[:] = v[n]
    w_var[:] = w[n]
    urms_var[:] = u_rms[n]
    vrms_var[:] = v_rms[n]
    wrms_var[:] = w_rms[n]
    Rho_var[:] = rho[n]
    Th_var[:] = theta[n]
    QV_var[:] = qv[n]
    Thsrf_var[:] = thsurf[n]
    QVsrf_var[:] = qvsurf[n]

    data_new.close()   
