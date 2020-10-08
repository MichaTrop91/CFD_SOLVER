# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# This script is used to generate a grid file for the model.
# Various parameters can be changed, including the building configuration.


from sys import path
path.append('./SRC/')

import numpy as np

# for defining geometries
from shapely.geometry import Polygon, Point 

# for modifying geometries
from shapely.ops import cascaded_union
from shapely.affinity import rotate, translate, scale, skew
from netCDF4 import Dataset

# for gridding the geometries
import buildings as bld


# name of grid file
simulation_name = 'play_ground'

# number of ghost cells
ng = 3

# define grid dimension sizes
nz = 25 + 2 * ng
ny = 80 + 2 * ng
nx = 80 + 2 * ng

# use constant grid spacings
dx = 5.0
dy = 5.0
dz = 5.0

# coordinates of lower-left corner
zstart = 0.0 - ng * dz
ystart = 0.0 - ng * dy
xstart = 0.0 - ng * dx


# cell-face coordinates 
x2 = np.linspace(xstart, xstart + dx * nx, nx + 1)
y2 = np.linspace(ystart, ystart + dy * ny, ny + 1)
z2 = np.linspace(zstart, zstart + dz * nz, nz + 1)


# use vertical coordinate stretching

#dz_min = dz                   #minimum grid-spacing near surface
#s_max = 8.0                   #maximum grid-stretching near-domain top
#n_min = 5                     #number of vertical layers with dz_min

#dz_max = s_max * dz_min       #maximum grid-spacing near domain-top
#z2 = [zstart]
#for n in range(n_min):
#    z2.append(z2[-1] + dz_min)
#
#for n in range(nz - n_min):
#    z2.append(z2[-1] + dz_min + (dz_max - dz_min) * (n + 1) / (nz - n_min))


# cell-center coordinates
x = 0.5 * (x2[1:] + x2[:-1])
y = 0.5 * (y2[1:] + y2[:-1])
z = 0.5 * (z2[1:] + z2[:-1])


#building geometry section
##########################


#define some basic geometries

circle_small = Point((0.0, 0.0)).buffer(10.0)
circle_medium = Point((0.0, 0.0)).buffer(20.0)
circle_large = Point((0.0, 0.0)).buffer(30.0)

rectangle_small = Polygon(((-10.0, -5.0), (-10.0, 5.0), (10.0, 5.0), (10.0, -5.0), (-10.0, -5.0)))
rectangle_medium = Polygon(((-15.0, -7.5), (-15.0, 7.5), (15.0, 7.5), (15.0, -7.5), (-15.0, -7.5)))
rectangle_large = Polygon(((-20.0, -10.0), (-20.0, 10.0), (20.0, 10.0), (20.0, -10.0), (-20.0, -10.0)))

ring_slim = circle_large.difference(circle_medium)
ring_thick = circle_large.difference(circle_small)

# place buildings across the domain
geoms = []

geoms.append(bld.building(ground=translate(circle_medium, xoff=175.0, yoff=185.0), height=30.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=translate(rectangle_large, xoff=155.0, yoff=145.0), height=20.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=rotate(translate(scale(rectangle_large, xfact=2.0), xoff=235.0, yoff=205.0), 45), height=10.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=translate(rectangle_large, xoff=135.0, yoff=235.0), height=20.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=translate(rectangle_large, xoff=175.0, yoff=235.0), height=20.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=translate(rectangle_large, xoff=135.0, yoff=275.0), height=20.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=translate(rectangle_large, xoff=175.0, yoff=275.0), height=20.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=rotate(translate(rectangle_large, xoff=205.0, yoff=255.0), 90.0), height=20.0, id_bldg=len(geoms)))
geoms.append(bld.building(ground=translate(ring_slim, xoff=245.0, yoff=135.0), height=30.0, id_bldg=len(geoms)))


grid = bld.grid(x2, y2, z2)
bld.grid_bldgs(grid, geoms)

bld.plot_buildings(grid, geoms, filename=simulation_name, filetype='eps', plot_gridlines=True)


# derive the area- and volume-scaling fields
print "calc grid factors"

# for well resolving grid spacings (faster)
ffx, ffy, ffz, fvol = bld.calc_ffactors(grid, geoms, n_hslice=2, method='geom_intersection', ff_min=0.05, angles=[0], ng=ng)

# for diffusively resolving grid spacings
#ffx, ffy, ffz, fvol = bld.calc_ffactors(grid, geoms, n_hslice=20, method='default', ff_min=0.05, angles=[0, -10, 10], ng=ng)


# fields for friction and mixing-length computation
print "calc surface field"
deff_v, deff_hx, deff_hy, area_v, area_hx, area_hy = bld.calc_surf_flds(grid, geoms, lim_d=0.1, ng=ng)


# surface roughness length
z0 = np.full([nz, ny, nx], 1e-3)


# define terrain-height function
hsurf = np.empty([nz, ny, nx])
hsurf[:] = 0.0 #flat terrain


dataset_new = Dataset(simulation_name + '.nc', 'w', format='NETCDF4')
ncdim_y = dataset_new.createDimension('y', ny)
ncdim_x = dataset_new.createDimension('x', nx)
ncdim_z = dataset_new.createDimension('ufl', nz)
ncdim_y2 = dataset_new.createDimension('y2', ny + 1)
ncdim_x2 = dataset_new.createDimension('x2', nx + 1)
ncdim_z2 = dataset_new.createDimension('uhl', nz + 1)
ncvar_x = dataset_new.createVariable('x', np.float64, ('x'))
ncvar_y = dataset_new.createVariable('y', np.float64, ('y'))
ncvar_z = dataset_new.createVariable('ufl', np.float64, ('ufl'))
ncvar_x2 = dataset_new.createVariable('x2', np.float64, ('x2'))
ncvar_y2 = dataset_new.createVariable('y2', np.float64, ('y2'))
ncvar_z2 = dataset_new.createVariable('uhl', np.float64, ('uhl'))
ncvar_fvol = dataset_new.createVariable('fvol', np.float64, ('ufl', 'y', 'x'))
ncvar_deffv = dataset_new.createVariable('deffv', np.float64, ('ufl', 'y', 'x'))
ncvar_deffhx = dataset_new.createVariable('deffhx', np.float64, ('ufl', 'y', 'x'))
ncvar_deffhy = dataset_new.createVariable('deffhy', np.float64, ('ufl', 'y', 'x'))
ncvar_areav = dataset_new.createVariable('areav', np.float64, ('ufl', 'y', 'x'))
ncvar_areahx = dataset_new.createVariable('areahx', np.float64, ('ufl', 'y', 'x'))
ncvar_areahy = dataset_new.createVariable('areahy', np.float64, ('ufl', 'y', 'x'))
ncvar_z0 = dataset_new.createVariable('z0', np.float64, ('ufl', 'y', 'x'))
ncvar_ffx = dataset_new.createVariable('fx', np.float64, ('ufl', 'y', 'x2'))
ncvar_ffy = dataset_new.createVariable('fy', np.float64, ('ufl', 'y2', 'x'))
ncvar_ffz = dataset_new.createVariable('fz', np.float64, ('uhl', 'y', 'x'))
ncvar_hsurf = dataset_new.createVariable('hsurf', np.float64, ('ufl', 'y', 'x'))

ncvar_x[:] = x
ncvar_y[:] = y
ncvar_z[:] = z

ncvar_x2[:] = x2
ncvar_y2[:] = y2
ncvar_z2[:] = z2

ncvar_fvol[:] = fvol
ncvar_ffx[:] = ffx
ncvar_ffy[:] = ffy
ncvar_ffz[:] = ffz
ncvar_deffv[:] = deff_v
ncvar_deffhx[:] = deff_hx
ncvar_deffhy[:] = deff_hy
ncvar_areav[:] = area_v
ncvar_areahx[:] = area_hx
ncvar_areahy[:] = area_hy
ncvar_z0[:] = z0
ncvar_hsurf[:] = hsurf

dataset_new.close()

