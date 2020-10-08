# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# This script contains classes and functions for gridding geometric building data.

import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon, Point
import shapely.ops as ops
from copy import copy

# only for plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from descartes import PolygonPatch
       

class cell:
    '''
    2d-grid cell to store buildings
    '''

    def __init__(self, lst):
        '''
        lst... list containing the polygon made up by the corner points, 
               and the centroid coordinate pair
        '''
        self.polygon = lst[0]
        self.x = lst[1]
        self.y = lst[2]
        self.bld_int = []
        self.bld_cont = []

    def cont_bld(self, bld):
        '''
        Add a building which is contained
        in the grid cell.

        bld... building
        '''

        bld_cont = self.bld_cont
        bld_cont.append(bld)
        self.bld_cont = bld_cont

    def int_bld(self, bld):
        '''
        Add a building which intersects
        the grid cell.

        bld... building
        '''

        bld_int = self.bld_int
        bld_int.append(bld)
        self.bld_int = bld_int


class grid:
    '''
    Stores the grid coordinates and
    2d-grid cells containing the buildings
    in a structured fashion.
    '''

    def __init__(self, x2, y2, z2):
        '''
        x2, y2, z2... coordinates of grid planes
        '''

        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

        self.x = 0.5 * (x2[1:] + x2[:-1])
        self.y = 0.5 * (y2[1:] + y2[:-1])
        self.z = 0.5 * (z2[1:] + z2[:-1])

        self.xinc = x2[1:] - x2[:-1]
        self.yinc = y2[1:] - y2[:-1]
        self.zinc = z2[1:] - z2[:-1]

        self.xfirst = x2[0] + (self.xinc[0]) / 2.0
        self.yfirst = y2[0] + (self.yinc[0]) / 2.0
        self.zfirst = z2[0] + (self.zinc[0]) / 2.0


        self.nx = len(x2) - 1
        self.ny = len(y2) - 1
        self.nz = len(z2) - 1

        cells = np.empty([self.ny, self.nx], dtype=list)
        vcell = np.vectorize(cell)

        for i in xrange(0, self.ny, 1):
            for j in xrange(0, self.nx, 1):
                lowleft = [x2[j] , y2[i]]
                lowright = [x2[j + 1], y2[i]]
                topleft = [x2[j], y2[i + 1]]
                topright = [x2[j + 1], y2[i + 1]]
                polygon = Polygon([lowleft, lowright, topright, topleft, lowleft])
                cells[i, j] = list([polygon, self.x[j], self.y[i]])

        self.cells = vcell(cells)


class building:
    '''
    Building consisting of a ground polygon, height
    and an identifier. 
    '''

    def __init__(self, ground, height, id_bldg):
        '''
        ground... polygon of ground surface
        height... roof height
        id_bldg... integer as identifier
        '''

        self.id_bldg = id_bldg
        self.height = height
        self.ground = ground.buffer(1e-5)
        self.center = Point(list(ground.centroid.coords[0][:]))
        self.next_blgs = []

    def add_values(self, values):
        '''
        Append additional arbitrary information.
        values... information to append
        '''

        self.values = values



def grid_bldgs(grid, build_lst):
    '''
    Go through a list of buildings
    and assign them to grid cells.
    
    grid... object of grid class
    build_lst... list of building class objects
    '''

    y = grid.y
    x = grid.x   

    for build in build_lst:
        envelope = build.ground.envelope.boundary.coords[:]
        ll = envelope[0]
        lr = envelope[1]
        tr = envelope[2]
        tl = envelope[3]
        x_min = np.argmin(abs(x - min(ll[0], tl[0])))
        x_max = np.argmin(abs(x - max(lr[0], tr[0])))
        y_min = np.argmin(abs(y - min(lr[1], ll[1])))
        y_max = np.argmin(abs(y - max(tr[1], tl[1])))
        

        for i in xrange(y_min, y_max + 1, 1):
            for j in xrange(x_min, x_max + 1, 1):
                if build.ground.intersects(grid.cells[i, j].polygon):
                    grid.cells[i, j].int_bld(build.id_bldg)
	
                if grid.cells[i, j].polygon.contains(build.center):
                    grid.cells[i, j].cont_bld(build.id_bldg)



def calc_surf_flds(grid, bldg_lst, lim_d=1.0, ng=3):
    '''
    Compute surface fields for surface-flux and mixing-length computation.

    grid... object of grid class
    bldg_lst... list of building class objects
    lim_d... minimum possible distance of a building face to a grid-cell face
    ng... number of ghost cells used
    '''

    nx = grid.nx
    ny = grid.ny
    nz = grid.nz

    xinc = grid.xinc.copy()
    yinc = grid.yinc.copy()
    
    uhl = grid.z2.copy()
    ufl = (uhl[1:] + uhl[:-1])/2.0

    deff_v = np.empty([nz, ny, nx])
    deff_hx = np.empty([nz, ny, nx])
    deff_hy = np.empty([nz, ny, nx])

    deff_v[:] = (uhl[1:] - uhl[:-1]).reshape(nz, 1, 1) / 2.0 
    deff_hx[:] = xinc.reshape(1, 1, nx) / 2.0
    deff_hy[:] = yinc.reshape(1, ny, 1) / 2.0

    area_v = np.full([nz, ny, nx], 0.0)
    area_hx = np.full([nz, ny, nx], 0.0)
    area_hy = np.full([nz, ny, nx], 0.0)

    area_v[ng] = xinc.reshape(1, 1, nx) * yinc.reshape(1, ny, 1) 

    eps = 1e-30

    if not len(bldg_lst):
        return deff_v, deff_hx, deff_hy, area_v, area_hx, area_hy
 
    else:
        xinc = xinc.tolist()
        yinc = yinc.tolist()
        uhl = uhl.tolist()
        ufl = ufl.tolist()

        deff_v = (deff_v.flatten()).tolist()
        deff_hx = (deff_hx.flatten()).tolist()
        deff_hy = (deff_hy.flatten()).tolist()

        area_v = (area_v.flatten()).tolist()
        area_hx = (area_hx.flatten()).tolist()
        area_hy = (area_hy.flatten()).tolist()

    for i in range(ny):
        for j in range(nx):
            bld_inds = grid.cells[i, j].bld_int
            if not len(bld_inds):
                continue

            a_ground = 0.0
            for ind in bld_inds:
                a_ground += bldg_lst[ind].ground.intersection(grid.cells[i, j].polygon).area

            for k in range(ng, nz):
                arr_ind = k * nx * ny +  i * nx + j
                
                a_roof = 0.0
                free_vol = 0.0

                bld_inds_z = []

                for ind in bld_inds:
                    if bldg_lst[ind].height >= uhl[k]:
                        bld_inds_z.append(ind)
                for ind in bld_inds_z:
                    dh = uhl[k + 1] - min([bldg_lst[ind].height, uhl[k + 1]])
                    vol = bldg_lst[ind].ground.intersection(grid.cells[i, j].polygon).area * dh
                    a_roof += vol / (dh + eps)
                    free_vol += vol

                if k == ng:
                    free_vol += (grid.cells[i, j].polygon.area - a_ground) * uhl[k + 1]
                    a_roof += grid.cells[i, j].polygon.area - a_ground

                if free_vol < eps:
                    deff_v[arr_ind] = (uhl[k + 1] - uhl[k]) / 2.0
                    area_v[arr_ind] = 0.0 
                else:
                    deff_v[arr_ind] = max(0.5 * (free_vol / (a_roof + eps)), lim_d)
                    area_v[arr_ind] = a_roof 

                area_hx_tmp = 0.0
                area_hy_tmp = 0.0
                a_cross_section = 0.0

                blds_merged = []

                n = 0
                for ind in bld_inds_z:
                    ground = bldg_lst[ind].ground
                    volume = ground.area * bldg_lst[ind].height
                    bld_inds_z_tmp = [ind2 for ind2 in bld_inds_z if ind2 != ind]
                    for ind2 in bld_inds_z_tmp:
                        if bldg_lst[ind2].ground.touches(ground):
                            ground = ops.cascaded_union([ground, bldg_lst[ind2].ground])
                            volume += bldg_lst[ind2].ground.area * bldg_lst[ind2].height
                            bld_inds_z.remove(ind2)
                    bld_inds_z.remove(ind)                    
                    blds_merged.append((ground, volume / ground.area, n))
                    n += 1

                for bld in blds_merged:
                    lenx = 0.0
                    leny = 0.0
                    perim = bld[0].boundary.intersection(grid.cells[i, j].polygon)
                    a_cross_section += bld[0].intersection(grid.cells[i, j].polygon).area
                    if isinstance(perim, MultiLineString):
                        segments = perim
                    elif isinstance(perim, LineString):
                        segments = [perim]
                    else:
                        continue
                    for seg in segments:
                        pts = seg.coords[:]
                        pt_last = pts[0]
                        for pt in pts[1:]:
                            lenx += abs(pt[0] - pt_last[0])
                            leny += abs(pt[1] - pt_last[1])
                            pt_last = pt                    
                    area_hx_tmp += leny * (min([bld[1], uhl[k + 1]]) - uhl[k])
                    area_hy_tmp += lenx * (min([bld[1], uhl[k + 1]]) - uhl[k])

                if a_cross_section - grid.cells[i, j].polygon.area > -eps:
                    area_hx[arr_ind] = 0.0
                    area_hy[arr_ind] = 0.0
                else:
                    area_hx[arr_ind] = area_hx_tmp
                    area_hy[arr_ind] = area_hy_tmp

                area_blds_x = 0.0
                area_blds_y = 0.0
                vol_between_x = 0.0
                vol_between_y = 0.0

                for ind, bld in enumerate(blds_merged):
                    env = bld[0].intersection(grid.cells[i, j].polygon).envelope.boundary
                    min_x = min([pt[0] for pt in env.coords[:]])
                    max_x = max([pt[0] for pt in env.coords[:]])
                    min_y = min([pt[1] for pt in env.coords[:]])
                    max_y = max([pt[1] for pt in env.coords[:]])
                    square_x = Polygon([
                                           (grid.cells[i, j].x - xinc[j] / 2.0, min_y), 
                                           (grid.cells[i, j].x + xinc[j] / 2.0, min_y),
                                           (grid.cells[i, j].x + xinc[j] / 2.0, max_y), 
                                           (grid.cells[i, j].x - xinc[j] / 2.0, max_y),
                                           (grid.cells[i, j].x - xinc[j] / 2.0, min_y)
                                       ])
                    square_y = Polygon([
                                           (min_x, grid.cells[i, j].y - yinc[i] / 2.0),
                                           (max_x, grid.cells[i, j].y - yinc[i] / 2.0),
                                           (max_x, grid.cells[i, j].y + yinc[i] / 2.0),
                                           (min_x, grid.cells[i, j].y + yinc[i] / 2.0),
                                           (min_x, grid.cells[i, j].y - yinc[i] / 2.0)
                                       ])

                    vol_square_x = square_x.area * (min([bld[1], uhl[k + 1]]) - uhl[k])
                    vol_square_y = square_y.area * (min([bld[1], uhl[k + 1]]) - uhl[k])

                   
                    ground = bld[0].intersection(square_x)
                    square_x = square_x.difference(ground)                    
                    vol_square_x -= ground.area * (min([bld[1], uhl[k + 1]]) - uhl[k])

                    ground = bld[0].intersection(square_y)
                    square_y = square_y.difference(ground)
                    vol_square_y -= ground.area * (min([bld[1], uhl[k + 1]]) - uhl[k])

                    
                    bld_inds_z_tmp = [ind2 for ind2 in range(len(blds_merged)) if ind2 != ind]
                    heights = [blds_merged[ind2][1] for ind2 in bld_inds_z_tmp]
                    arg_sorted = np.argsort(np.array(heights), kind='mergesort')

                    # sorted from tallest to lowest building
                    bld_inds_z_tmp_sorted = [bld_inds_z_tmp[arg] for arg in arg_sorted[::-1]]
 
                    for ind2 in bld_inds_z_tmp_sorted:
                        
                        ground = blds_merged[ind2][0].convex_hull.intersection(square_x)
                        if not isinstance(ground, Polygon) or not isinstance(ground, MultiPolygon):
                            continue
                        pos_low_corner = np.argmin(np.array([pt[1] for pt in ground.boundary.coords[:]]))
                        pos_top_corner = np.argmax(np.array([pt[1] for pt in ground.boundary.coords[:]]))
                        low_corner = ground.boundary.coords[pos_low_corner]
                        top_corner = ground.boundary.coords[pos_top_corner]

                        if ground.centroid.coords[0][0] < bld[0].centroid.coords[0][0]:
                            extension = Polygon([
                                                    (grid.cells[i, j].x - xinc[j] / 2.0, low_corner[1]),
                                                    low_corner,
                                                    top_corner,
                                                    (grid.cells[i, j].x - xinc[j] / 2.0, top_corner[1]),
                                                    (grid.cells[i, j].x - xinc[j] / 2.0, low_corner[1])
                                                ])
                        else:
                            extension = Polygon([
                                                    (grid.cells[i, j].x + xinc[j] / 2.0, low_corner[1]),
                                                    low_corner,
                                                    top_corner,
                                                    (grid.cells[i, j].x + xinc[j] / 2.0, top_corner[1]),
                                                    (grid.cells[i, j].x + xinc[j] / 2.0, low_corner[1])
                                                ])
                        ground = ops.cascaded_union([ground, extension])
                                 
                        vol_square_x -= ground.area * (min([bld[1], blds_merged[ind2][1], uhl[k + 1]]) - uhl[k])
                        square_x = square_x.difference(ground)

                    for ind2 in bld_inds_z_tmp_sorted:
                        ground = blds_merged[ind2][0].convex_hull.intersection(square_y)
                        if not isinstance(ground, Polygon) or not isinstance(ground, MultiPolygon):
                            continue
                        pos_left_corner = np.argmin(np.array([pt[0] for pt in ground.boundary.coords[:]]))
                        pos_right_corner = np.argmax(np.array([pt[0] for pt in ground.boundary.coords[:]]))
                        left_corner = ground.boundary.coords[pos_left_corner]
                        right_corner = ground.boundary.coords[pos_right_corner]
                        if ground.centroid.coords[0][1] < bld[0].centroid.coords[0][1]:
                            extension = Polygon([
                                                    (left_corner[0], grid.cells[i, j].y - yinc[i] / 2.0),
                                                    (right_corner[0], grid.cells[i, j].y - yinc[i] / 2.0),
                                                    right_corner,
                                                    left_corner,
                                                    (left_corner[0], grid.cells[i, j].y - yinc[i] / 2.0)
                                                ])
                        else:
                                                 
                            extension = Polygon([
                                                    (left_corner[0], grid.cells[i, j].y + yinc[i] / 2.0),
                                                    (right_corner[0], grid.cells[i, j].y + yinc[i] / 2.0),
                                                    right_corner,
                                                    left_corner,
                                                    (left_corner[0], grid.cells[i, j].y + yinc[i] / 2.0)
                                                ])                                                  
                        ground = ops.cascaded_union([ground, extension])

                        vol_square_y -= ground.area * (min([bld[1], blds_merged[ind2][1], uhl[k + 1]]) - uhl[k])
                        square_y = square_y.difference(ground)                                              

                        vol_square_y -= ground.area * (min([bld[1], blds_merged[ind2][1], uhl[k + 1]]) - uhl[k])
                        square_y = square_y.difference(ground)

                    vol_between_x += vol_square_x
                    vol_between_y += vol_square_y
                    area_blds_x += (max_y - min_y) * (min([bld[1], uhl[k + 1]]) - uhl[k])
                    area_blds_y += (max_x - min_x) * (min([bld[1], uhl[k + 1]]) - uhl[k])

                deff = min([0.5 * vol_between_x / (area_blds_x + eps), xinc[j]])
                if vol_between_x < eps:
                    area_hx[arr_ind] = 0.0
                else: 
                    deff_hx[arr_ind] = max(deff, lim_d)

                deff = min([0.5 * vol_between_y / (area_blds_y + eps), yinc[i]])
                if vol_between_y < eps:
                    area_hy[arr_ind] = 0.0
                else:
                    deff_hy[arr_ind] = max(deff, lim_d)

                if a_ground >= grid.cells[i, j].polygon.area:
                    area_hx[arr_ind] = 0.0
                    area_hy[arr_ind] = 0.0


    deff_v = np.array(deff_v).reshape(nz, ny, nx)
    deff_hx = np.array(deff_hx).reshape(nz, ny, nx)
    deff_hy = np.array(deff_hy).reshape(nz, ny, nx)
    
    area_v = np.array(area_v).reshape(nz, ny, nx)
    area_hx = np.array(area_hx).reshape(nz, ny, nx)           
    area_hy = np.array(area_hy).reshape(nz, ny, nx)

    return deff_v, deff_hx, deff_hy, area_v, area_hx, area_hy
    


def calc_ffactors(grid, bldg_lst, n_hslice, angles=[0], method='geom_intersection', ff_min=1e-2, fv_min = 1e-40, fr_avg=0.2, ng=3):
    '''
    Compute the area- and volume scaling fields.

    grid... object of grid class
    bldg_lst... list of building class objects
    n_hslice... number of horizontal slices for subsampling
    angles... list of offset angles of slicing planes
    method... determine how to calculate the area-scaling field
              geom_intersection: use geometric intersection for well resolved buildings
              default: consider both adjacent cell volumes to estimate the cell-face area scaling factors for diffusive obstacles (much slower)
    ff_min... minimum area-scaling factor
    fv_min... minimum volume-scaling factor
    fr_avg... fraction of cell volume to incorporate in the averaging for non-assigned cell faces
    ng... number of ghost cells used 
    '''

    nx = grid.nx
    ny = grid.ny
    nz  = grid.nz

    xinc = grid.xinc.copy()
    yinc = grid.yinc.copy()

    u_hl = grid.z2.copy()
    u_fl = (u_hl[1:] + u_hl[:-1]) / 2.0

    eps = 1e-30

    ffx = np.full([nz, ny, nx + 1], 1.0)
    ffy = np.full([nz, ny + 1, nx], 1.0)
    ffz = np.full([nz + 1, ny, nx], 1.0)
    fvol = np.full([nz, ny, nx], 1.0)
    
    # ground surface
    ffx[:ng] = 0.0
    ffy[:ng] = 0.0
    ffz[:ng + 1] = 0.0
    fvol[:ng] = fv_min

    if not len(bldg_lst):
        return ffx, ffy, ffz, fvol

    else:

        ffx = (ffx.flatten()).tolist()
        ffy = (ffy.flatten()).tolist()
        ffz = (ffz.flatten()).tolist()
        fvol = (fvol.flatten()).tolist()
        xinc = xinc.tolist()
        yinc = yinc.tolist()
        u_hl = u_hl.tolist()
        u_fl = u_fl.tolist()

    for i in xrange(ny):
        for j in xrange(nx):
            if not len(grid.cells[i,j].bld_int):
                pass
            else:
                center = [grid.cells[i, j].x, grid.cells[i,j].y]
                bld_ids = grid.cells[i, j].bld_int
                polygons = [(bldg_lst[bld_id].ground).intersection(grid.cells[i, j].polygon) for bld_id in bld_ids]

                for k in range(ng, nz, 1):
                    arr_ind = k * nx * ny +  i * nx + j
                    arr_ind_kp = (k + 1) * nx * ny +  i * nx + j
                    arr_indy = k * nx * (ny + 1) +  (i + 1) * nx + j
                    arr_indy_ip = k * nx * (ny + 1) +  (i + 1) * nx + j
                    arr_indx = k * (nx + 1) * ny +  i * (nx + 1) + j + 1
                    arr_indx_jp = k * (nx + 1) * ny +  i * (nx + 1) + j + 1

                    vol_cell = grid.cells[i, j].polygon.area * (u_hl[k + 1] - u_hl[k])
                    bld_ids_tmp = [bld_id for bld_id in bld_ids if bldg_lst[bld_id].height > u_hl[k]]
                    if not len(bld_ids_tmp):
                        continue
                    polygons = [(bldg_lst[bld_id].ground).intersection(grid.cells[i, j].polygon) for bld_id in bld_ids_tmp]
                    heights = [bldg_lst[bld_id].height for bld_id in bld_ids_tmp]
                    vol_blds = sum([polygon.area * (min(heights[m], u_hl[k + 1]) - u_hl[k]) for m, polygon in enumerate(polygons)])
                    fvol[arr_ind] = np.maximum(1.0 - vol_blds / vol_cell, fv_min)

                    ffz[arr_ind_kp] = (
                                           1.0 - sum([polygon.area for m, polygon in enumerate(polygons) if heights[m] >= u_hl[k + 1]]) / 
                                           grid.cells[i, j].polygon.area 
                                       )
                    min_val_l = 1.0
                    min_val_r = 1.0
                    min_loc = 0.0
                    new_val = 1.0                        

                    for angle in angles:  
                        intersections = []
                        pos = []
                        for h in range(n_hslice):
                            shift = np.tan(angle / 180.0 * np.pi) * yinc[i] / 2.0
                            pos_x = - xinc[j] / 2. + h * xinc[j] / (n_hslice - 1) 
                            pos.append(pos_x)
              
                            line = LineString(
                                                 [[max(min(pos_x + center[0] - shift, center[0] + xinc[j] / 2.), 
                                                   center[0] - xinc[j] / 2.), center[1] - yinc[i] / 2.],
                                                 [max(min(pos_x + center[0] + shift, center[0] + xinc[j] / 2.),  
                                                  center[0] - xinc[j] / 2.), center[1] + yinc[i] / 2.]]

                                             )
                                             
                            a_cell = line.length * (u_hl[k + 1] - u_hl[k])
                            a_blds = 0.0

                            for m, polygon in enumerate(polygons):
                                if line.intersects(polygon):
                                    a_blds += polygon.intersection(line).length * (max(min(heights[m], u_hl[k + 1]), u_hl[k]) - u_hl[k])

                            frac_bld = a_blds / a_cell
                            intersections.append((1.0 - frac_bld))
                    
                        if method == 'geom_intersection':
                            ffx[arr_indx] = min(ffx[arr_indx], intersections[0])
                            ffx[arr_indx_jp] = min(ffx[arr_indx_jp], intersections[-1])
                            continue
                        else:    # diffusive intersection
                            pass

                        min_val_pot = min(intersections)

                        if min_val_pot < new_val:
                            min_loc = np.mean([pos[n] for n, num in enumerate(intersections) if num == min(intersections)])
                            new_val = min_val_pot

                        if new_val <= ff_min:                   
                            break

                    n_avg = max((1, int(n_hslice * fr_avg)))

                    if min_loc <= 0.0:

                        ffx[arr_indx] = min(ffx[arr_indx], new_val)
                        ffx[arr_indx_jp] = min(ffx[arr_indx_jp], sum(intersections[-n_avg:]) / (n_avg + 1e-20))
                    else:

                        ffx[arr_indx_jp] = min(ffx[arr_indx_jp], new_val)
                        ffx[arr_indx] = min(ffx[arr_indx], sum(intersections[:n_avg]) / (n_avg + 1e-20))

                    min_val_l = 1.0
                    min_val_r = 1.0
                    new_val = 1.0
                    min_loc = 0.0

                    for angle in angles:
                        pos = []
                        intersections = []
                        for h in range(n_hslice):
                            shift = np.tan(angle / 180.0 * np.pi) * xinc[j] / 2.0
                            pos_y = - yinc[i] / 2. + h  * yinc[i] / (n_hslice - 1)  
                            pos.append(pos_y)
                            line = LineString(
                                                 [[center[0] - xinc[j] / 2., 
                                                  max(min(pos_y + center[1] - shift, center[1] + yinc[i] / 2.), center[1] - yinc[i] / 2)],
                                                 [center[0] + xinc[j] / 2., 
                                                  max(min(pos_y + center[1] + shift, center[1] + yinc[i] / 2.), center[1] - yinc[i] / 2)]]
                                             )
                                             

                            a_cell = line.length * (u_hl[k + 1] - u_hl[k])
                            a_blds = 0.0

                            for m, polygon in enumerate(polygons):
                                if line.intersects(polygon):
                                    a_blds += polygon.intersection(line).length * (max(min(heights[m], u_hl[k + 1]), u_hl[k]) - u_hl[k])

                            frac_bld = a_blds / a_cell
                            intersections.append((1.0 - frac_bld))


                        if method == 'geom_intersection':
                            ffy[arr_indy] = min(ffy[arr_indy], intersections[0])
                            ffy[arr_indy_ip] = min(ffy[arr_indy_ip], intersections[-1])
                            continue
                        else:
                            pass

                        min_val_pot = min(intersections)

                        if min_val_pot < new_val:
                            min_loc = np.mean([pos[n] for n, num in enumerate(intersections) if num == min(intersections)])
                            new_val = min_val_pot

                        if new_val <= ff_min:
                            break                                      

                    n_avg = max((1, int(n_hslice * fr_avg)))
                    if min_loc <= 0.0:
                        ffy[arr_indy] = min(ffy[arr_indy], new_val)
                        ffy[arr_indy_ip] = min(ffy[arr_indy_ip], sum(intersections[-n_avg:]) / (n_avg + 1e-20))

                    else:

                        ffy[arr_indy_ip] = min(ffy[arr_indy_ip], new_val)
                        ffy[arr_indy] = min(ffy[arr_indy], sum(intersections[:n_avg]) / (n_avg + 1e-20))


    fvol = (np.array(fvol)).reshape(nz, ny, nx)
    ffx = (np.array(ffx)).reshape(nz, ny, nx + 1)
    ffy = (np.array(ffy)).reshape(nz, ny + 1, nx)
    ffz = (np.array(ffz)).reshape(nz + 1, ny, nx)

    fvol = np.maximum(np.minimum(1.0, fvol), 0.0)
    ffz = np.maximum(np.minimum(1.0, ffz), 0.0)
    ffx[:, :, 1:-1] = np.maximum(ffx[:, :, 1:-1], ff_min)
    ffy[:, 1:-1] = np.maximum(ffy[:, 1:-1], ff_min)      

    return ffx, ffy, ffz, fvol             
    

def plot_buildings(grid, bldg_lst, filename, filetype, plot_gridlines=True, figsize=10):
    '''
    Creates a plot of the ground intersection
    from a building configuration.

    grid... object of grid class
    bldg_lst... list of objects of building type
    filename... filename of plot
    filetype... filetype of plot
    plot_gridlines... if True, grid lines are added to the plot
    figsize... figsize in plt.figure
    '''    

    x2 = grid.x2
    y2 = grid.y2
    nx = grid.nx
    ny = grid.ny

    x_min, x_max = x2[0], x2[-1]
    y_min, y_max = y2[0], y2[-1]
    
    x_span = x_max - x_min
    y_span = y_max - y_min

    if x_span > y_span:   
        fig = plt.figure(figsize=[figsize, figsize * y_span / x_span])
    else:
        fig = plt.figure(figsize=[figsize * x_span / y_span, figsize])


    ax = fig.add_subplot(1,1,1)
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)

    for build in bldg_lst:
        patch = PolygonPatch(build.ground, fc=[0.5, 0.5, 0.5, 1], ec= [0, 0, 0, 0])
        ax.add_patch(patch)

    if plot_gridlines:
        for i in range(ny):
            for j in range(nx):
                patch = PolygonPatch(grid.cells[i,j].polygon, fc=[0, 0, 0, 0], ec= [0, 0, 0, 1], linewidth=0.2)
                ax.add_patch(patch)
    plt.savefig('./' + filename + '.' + filetype)
