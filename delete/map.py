import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style()
import pandas as pd
import numpy as np
boundaries = pd.read_csv('Boundaries.csv')
x = boundaries['X'].values
y = boundaries['Y'].values

plt.figure()
plt.scatter(x,y)

# This is the top right corrner with coordinate of (259987, 794418)
top_right_i = 98
top_right = (x[top_right_i], y[top_right_i])
plt.scatter(*top_right, c='red', label='98')

# This is the bottom point with coordinate of (259907, 794125)
bottom_i = 6
bottom = (x[bottom_i], y[bottom_i])
plt.scatter(*bottom, c='green', label=str(bottom_i))
plt.legend()


from scipy.interpolate import interp1d
f_x = interp1d([bottom[0],top_right[0]], [259907,259987], bounds_error=False, fill_value="extrapolate")
f_y = interp1d([bottom[1],top_right[1]], [794125,794418], bounds_error=False, fill_value="extrapolate")

bound = np.c_[f_x(boundaries['X']),f_y(boundaries['Y'])]

plt.figure()
plt.scatter(bound[:,0], bound[:,1])
plt.ticklabel_format(useOffset=False)

x_coor_range = np.arange(259900,260010,1)
y_coor_range = np.arange(794100,794450,1)

x_coor_grid, y_coor_grid = np.meshgrid(x_coor_range, y_coor_range)
z_grid = np.empty((x_coor_grid.shape))
z_grid[:] = -1

valued_z = []
valued_xy = []

for instance in bound:
    x_closest = x_coor_range[np.argmin(np.abs(x_coor_range - instance[0]))]
    y_closest = y_coor_range[np.argmin(np.abs(y_coor_range - instance[1]))]
    valued_xy.append([x_closest, y_closest])
    valued_z.append(0)

valued_xy.append([259960, 794300])
valued_z.append(1)
valued_xy.append([259920, 794175])
valued_z.append(1)
valued_xy.append([259950, 794400])
valued_z.append(1)
valued_xy.append([259960, 794235])
valued_z.append(1)
valued_xy.append([259960, 794350])
valued_z.append(1)
valued_xy.append([259940, 794200])
valued_z.append(1)
valued_xy.append([259913, 794418])
valued_z.append(1)
valued_xy.append([259906, 794134])
valued_z.append(1)
valued_xy.append([259914, 794336])
valued_z.append(-1)

valued_xy = np.array(valued_xy)
valued_z = np.array(valued_z)

xy = np.c_[x_coor_grid.flatten(),y_coor_grid.flatten()]

from scipy.interpolate import griddata
interpolated_z = griddata(valued_xy, valued_z, xy, 'cubic').reshape(z_grid.shape)
interpolated_z = np.where(interpolated_z<0, np.nan,interpolated_z)
plt.contourf(x_coor_grid,y_coor_grid,interpolated_z)


out_of_bound_i = np.argwhere(np.isnan(interpolated_z))
z = np.ones(x_coor_grid.shape)
z[out_of_bound_i[:,0],out_of_bound_i[:,1]] = 0
plt.figure()
plt.contourf(x_coor_grid, y_coor_grid, z, [-1,0,1,2])

np.save('x.np',x_coor_grid)
np.save('y.np',y_coor_grid)
np.save('z.np',z)

plt.ticklabel_format(useOffset=False)

valued_z = []
valued_xy = []
for contour_value in [160,165,170,175]:
    data = pd.read_csv('{}contour.csv'.format(contour_value))
    for instance in data.values:
        valued_xy.append(instance)
        valued_z.append(contour_value)
valued_z = np.array(valued_z)
valued_xy = np.array(valued_xy)

'''out_of_bound_i = np.argwhere(interpolated_z<0)
z_grid[:] = 0
z_grid[out_of_bound_i] = np.nan
plt.figure()
plt.contourf(x_coor_grid,y_coor_grid,z_grid)'''

'''

for instance in c165:
    x_closest = x_coor_range[np.argmin(np.abs(x_coor_range - instance[0]))]
    y_closest = y_coor_range[np.argmin(np.abs(y_coor_range - instance[1]))]
    valued_xy.append([x_closest, y_closest])
    valued_z.append(165)

for instance in c170:
    x_closest = x_coor_range[np.argmin(np.abs(x_coor_range - instance[0]))]
    y_closest = y_coor_range[np.argmin(np.abs(y_coor_range - instance[1]))]
    valued_xy.append([x_closest, y_closest])
    valued_z.append(170)

'''

'''valued_xy.append([259908,794458])
valued_z.append(175)
valued_xy.append([259991,794423])
valued_z.append(175)
valued_xy.append([259852,794028])
valued_z.append(160)
valued_xy.append([260010,794194])
valued_z.append(165)'''

fig = plt.figure()
interpolated_z = griddata(valued_xy, valued_z, xy, 'cubic').reshape(z_grid.shape)
interpolated_z[out_of_bound_i[:,0],out_of_bound_i[:,1]] = np.nan
cs_contourf = plt.contourf(x_coor_grid,y_coor_grid,interpolated_z,1000, cmap='afmhot_r',vmax=190)
clabel = plt.contour(x_coor_grid,y_coor_grid,interpolated_z,[160,162.5,165,167.5,170,172.5],colors='red', linestyles='dashed')
plt.clabel(clabel, colors='red')
plt.ticklabel_format(useOffset=False)

out_of_bound_i = np.argwhere(np.isnan(interpolated_z))
z = np.ones(x_coor_grid.shape)
z[out_of_bound_i[:,0],out_of_bound_i[:,1]] = 0
plt.contour(x_coor_grid, y_coor_grid, z,colors='black', linewidths =2)
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.tight_layout()