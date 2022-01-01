import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style()
sns.set_context("notebook", font_scale=1.5)

'''
Building the interpolation function which get x and y coordinate and returns the interpolated altitude
'''

from scipy.interpolate import griddata

valued_z = []
valued_xy = []
for contour_value in [160, 165, 170, 175]:
    data = pd.read_csv('AltitudeData/{}contour.csv'.format(contour_value))
    for instance in data.values:
        valued_xy.append(instance)
        valued_z.append(contour_value)
valued_z = np.array(valued_z)
valued_xy = np.array(valued_xy)

def interpolate_altitude(xy):
    return griddata(valued_xy, valued_z, xy, 'cubic')


'''
Creating:
    1) The x and y grid which are 2D arrays holding all of the spatial data
    2) The altitude grid which is a 2D array holding all of the coordinates altitudes
'''

x = np.load('CoordinatesData/x.npy')
y = np.load('CoordinatesData/y.npy')
z = np.load('CoordinatesData/z.npy') # 2D array - values of 1 is in our plot, 0 out of the plot
out_of_bound_i = np.argwhere(z == 0) # Indices where the coordinates are out of our plot

xy = np.c_[x.flatten(), y.flatten()]
altitude = interpolate_altitude(xy).reshape(z.shape) # The altitude array
altitude[out_of_bound_i[:, 0], out_of_bound_i[:, 1]] = np.nan

# Plotting map
fig, axis = plt.subplots(figsize=(15, 8))
cs_contourf = plt.contourf(x, y, altitude, 1000, cmap='afmhot_r', vmax=190)
clabel = plt.contour(x, y, altitude, [165, 167.5, 170, 172.5],
                     colors='red',
                     linestyles='dashed')
plt.ticklabel_format(useOffset=False)

cbar = fig.colorbar(cs_contourf, ticks=[165, 172.5])
cbar.ax.set_yticklabels([165, 172.5], color='red')
cbar.set_label('Altitude [m]')

plt.contour(x, y, z, colors='black', linewidths=1.5)
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.tight_layout()
axis.set_aspect('equal', 'box')
plt.ylim(794100, 794450)
plt.xlim(259775, 260150)

# plt.savefig('map.png')

irr_area_points = pd.read_csv('CoordinatesData/IrrigationAreasPoints.csv').values

def plot_line(indices):
    i1, i2 = indices
    points = irr_area_points[[i1, i2]]
    plt.plot(points[:, 0], points[:, 1], c='blue')

def calculate_distance(indices):
    i1, i2 = indices
    return np.sqrt(np.sum((irr_area_points.iloc[i2] - irr_area_points.iloc[i1]) ** 2))

# Scattering the irrigation area corners
for point, i in zip(irr_area_points, range(0,len(irr_area_points))):
    plt.scatter(point[0], point[1], c='blue')
    if i in [0,2,5]:
        plt.annotate(str(i), (point[0]-15, point[1]), c='white')
    else:
        if i in [4,6,7]:
            plt.annotate(str(i), (point[0] + 15, point[1]-5), c='white')
        else:
            plt.annotate(str(i), (point[0]+15, point[1]), c='white')


# Plotting the irrigation area lines
for indices in [[0, 1], [1, 3], [3, 2], [2, 0], [2, 4], [4, 7], [7, 6], [6, 5], [5, 2]]:
    plot_line(indices)