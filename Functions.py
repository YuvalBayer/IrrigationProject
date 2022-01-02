import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style()
sns.set_context("notebook", font_scale=1.5)
from scipy.interpolate import griddata


'''
All of the documentation is in notebook of week 2
'''
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


x = np.load('CoordinatesData/x.npy')
y = np.load('CoordinatesData/y.npy')
z = np.load('CoordinatesData/z.npy') # 2D array - values of 1 is in our plot, 0 out of the plot
out_of_bound_i = np.argwhere(z == 0) # Indices where the coordinates are out of our plot

xy = np.c_[x.flatten(), y.flatten()]
altitude = interpolate_altitude(xy).reshape(z.shape) # The altitude array
altitude[out_of_bound_i[:, 0], out_of_bound_i[:, 1]] = np.nan

# Plotting map
def plot_map(is_contour_lines=True, is_boundary=True):
    fig, axis = plt.subplots(figsize=(15, 8))
    cs_contourf = plt.contourf(x, y, altitude, 1000, cmap='afmhot_r', vmax=190)
    if is_contour_lines:
        clabel = plt.contour(x, y, altitude, [165, 167.5, 170, 172.5],
                             colors='red',
                             linestyles='dashed')
        plt.ticklabel_format(useOffset=False)
        cbar = fig.colorbar(cs_contourf, ticks=[165, 167.5, 170, 172.5])
        cbar.ax.set_yticklabels([165, 167.5, 170, 172.5], color='red')
        cbar.set_label('Altitude [m]')
    if is_boundary:
        plt.contour(x, y, z, colors='black', linewidths=1.5)
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.tight_layout()
    axis.set_aspect('equal', 'box')
    plt.ylim(794100, 794450)
    plt.xlim(259775, 260150)
