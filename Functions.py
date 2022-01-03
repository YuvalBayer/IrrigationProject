import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style()
sns.set_context("notebook", font_scale=1.5)
from scipy.interpolate import griddata
from scipy.optimize import fsolve

'''
-------------------------------------------------From NB 1--------------------------------------------------------------

See documentation in NB1
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

'''
-------------------------------------------------From NB 2--------------------------------------------------------------

See documentation in NB2
'''
def calculate_distance(line):
    return np.sqrt(np.sum((line[1] - line[0]) ** 2))

def calculate_coor_slope(line):
    return (line[1][1] - line[0][1])/(line[1][0] - line[0][0])

def calculate_line_point_from(r,line):
    result = np.array([])
    for i in [0,1]:
        x1 = line[i][0]
        y1 = line[i][1]
        D = 6 * r
        m = calculate_coor_slope(line_0)
        b = y1 - np.sqrt(D ** 2 / (1 + m ** 2))
        a = x1 - m * (b - y1)
        result = np.append(result, np.array([a,b]))
    return result.reshape(2,2)

def import_lines():
    return np.load('LinesData/lines.npy')

def import_main_line():
    return np.load('LinesData/main_line.npy')

def get_profile(line, num=1000):
    x_line = np.linspace(line[0, 0], line[1, 0], num)
    y_line = np.linspace(line[0, 1], line[1, 1], num)
    xy_line = np.c_[x_line, y_line]
    L = calculate_distance(line)
    x = np.linspace(0, L, num)  # The distance from main line
    altitude = interpolate_altitude(xy_line).flatten()
    return np.c_[x,altitude]

'''
-------------------------------------------------From NB 3--------------------------------------------------------------

See documentation in NB3
'''

class DripLineEstimator():
    def __init__(self, L_space=0.4, D=0.016, C=140, a=0.999, x=0.478):
        self.L_space = L_space
        self.D = D  # [m]
        self.C = C
        self.alpha = 1.852
        self.beta = 4.87
        self.a = a
        self.x = x

    def drip_flow(self, pressure):  # Getting the pressure in [m]
        pressure = pressure * (9800 / 1e5)  # [bar]
        flow = self.a * pressure ** self.x  # [l/h]
        return flow / (1000 * 3600)  # [m^3/s]

    def run(self,P_terminal_guess):
        self.P_line = np.array([P_terminal_guess])
        self.Q_line = np.array([])

        for i in np.arange(-1, -len(self.x_points), -1):
            z = self.z_points[i]
            z_pre = self.z_points[i - 1]
            q = self.drip_flow(self.P_line[-1])
            self.Q_line = np.append(self.Q_line, q)
            hf = self.L_space * ((np.sum(self.Q_line) / self.C) ** self.alpha) * (10.67 / (self.D ** self.beta))
            self.P_line = np.append(self.P_line, hf + self.P_line[-1] + z - z_pre)

        # Reversing the order
        self.P_line = self.P_line[::-1]
        self.Q_line = self.Q_line[::-1]

        # Seperating the main line pressure from the drip line
        self.P_main_line = self.P_line[0]
        self.P_line = self.P_line[1:]

        return self.P_line, self.Q_line, self.P_main_line

    def objective(self, variable):
        P_terminal_guess = variable[0]
        _, _, P_main_line = self.run(P_terminal_guess)
        return P_main_line - self.P_main_line_value

    def estimate(self, P_main_line_value, x_main_line, x_emitters, z_main_line, z_emitters):
        self.P_main_line_value = P_main_line_value
        self.x_points = np.append(x_main_line, x_emitters)
        self.z_points = np.append(z_main_line, z_emitters)

        # Finding the right terminal value
        P_terminal_value = fsolve(self.objective, P_main_line_value)[0]

        # Returning all of the pressures and flows values for the drip line
        return self.run(P_terminal_value)
