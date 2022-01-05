import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.interpolate import griddata
from scipy.optimize import fsolve

sns.set_context("notebook", font_scale=1.5)


'''
-------------------------------------------------From NB 1--------------------------------------------------------------

See documentation in NB1
'''

# Arranging the altitude data from the map into numpy array
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
    '''
    The function interpolate the altitude of a point

    :param xy: 2D array with X and Y columns
    :return: Array of altitude
    '''
    return griddata(valued_xy, valued_z, xy, 'cubic')


# Interpolating the altitude for all plot's coordinates
x = np.load('CoordinatesData/x.npy')
y = np.load('CoordinatesData/y.npy')
z = np.load('CoordinatesData/z.npy') # 2D array - values of 1 is in our plot, 0 out of the plot
out_of_bound_i = np.argwhere(z == 0) # Indices where the coordinates are out of our plot

xy = np.c_[x.flatten(), y.flatten()]
altitude = interpolate_altitude(xy).reshape(z.shape) # The altitude array
altitude[out_of_bound_i[:, 0], out_of_bound_i[:, 1]] = np.nan



def plot_map(is_contour_lines=True, is_boundary=True):
    '''
    Plotting the field topography

    :param is_contour_lines: True for plotting the altitude contours
    :param is_boundary: True for plotting thick boundaries of the field
    '''
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
    plt.xlim(259890, 260020)
    plt.xticks(ticks=[259900,260000], labels=[259900,260000])

'''
-------------------------------------------------From NB 2--------------------------------------------------------------

See documentation in NB2
'''
def calculate_distance(line):
    '''
    Calculate the distance between two coordinate points

    :param line: lines in the map (2x2 array, each row is x and y coordinates as columns)
    :return: the distance
    '''
    return np.sqrt(np.sum((line[1] - line[0]) ** 2))

def calculate_coor_slope(line):
    '''
    Calculating the coordinate slope between two points

    :param line: lines in the map (2x2 array, each row is x and y coordinates as columns)
    :return: slope
    '''
    return (line[1][1] - line[0][1])/(line[1][0] - line[0][0])

def import_lines():
    '''
    Importing all of the drip lines

    :return: 2D array (each line is 1x4 array of 4 points - 2 x and 2 y coordinates)
    '''
    return np.load('LinesData/lines.npy')

def import_main_line():
    '''
    Importing the main line point coordinates

    :return: 2D array (each point (row) is 1x2 array of X and Y coordinates)
    '''
    return np.load('LinesData/main_line.npy')

def get_profile(line, num=1000):
    '''
    Calculating the altitude profile of drip line

    :param line: line in the map (2x2 array, each row is x and y coordinates as columns)
    :param num: number of data points
    :return: 2D array - the columns: x[m] - the distance from main line connection point
                                     altitude [m] - altitude
    '''
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
    '''
        This is an estimator for the flow and pressure of drip line (per each emitter)
    '''
    def __init__(self, space_drip=0.4, D_drip=0.016, C_drip=140, a=0.999, x=0.478):
        '''

        :param space_drip: The space between emitter [m]
        :param D_drip: The drip pipe diameter [m]
        :param C_drip: The drip pipe Hazen-Williams Coefficient [-]
        :param a: The dripper flow constant [l h^-1 bar^-1]
        :param x: The dripper flow exponent [-]
        '''
        self.space_drip = space_drip
        self.D_drip = D_drip
        self.C_drip = C_drip
        self.alpha = 1.852
        self.beta = 4.87
        self.a = a
        self.x = x

    def drip_flow(self, pressure):
        '''
        The function calculating a flow of a dripper by a given pressure (according to a*P^x)

        :param pressure: The pressure value [m]
        :return: The flow value [m^3/s]
        '''
        pressure = pressure * (9800 / 1e5)  # [bar]
        flow = self.a * pressure ** self.x  # [l/h]
        return flow / (1000 * 3600)  # [m^3/s]

    def run(self,P_terminal_guess):
        '''
        Running a simulation of calculating the pressure and flow from the terminal point of the drip line

        :param P_terminal_guess: The pressure at the terminal point of the drip line
        :return: 1) Array of pressures for all of the emitters
                 2) Array of flows for all of the emitters
                 3) The pressure value at the connection point of the drip line to the main line
        '''
        self.P_line = np.array([P_terminal_guess])
        self.Q_line = np.array([])

        for i in np.arange(-1, -len(self.x_points), -1):
            z = self.z_points[i]
            z_pre = self.z_points[i - 1]
            q = self.drip_flow(self.P_line[-1])
            self.Q_line = np.append(self.Q_line, q)
            hf = self.space_drip * ((np.sum(self.Q_line) / self.C_drip) ** self.alpha) * (10.67 / (self.D_drip ** self.beta))
            self.P_line = np.append(self.P_line, hf + self.P_line[-1] + z - z_pre)

        # Reversing the order
        self.P_line = self.P_line[::-1]
        self.Q_line = self.Q_line[::-1]

        # Seperating the main line pressure from the drip line
        self.P_main_line = self.P_line[0]
        self.P_line = self.P_line[1:]

        return self.P_line, self.Q_line, self.P_main_line

    def objective(self, variable):
        '''
        Objective function that returns difference between the (1) running and the (2) true value the of:
        the pressure at the connection point of the drip line to the main line.

        This objective function is in depend on the pressure at the end of the drip line.

        The goal is to find the pressure at the end of the drip line for which the objective function is zero (i.e.,
        for which the pressure at the connection point of the drip line to the main line is true)

        :param variable: Guess of the pressure at the end of the drip line
        :return: True value of the pressure at the end of the drip line
        '''
        P_terminal_guess = variable[0]
        _, _, P_main_line = self.run(P_terminal_guess)
        return P_main_line - self.P_main_line_value

    def estimate(self, P_main_line_value, x_main_line, x_emitters, z_main_line, z_emitters):
        '''
        A function the returns the true values of pressure and flow of the drip line by a given pressure at the
        connection point of the drip line to the main line

        :param P_main_line_value: The pressure at the connection point of the drip line to the main line
        :param x_main_line: the distance of the connection point of the drip line to the main line from main line = 0
        :param x_emitters: the distance from the main line of all emitters
        :param z_main_line: the altitude of the connection point of the drip line to the main line
        :param z_emitters: the altitude of all emitters
        :return:
        '''
        self.P_main_line_value = P_main_line_value
        self.x_points = np.append(x_main_line, x_emitters)
        self.z_points = np.append(z_main_line, z_emitters)

        # Finding the right terminal value
        P_terminal_value = fsolve(self.objective, P_main_line_value)[0]

        # Returning all of the pressures and flows values for the drip line
        return self.run(P_terminal_value)



'''
-------------------------------------------------From NB 4--------------------------------------------------------------

See documentation in NB4
'''


class System():
    '''
    Modeling the irrigation system
    '''
    def __init__(self, lines, C_main, D_main, C_drip, D_drip, space_drip, a, x):
        '''

        :param lines: all of the lines in the system (each line is 1x4 array of 4 points - 2 x and 2 y coordinates)
        :param C_main: The main pipe diameter [m]
        :param D_main: The main pipe Hazen-Williams Coefficient [-] -> Array which can express the diameter change
        :param C_drip: The drip pipe diameter [m]
        :param D_drip: The drip pipe Hazen-Williams Coefficient [-]
        :param space_drip: The space between emitter [m]
        :param a: The dripper flow constant [l h^-1 bar^-1]
        :param x: The dripper flow exponent [-]
        '''
        self.lines = lines
        self.C_main = C_main
        self.D_main = D_main  # array
        self.C_drip = C_drip
        self.D_drip = D_drip
        self.a = a
        self.x = x
        self.space_drip = space_drip
        self.alpha = 1.852
        self.beta = 4.87

    def estimate_line(self, line, main_line_pressure):
        '''
        Estimating a drip line flow and pressure

        :param line:
        :param main_line_pressure: The pressure at the connection point of the drip line to the main line
        :return: All of the emitters pressure and flow (in [m] and [m^3/s] respectively)
        '''
        # Drip line estimator
        est = DripLineEstimator(self.space_drip, self.D_drip, self.C_drip, self.a, self.x)

        # Profile of the drip line
        profile = get_profile(line, 1000)
        x = profile[:, 0].flatten()  # Distance from main line
        zx = profile[:, 1].flatten()  # Altitude

        L = calculate_distance(line)  # Drip line length

        # Two options for the last x_emitter value - 66 or 24.8, according to L
        last_value = [66, 24.8][np.argmin(np.absolute((np.array([66, 25]) - L)))]

        # Adding the closest location of the emitter
        x_emitters = np.arange(0.4, last_value + 0.4, 0.4)
        i = []
        for e in x_emitters:
            i.append(np.argmin(np.absolute(e - x.flatten())))

        x_emitters = x[i]  # All of the emitter x values (distance from main line)
        z_emitters = zx[i]  # All of the emitter z values (altitude)

        x_main_line = x[0]  # Main line distance from main line = 0 (it's necessary for later calculations)
        z_main_line = zx[0]  # Main line altitude (in the connection to the drip line)

        # Estimating all of the emitters pressure and flow, and the main line pressure.
        P_line, Q_line, P_main_line = est.estimate(main_line_pressure, x_main_line, x_emitters, z_main_line, z_emitters)

        return P_line, Q_line

    def run(self, P_terminal):
        '''
        Running a simulation of calculating the pressure and flow from the terminal point of the main line to the begining

        :param P_terminal: The terminal value of the pressure at the last connection point
        :return: None, all of the data is saved as attributes of the object
        '''

        self.P_main = np.array([P_terminal]) # All of the pressure connection points
        self.Q_main = np.array([]) # All of the flows of the connection points
        self.L_main = np.array([]) # All of the distances of the connection points from the first connection point
        self.Z_main = np.array([]) # All of the altitude of the connection points

        # All of the drip lines flow and pressure (of each emitter)
        self.q_emitters = []
        self.p_emitters = []

        # Running all line from the last to first (excluding the first line)
        for i in np.arange(-1, -len(self.lines), -1):
            # Calculating all of the drip line's emitters flow and pressure,
            # using the main line pressure - P_main[-1]
            P_dripline, Q_dripline = self.estimate_line(self.lines[i].reshape(2, 2), self.P_main[-1])

            # Saving the drip line data
            self.q_emitters.append(Q_dripline)
            self.p_emitters.append(P_dripline)

            # Adding the whole drip line flow to the main line data
            self.Q_main = np.append(self.Q_main, np.sum(Q_dripline))

            # Current (N) and previous (N-1) connection points
            cur_point = self.lines[i].reshape(2, 2)[0]
            pre_point = self.lines[i - 1].reshape(2, 2)[0]

            # Calculating the distance to the previous point
            section_L = np.concatenate((cur_point, pre_point))
            self.L_main = np.append(self.L_main, calculate_distance(section_L.reshape(2, 2)))

            # Calculating the altitude for the current point and the previous (N-1)
            z = interpolate_altitude(cur_point)[0]
            z_pre = interpolate_altitude(pre_point)[0]

            # Adding main line altitude to data
            self.Z_main = np.append(self.Z_main, z)

            hf = self.L_main[-1] * ((np.sum(self.Q_main) / self.C_main) ** self.alpha) * (10.67 / (self.D_main[i] ** self.beta))
            self.P_main = np.append(self.P_main, hf + self.P_main[-1] + z - z_pre)

        # Adding the first drip line to the rest of the data:
        P_dripline, Q_dripline = self.estimate_line(self.lines[0].reshape(2, 2), self.P_main[-1])
        self.q_emitters.append(Q_dripline)
        self.p_emitters.append(P_dripline)
        self.Z_main = np.append(self.Z_main, z_pre)  # The first connection point altitude
        self.Q_main = np.append(self.Q_main, np.sum(Q_dripline))  # The first drip line whole flow

        # Reversing all results
        self.x_main = np.append(0, np.cumsum(self.L_main[::-1]))  # The distance of all conection points from the top
        self.P_main = self.P_main[::-1]
        self.Q_main = np.cumsum(self.Q_main)[::-1] # Reversed cumulative flow
        self.Z_main = self.Z_main[::-1]
    def objective(self, variable):
        '''
        Objective function that returns difference between the (1) running and the (2) true value the of:
        the pressure at the connection point of the FIRST drip line to the main line.

        The goal is to find the pressure at the end of the main line for which the objective function is zero (i.e.,
        for which the pressure at the first connection point of the first drip line to the main line is true)

        This objective function is in depend on the pressure at the end of main line.

        :param variable: Guess of the pressure at the end of main line
        :return: True value of the pressure at the end of main line
        '''
        P_terminal_guess = variable[0]
        self.run(P_terminal_guess)
        return self.P_main[0] - self.P_main_init

    def estimate(self, P_main_init):
        '''
        A function that calculate all of the true values of the flow and pressure for all drip lines emitters and all
        of the connection points of the main line

        :param P_main_init: The value of the pressure at the first connection point of the first drip line to the main
        line
        :return: None, all of the data is saved as attributes of the object
        '''
        self.P_main_init = P_main_init

        # Finding the right terminal value using the initial pressure value as the guess
        P_main_terminal = fsolve(self.objective, self.P_main_init)[0]

        # Running with the right terminal value
        self.run(P_main_terminal)

        # DataFraming the drip lines data, flow in [l/h], pressure in [bar]
        self.q_emitters = pd.DataFrame(self.q_emitters[::-1],
                                  columns=['E{}'.format(i) for i in range(1, 166)],
                                  index=['L{}'.format(i) for i in range(1, 45)]) * 1000 * 3600  # [l/h]
        self.p_emitters = pd.DataFrame(self.p_emitters[::-1],
                                       columns=['E{}'.format(i) for i in range(1, 166)],
                                       index=['L{}'.format(i) for i in range(1, 45)]) * (9800 * 1e-5)  # [bar]

    def plot_main_profile(self):
        '''
        Plotting main line pressure, flow and altitude with respect to the distance from the first connection point
        '''
        fig, axes = plt.subplots(3,1, figsize=(15,8),sharex=True)
        axes[0].plot(self.x_main, self.P_main)
        axes[0].set_ylabel('Pressure Head [m]')
        axes[1].step(self.x_main, self.Q_main, where='post')
        axes[1].set_ylabel(r'Q [m$^3$/s]')
        axes[2].plot(self.x_main, self.Z_main)
        axes[2].set_ylabel('Z [m]')
        axes[2].set_xlabel('X [m]')
    def plot_emitters(self, which_data):
        '''
        Plotting the spatial emitters data

        :param which_data: 'p' for plotting pressure and 'q' for plotting flow
        :return:
        '''
        data = {'q': self.q_emitters, 'p': self.p_emitters}[which_data]
        color = {'q': 'Blues', 'p': 'Reds'}[which_data]
        ylabel = {'q': 'Flow [l/h]', 'p': 'Pressure [bars]'}[which_data]
        vmin = np.nanmin(data.values) * 0.9
        vmax = np.nanmax(data.values)

        plot_map(False, False)  # Plotting map
        plt.grid(b=False)
        for line, data_emitters in zip(self.lines, data.values):
            # Slicing only not-nan values
            data_emitters = data_emitters[~np.isnan(data_emitters)]

            line = line.reshape(2, 2)
            x_emitters = np.linspace(line[0, 0], line[1, 0], len(data_emitters))
            y_emitters = np.linspace(line[0, 1], line[1, 1], len(data_emitters))
            plt.scatter(x_emitters, y_emitters, c=data_emitters,
                        alpha=0.5, s=50, cmap=color, vmin=vmin, vmax=vmax)
        sc = plt.scatter([], [], c=[], alpha=1, s=50, cmap=color, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc)
        cbar.set_label(ylabel)

        # Plotting the maximum deviation
        data_min = np.nanmin(data.values)
        data_max = np.nanmax(data.values)
        maximum_deviation = (data_max - data_min) / data_max * 100
        plt.scatter([], [], c='white', label='Maximum deviation: {:.2f}%'.format(maximum_deviation))
        plt.legend()