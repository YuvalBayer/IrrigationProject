import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style()

ET0 = np.array([1.4, 2.0, 3.2, 4.2, 5.8, 6.8, 6.7, 6.2, 5.2, 3.5, 2.1, 1.3])
Kc = np.array([0.4, 0.5, 0.55, 0.55, 0.6, 0.65, 0.65, 0.65, 0.6, 0.55, 0.55, 0.5])
ETc = Kc * ET0

plant_demand = pd.DataFrame(np.c_[ET0, Kc, ETc], index=range(1,13),
                            columns=[r'$ET_0 $ [mm/day]',r'$K_c$',r'$ET_c $ [mm/day]'])
plant_demand.index.name = 'Month'


COLORS = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 RGB instances for 12 months colors

def plot_monthly_series(axis, s):
    label = s.name
    color_indices = s.index - 1  # Indices to choose from colors - by month number
    axis.bar(s.index,
            s.values,
            color=COLORS[color_indices],
            edgecolor='black',
            lw=3,
             alpha=0.75)
    axis.grid(axis='x')
    axis.set_ylabel(label)
    axis.set_ylim(0,max(s.values)*1.2)
    # Plotting values
    for index, value in zip(s.index, s.values):
        axis.text(index, value*1.05, str(round(value, 2)), ha='center')


sns.set_context("notebook", font_scale=1.5)
fig = plt.figure(figsize=(10,8))
axes = [plt.subplot(3,1,i) for i in range(1,4)]

for i, col_name in zip(range(len(axes)), plant_demand):
    plot_monthly_series(axes[i], plant_demand[col_name])
    if i != len(axes)-1:
        axes[i].set_xticklabels([])
    else:
        axes[i].set_xlabel('Month')

data = pd.read_csv('Dafna_monthly_rain_11-21.csv')