from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, sqrt
import itertools
import os
import re
from mpl_toolkits import mplot3d
import random
import csv
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.optimize


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

import os
files = [f for f in os.listdir('.') if f.endswith("z_values_histogram.csv")]
# ~ print (files)

counts=[]
distance=[]


with open(files[0]) as csv_file:
	for row in csv.reader(csv_file, delimiter=','):
		if row[0]!='0' and row[1]!='0':
				counts.append(float(row[0]))
				distance.append(float(row[1]))
				

counts=np.array(counts)
distance=np.array(distance)
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

p0=(93,0.1,-3.52)

params, cv = scipy.optimize.curve_fit(monoExp, distance, counts, p0)
m, t, b = params


# ~ squaredDiffs = np.square(counts - monoExp(distance, m, t, b))
# ~ squaredDiffsFromMean = np.square(counts - np.mean(counts))
# ~ rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
# ~ print(f"R² = {rSquared}")

# ~ print(f"Y = {m} * e^(-{t} * x) + {b}")

# ~ print (counts)
# ~ print (distance)


function_max=monoExp(0,m,t,b)
print(function_max)

norm=[]
for i in distance:
	norm.append(monoExp(i, m, t, b)/function_max)

p0=(0,20,0)
param_bounds=([-0.9999,-np.inf,-0.00001],[1.000001,np.inf,0.000001])
norm=np.array(norm)
params, cv = scipy.optimize.curve_fit(monoExp, distance, norm, p0, bounds=param_bounds)
n, o, c = params

print('fit only normalized', f"Y = {n} * e^(-{o} * x) + {c}")



p0_mass=(0,20,0)
param_bounds=([-0.9999,-np.inf,-0.00001],[1.000001,np.inf,0.000001])
Y=norm
xe_density=0.005894
X=distance*xe_density
X=np.array(X)

params, cv = scipy.optimize.curve_fit(monoExp, X, Y, p0_mass, bounds=param_bounds)
a, d, r = params

print('fit mass attenuation', f"Y = {a} * e^(-{d} * x) + {r}")





fig = plt.figure(figsize=(10,10))

#PLOTS
plt.plot(distance, counts,"s", color='tab:blue',markersize=1,linestyle='none',label='Counts in Xenon [22.6]')
# ~ plt.plot(distance, norm,".", color='tab:orange',markersize=1,linestyle='none',label='Normalized function')
plt.plot(distance, monoExp(distance, m, t, b),"*", color='tab:blue',markersize=0,linestyle='dashed',label='Fitted Function:\n $y = %0.2f e^{-%0.2f t} + %0.2f$' % (m, t, b))
# ~ plt.plot(distance, monoExp(distance, n, o, c),"*", color='tab:green',markersize=0,linestyle='dashed',label='Fitted Function:\n $y = %0.2f e^{-%0.2f t} + %0.2f$' % (n, o, c))


# ~ plt.xlim(0.1,0.8)
# ~ plt.ylim(0.1,0.8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid(True)
plt.legend(loc='upper left',fontsize='x-large')
plt.xlabel('Distance between the  window center to the photosensor (cm)', fontsize='x-large')
plt.ylabel('Counts', fontsize='x-large')

fig = plt.figure(figsize=(10,10))

#PLOTS
# ~ plt.plot(distance, counts,"s", color='tab:blue',markersize=1,linestyle='none',label='Counts in Xenon [22.6]')
plt.plot(distance, norm,".", color='tab:orange',markersize=1,linestyle='none',label='Normalized function 22.6 kev in Xenon')
# ~ plt.plot(distance, monoExp(distance, m, t, b),"*", color='tab:blue',markersize=0,linestyle='dashed',label='Fitted Function:\n $y = %0.2f e^{-%0.2f t} + %0.2f$' % (m, t, b))
plt.plot(distance, monoExp(distance, n, o, c),"*", color='tab:green',markersize=0,linestyle='dashed',label='Fitted Function:\n $y = %0.2f e^{-%0.2f t} + %0.2f$' % (n, o, c))


# ~ plt.xlim(0.1,0.8)
# ~ plt.ylim(0.1,0.8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.grid(True)
plt.legend(loc='upper left',fontsize='x-large')
plt.xlabel('Distance between the  window center to the photosensor (cm)', fontsize='x-large')
plt.ylabel('Counts', fontsize='x-large')


fig, ax2 = plt.subplots()

plt.yticks(fontsize=14)
plt.xticks(fontsize=14)




plt.grid(True)

# ~ ax2.set_ylim(250,800)
ax2.set_xlabel('mass thickness ($g.cm^{-2}$)', fontsize='x-large')
ax2.set_ylabel('$I/I_0$', fontsize='x-large')

ax2.plot(X, Y,"s", color='tab:orange',markersize=1,linestyle='solid',label='Points for 22.6 kev x-rays in Xenon (normalized)')
ax2.plot(X, monoExp(X,  a, d, r),"*", color='tab:blue',markersize=0,linestyle='dashed',label='Fitted Function:\n $y = %0.2f *e^{-%0.2f } + %0.2f $' % (a, d, r))

# ~ ax2.plot(diameter[:5], yeald_single_no_ref[:5],"*", color='tab:blue',markersize=11,linestyle='dashed',label='Yeald (single gaussian) [not considering reflections]')
# ~ ax2.plot(diameter[:5], nVUV[:5],"h", color='tab:green',markersize=11,linestyle='dashed',label='Simulation')

# ~ y_phy_var=[250,800]
# ~ ax2.plot(laapd_diameter_1_times, y_phy_var ,marker='',color=colors['black'],linestyle='dashed',markersize=11,label='')
# ~ ax2.plot(laapd_diameter_2_times, y_phy_var ,marker='',color=colors['black'],linestyle='dashed',markersize=11,label='')
# ~ ax2.plot(laapd_area_2_times, y_phy_var ,marker='',color=colors['black'],linestyle='dashed',markersize=11,label='')
# ~ ax2.plot(laapd_area_6_times, y_phy_var ,marker='',color=colors['black'],linestyle='dashed',markersize=11,label='')




ax2.annotate('Mass attenuation coefficient [NIST]\n 22.1 keV -- $\sim$  17.58 $cm^{2}.g^{-1}$  ', fontsize='x-large',xy=(0.045, 0.92), xytext=(0.035, 0.6),arrowprops=dict(facecolor='black', shrink=0.05))
# ~ ax2.annotate('Photosensor \n 2 '+r'$\times$ area', fontsize='x-large', xy=(2.312, 500), xytext=(1.7, 550),arrowprops=dict(facecolor='black', shrink=0.05))
# ~ ax2.annotate('Photosensor \n 2 '+r'$\times$ diameter', fontsize='x-large', xy=(3.2, 500), xytext=(2.6, 550),arrowprops=dict(facecolor='black', shrink=0.05))
# ~ ax2.annotate('Photosensor \n 6 '+r'$\times$ area', fontsize='x-large', xy=(4, 500), xytext=(3.4, 550),arrowprops=dict(facecolor='black', shrink=0.05))


legend = ax2.legend(loc='upper right', fontsize='x-large')
fig.tight_layout()  # otherwise the right y-label is slightly clipped


plt.show()


