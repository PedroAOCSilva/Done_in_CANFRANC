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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

#Function to generate a random 3D vector.
def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return (x,y,z)

#Initialize the variables

import os
files = [f for f in os.listdir('.') if f.endswith("22_6_degrad_clean_primary_electron_positions_full.csv")]
print (files)

X=[]
Y=[]
Z=[]
T=[]
count_row=0

with open(files[0]) as csv_file:
	for row in csv.reader(csv_file, delimiter=','):
			count_row=count_row+1
			
			# ~ print(int(len(row)))
			counter=0
			# ~ print(len(row))
			while counter < int(len(row)):
				X.append(float(row[counter])/10000)
				Y.append(float(row[counter+1])/10000)
				Z.append(float(row[counter+2])/10000)
				T.append(float(row[counter+3])/1000)
				counter=counter+4
				

# ~ print (count_row)
# ~ print (len (T))

# ~ nbins=5000				
# ~ fig = plt.figure()
# ~ #An "interface" to matplotlib.axes.Axes.hist() method
# ~ n, bins, patches = plt.hist(x=Z, bins=nbins, color='#0504aa',
                            # ~ alpha=0.7, rwidth=0.85)
# ~ plt.grid(True)                           
# ~ plt.xlabel('Distance in the Z direction [$\mu$m]' ,fontsize='x-large')
# ~ plt.ylabel('Number of primaries',fontsize='x-large')
# ~ plt.gca().invert_xaxis()
# ~ plt.yticks(fontsize=14)
# ~ plt.xticks(fontsize=14)

# ~ fig = plt.figure()
# ~ #An "interface" to matplotlib.axes.Axes.hist() method
# ~ n, bins, patches = plt.hist(x=X, bins=nbins, color='#0504aa',
                            # ~ alpha=0.7, rwidth=0.85)

# ~ plt.xlabel('Distance in the X direction [$\mu$m]' ,fontsize='x-large')
# ~ plt.ylabel('Number of primaries',fontsize='x-large')
# ~ plt.grid(True)
# ~ plt.yticks(fontsize=14)
# ~ plt.xticks(fontsize=14)


# ~ fig = plt.figure()
# ~ #An "interface" to matplotlib.axes.Axes.hist() method
# ~ n, bins, patches = plt.hist(x=Y, bins=nbins, color='#0504aa',
                            # ~ alpha=0.7, rwidth=0.85)

# ~ plt.xlabel('Distance in the Y direction [$\mu$m]' ,fontsize='x-large')
# ~ plt.ylabel('Number of primaries',fontsize='x-large')
# ~ plt.grid(True)
# ~ plt.yticks(fontsize=14)
# ~ plt.xticks(fontsize=14)

# ~ fig = plt.figure()
# ~ #An "interface" to matplotlib.axes.Axes.hist() method
# ~ n, bins, patches = plt.hist(x=T, bins=nbins, color='#0504aa',
                            # ~ alpha=0.7, rwidth=0.85)

# ~ plt.xlabel('Time for full x-ray absorption [ps]' ,fontsize='x-large')
# ~ plt.ylabel('Number of primaries',fontsize='x-large')

# ~ plt.yticks(fontsize=14)
# ~ plt.xticks(fontsize=14)
# ~ plt.grid(True)
                            
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)

# plot data
ax.scatter(X, Y, Z,c='C0',s=0.5,label='Primary electron')

#modify axes
# ~ ax.axes.set_xlim3d(left=-4, right=4) 
# ~ ax.axes.set_ylim3d(bottom=-4, top=4) 
# ~ ax.axes.set_zlim3d(bottom=-4, top=4) 
# ~ ##ax.minorticks_on()
ax.tick_params(axis='both',which='minor',length=5,width=2,labelsize=14)
ax.tick_params(axis='both',which='major',length=5,width=2,labelsize=14)
ax.set_xlabel('X axis [$cm$]',fontsize=14)
ax.set_ylabel('Y axis [$cm$]',fontsize=14)
ax.set_zlabel('Z axis [$cm$]',fontsize=14)
# ~ ##ax.set_xlim(-40,40)
# ~ ##ax.set_ylim(-40,40)
ax.xaxis.labelpad=12
ax.yaxis.labelpad=15
ax.zaxis.labelpad=12
plt.legend(loc='center left',fontsize=14)
#display 
plt.show()
