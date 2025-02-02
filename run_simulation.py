#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION SAMPLES FOR TWO OVERLAPPING & INTERACTING BRANCHED NETWORKS

Uses the modules defined in the scripts "branching_rules" and "simulation_modules"

@author: Mehmet Can Ucar
"""
# Change the path to the script directory:
import os
cwd = os.path.dirname(os.path.realpath(__file__))
os.chdir(cwd)

import numpy as np
import matplotlib.pyplot as plt

import simulation_modules as sim


#%%
def run_sim(prob,fc,fs,fint,tmax,v):
    # Run simulation:
    test_run = sim.simulation_loop(prob,fc,fs,fint,tmax)
    
    # Assign coordinates, active tip lengths over time, and angle values:
    coordinates, evolve, angles = test_run['coords'], test_run['evolve'], test_run['angles']
    coordinates2, evolve2, angles2 = test_run['coords2'], test_run['evolve2'], test_run['angles2']
    
#%%    
    np.save(f'../sim_data/fs01fn03/test_simulation_pb_{prob[0]}_{prob[1]}_fc_{fc[0]}_{fc[1]}_fs_{fs[0]}_{fs[1]}_fint_{fint[0]}_{fint[1]}_coordinates0_v{v}.npy',coordinates)
    np.save(f'../sim_data/fs01fn03/test_simulation_pb_{prob[0]}_{prob[1]}_fc_{fc[0]}_{fc[1]}_fs_{fs[0]}_{fs[1]}_fint_{fint[0]}_{fint[1]}_angles0_v{v}.npy',angles)
    np.save(f'../sim_data/fs01fn03/test_simulation_pb_{prob[0]}_{prob[1]}_fc_{fc[0]}_{fc[1]}_fs_{fs[0]}_{fs[1]}_fint_{fint[0]}_{fint[1]}_evolve0_v{v}.npy',evolve)
    np.save(f'../sim_data/fs01fn03/test_simulation_pb_{prob[0]}_{prob[1]}_fc_{fc[0]}_{fc[1]}_fs_{fs[0]}_{fs[1]}_fint_{fint[0]}_{fint[1]}_coordinates1_v{v}.npy',coordinates2)
    np.save(f'../sim_data/fs01fn03/test_simulation_pb_{prob[0]}_{prob[1]}_fc_{fc[0]}_{fc[1]}_fs_{fs[0]}_{fs[1]}_fint_{fint[0]}_{fint[1]}_angles1_v{v}.npy',angles2)
    np.save(f'../sim_data/fs01fn03/test_simulation_pb_{prob[0]}_{prob[1]}_fc_{fc[0]}_{fc[1]}_fs_{fs[0]}_{fs[1]}_fint_{fint[0]}_{fint[1]}_evolve1_v{v}.npy',evolve2)

#%%
""" 
Decide on branching probability, external guidance and self-interaction strengths:
"""

prob = [0.07,0.07]  # branching probabilities
fc = [0,0]    # external guidance strength
fs = [0,0]   # self-avoidance strength
fint = [-0.4,-0.4]   # neighbor interaction strength
tmax = 60  # maximal simulation time

#%% Run single simulation to test:
test_run = sim.simulation_loop(prob,fc,fs,fint,tmax)

# Assign coordinates, active tip lengths over time, and angle values:
coordinates, evolve, angles = test_run['coords'], test_run['evolve'], test_run['angles']
coordinates2, evolve2, angles2 = test_run['coords2'], test_run['evolve2'], test_run['angles2']

#%% Run several simulations with chosen parameter set:
#for j in np.arange(v,10):
  #  run_sim(prob, fc, fs, fint, tmax,j)

#%% Plot the entire network:

fig, ax = plt.subplots(figsize=(6,8))

# this function can be used to plot the network until a certain time point

def step(till,evolve):
    step = np.sum(evolve[:till])
    return int(step)

# choose time1 to plot until a certain timepoint (time1=tmax for the complete network)
time1 = 41
time2 = time1+1

x,y = coordinates[:step(time2,evolve),0],coordinates[:step(time2,evolve),1]
x2,y2 = coordinates2[:step(time2,evolve2),0],coordinates2[:step(time2,evolve2),1]

ax.plot(x,y,'o', color='steelblue',ms=0.5)
ax.plot(x2,y2,'o', color='darkslateblue',ms=0.5)

ax.plot(x[0],y[0],'x',color='firebrick',markersize=6)
ax.plot(x2[0],y2[0],'x',color='firebrick',markersize=6)

# plot active tips with different color
ax.plot(x[step(time1,evolve):step(time2,evolve)],y[step(time1,evolve):step(time2,evolve)],'o',color='darkorange',markersize=3)
ax.plot(x2[step(time1,evolve2):step(time2,evolve2)],y2[step(time1,evolve2):step(time2,evolve2)],'o',color='firebrick',markersize=3)

# Create & plot convex hull around branched networks
from scipy.spatial import ConvexHull

xy = coordinates[:step(time2,evolve),[0,1]]
xy2 = coordinates2[:step(time2,evolve2),[0,1]]
hull = ConvexHull(xy)
hull2 = ConvexHull(xy2)
plt.plot(xy[hull.vertices,0], xy[hull.vertices,1],'--',color='darkorange',alpha=0.5, lw=1.5)
plt.plot(xy2[hull2.vertices,0], xy2[hull2.vertices,1],'--',color='firebrick',alpha=0.5, lw=1.5)

#ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, labelleft=False)

plt.gca().set_aspect('equal')

#%% Generate movie from snapshots:

    # To label the files:
time_list = np.arange(5,tmax)
str_list = np.char.zfill([str(j) for j in time_list],3)

for k in np.arange(tmax):
    fig, ax = plt.subplots(figsize=(6,8))
    
    # this function can be used to plot the network until a certain time point
    
    def step(till,evolve):
        step = np.sum(evolve[:till])
        return int(step)
    
    # choose time1 to plot until a certain timepoint (time1=tmax for the complete network)
    time1 = time_list[k]
    time2 = time_list[k]+1
    
    x,y = coordinates[:step(time2,evolve),0],coordinates[:step(time2,evolve),1]
    x2,y2 = coordinates2[:step(time2,evolve),0],coordinates2[:step(time2,evolve),1]
    
    ax.plot(x,y,'o', color='steelblue',ms=0.5)
    ax.plot(x2,y2,'o', color='darkslateblue',ms=0.5)
    
    ax.plot(x[0],y[0],'x',color='firebrick',markersize=6)
    ax.plot(x2[0],y2[0],'x',color='firebrick',markersize=6)
    
    # plot active tips with different color
    ax.plot(x[step(time1,evolve):step(time2,evolve)],y[step(time1,evolve):step(time2,evolve)],'o',color='darkorange',markersize=3)
    ax.plot(x2[step(time1,evolve):step(time2,evolve)],y2[step(time1,evolve):step(time2,evolve)],'o',color='firebrick',markersize=3)
    
    xy = coordinates[:step(time2,evolve),[0,1]]
    xy2 = coordinates2[:step(time2,evolve),[0,1]]
    hull = ConvexHull(xy)
    hull2 = ConvexHull(xy2)
    plt.plot(xy[hull.vertices,0], xy[hull.vertices,1],'--',color='darkorange',alpha=0.5, lw=1.5)
    plt.plot(xy2[hull2.vertices,0], xy2[hull2.vertices,1],'--',color='firebrick',alpha=0.5, lw=1.5)
    
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, labelleft=False)
    
    plt.gca().set_aspect('equal')
    plt.axis('off')

    fig.savefig('movie/'+str_list[k]+'.png',dpi=300)
    plt.close(fig)