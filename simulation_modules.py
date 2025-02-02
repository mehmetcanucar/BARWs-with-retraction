#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRANCHING AND ANNIHILATING RANDOM WALK SIMULATION MODULES

Script to run simulations of branching and annihilating random walks based on 
    https://www.nature.com/articles/s41467-021-27135-5 

Here we extend this framework to simulate multiple neighboring branching networks
that can interact with each other. 
Uses the modules defined in the script "branching_rules.py".

@author: Mehmet Can Ucar
"""
import numpy as np
import branching_rules as br


#%%
# we first set the elementary step size to be equal to 1:
lstep = 1

class Tissue:
    # these attributes will be defined at the first instance of the class (i.e. whenever we generate an object for it)
    def __init__(self, prob, tip, angle, min_angle = np.pi/10, rad_avoid = 10*lstep, rad_termin = 1.5*lstep):
        """
        Initialize parameters

        Parameters
        ----------
        prob : float
            Branching probability.
        tip : array
            List of active tip coordinates.
        angle : array
            List of angles of active tips.
        min_angle : float, optional
            Minimal angular displacement during a branching event. The default is np.pi/10.
        rad_avoid : float, optional
            Radius of self-avoidance / attraction. The default is 5*lstep.
        rad_termin : float, optional
            Radius of annihilation/termination for the active tips. The default is 1.5*lstep.

        Returns
        -------
        None.

        """
        self.tip = tip
        self.angle = angle
        self.min_angle = min_angle
        self.rad_avoid = rad_avoid
        self.rad_termin = rad_termin
        self.prob = prob
        
    def evolve(self, coords, coords_other):
        """
        Method for branching + elongation + annihilation of tips.

        Parameters
        ----------
        coords : array
            List of ALL coordinate points, together with branch and parent labels for the entire network.
        coords_other: array
            List of all coordinate points from the neighboring network
        Returns
        -------
        None.

        """
        
        updated_info = br.branching(self.prob, self.tip, self.angle, coords, coords_other, self.min_angle, self.rad_termin)
        
        # update the coordiantes and angles of all active tips:
        self.tip = updated_info['tip']
        self.angle = updated_info['angle']
        
    # this method will incorporate the external guidance, self-avoidance and hetero-interactions between different tissues!
    def guidance(self, coord_last, coords_till, coords_other, fc, fs, fint):
        """
        Method for the effect of external potential and self-avoidance on the tips.

        Parameters
        ----------
        coord_last : array
            List of all active coordinate points at the last time step of the loop.
        coords_till : array
            List of all coordinate points until a N time steps before (to prevent immediate annihilation events).
        fc : float
            Strength of the external guidance/potential (typically between 0 and 0.3 is a good range).
        fs : float
            Strength of self-avoidance / attraction of active tips of the network to its branches 
            (same range as fc). fs<0 for self-repulsion, fs>0 for self-attraction
        fint : float
            Strength of interaction with the neighboring network
            (Same range as fs). fint<0 for repulsion, fint>0 for attraction with neighbor.
        Returns
        -------
        None.

        """
        updated_info = br.guidance_avoidance_interaction(self.tip, self.angle, coord_last, coords_till, coords_other, fc, fs, fint, self.rad_avoid)
        
        # update the coordiantes and angles of all active tips:
        self.tip = updated_info['tip']
        self.angle = updated_info['angle']
        
    def __repr__(self):
        return '[Tissue Tips: %s, Angles: %s]' % (self.tip, self.angle)
    

#%%

def simulation_loop(prob,fc,fs,fint,tmax):
    """
    Simulation loop for the BARWs with external guidance and self-interactions.

    Parameters
    ----------
    prob : array of floats (size given by the number of networks)
        List of branching probabilities of the networks.
    fc : array of floats (size given by the number of networks)
        Strength of the external guidance/potential (typically between 0 and 0.3 is a good range).
    fs : array of floats (size given by the number of networks)
        Strength of self-avoidance / attraction of active tips of the network to its branches 
        (same range as fs). fs<0 for self-repulsion, fs>0 for self-attraction
    fint = array of floats (size given by the number of networks)
        Strength of interaction with the neighboring network
        (Same range as fs). fint<0 for repulsion, fint>0 for attraction with neighbor.
    tmax : int
        Maximal simulation time.

    Returns
    -------
    d : dict
        List of all coordinates (coordinates) & angle values (angles) 
        (as well as their parent and branch labels), also a list (evolve) 
        containing info on number of active tips at each simulation step 
        (can be used to analyze network statistics at intermediate time points)

    """
    # starting coordinates of the networks:
    start_x, start_y = [50.0,150.0], [100.0,100.0]
    
    tiss_list = []
    for j in range(len(prob)):
        tiss_list.append(Tissue(prob[j], tip=np.array([[start_x[j],start_y[j],0,1]]), angle = np.array([np.pi/2])))
        
    # here we generate instances of the Tissue class (NUMBER DEFINED BEFORE):

    neuron, neuron2 = tiss_list[0], tiss_list[1]
    
    # these lists will carry the info that we need to update at every simulation step:
    coordinates = np.array([neuron.tip[0]])    
    coords_last = np.array([])
    coords_until = coordinates
    angle_list = np.array([[0,0]])
    last_tiplength_fin = np.array([len(neuron.tip)])
    
    # define similarly for the 2. tissue (I guess this is the redundant part that can be written more nicely in a function)
    coordinates2 = np.array([neuron2.tip[0]])  
    coords_last2 = np.array([])
    coords_until2 = coordinates2
    angle_list2 = np.array([[0,0]])
    last_tiplength_fin2 = np.array([len(neuron2.tip)])
    
    for t in range(tmax):
        # evaluate as long as there's an active tip in neuron 1:
        if len(neuron.tip)>0:
            neuron.evolve(coordinates,coordinates2)
            neuron.guidance(coords_last,coords_until,coordinates2,fc[0],fs[0],fint[0])
                
        # evaluate as long as there's an active tip in neuron 2:
        if len(neuron2.tip)>0:
            neuron2.evolve(coordinates2,coordinates)
            neuron2.guidance(coords_last2,coords_until2,coordinates,fc[1],fs[1],fint[1])

    
        # set angle values to be within [-pi,pi]
        angle = (neuron.angle+np.pi) % (2*np.pi) - np.pi     
        # save the angles of nodes [in degrees!] including the generation number
        angle_list = np.append(angle_list,np.column_stack((np.degrees(angle),neuron.tip[:,-1])),axis=0)

        # save the coordinates of all nodes
        coordinates = np.append(coordinates,np.array(neuron.tip),axis=0)    
        last_tiplength_fin = np.append(last_tiplength_fin,len(neuron.tip))
        coords_last = coordinates[-int(np.sum(last_tiplength_fin[-1:])):]
        if t>2:
            coords_until = coordinates[:-int(np.sum(last_tiplength_fin[-2:]))]
                
        # set angle values to be within [-pi,pi]
        angle2 = (neuron2.angle+np.pi) % (2*np.pi) - np.pi     
        # save the angles of nodes [in degrees!] including the generation number
        angle_list2 = np.append(angle_list2,np.column_stack((np.degrees(angle2),neuron2.tip[:,-1])),axis=0)

        # save the coordinates of all nodes
        coordinates2 = np.append(coordinates2,np.array(neuron2.tip),axis=0)    
        last_tiplength_fin2 = np.append(last_tiplength_fin2,len(neuron2.tip))
        coords_last2 = coordinates2[-int(np.sum(last_tiplength_fin2[-1:])):]
        if t>2:
            coords_until2 = coordinates2[:-int(np.sum(last_tiplength_fin2[-2:]))]
              
        # if there are no active tips overall, stop the simulation:
        if len(neuron.tip)==0 and len(neuron2.tip)==0:
            break
                    
    d = dict()
    d['coords'] = coordinates
    d['coords2'] = coordinates2

    d['angles'] = angle_list
    d['angles2'] = angle_list2

    d['evolve'] = last_tiplength_fin
    d['evolve2'] = last_tiplength_fin2

    
    return d
    
