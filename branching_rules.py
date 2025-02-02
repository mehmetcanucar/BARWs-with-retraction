#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRANCHING AND ANNIHILATING RANDOM WALK (BARW) MODULES

Modules involving rules for BARWs under external guidance and self-interactions. 
Theory and technical details are outlined in the following publication:
    
    https://www.nature.com/articles/s41467-021-27135-5 

@author: Mehmet Can Ucar
"""
import numpy as np
from numpy import linalg as LA


#%%
def distance(coord,tip):
    """
    Returns Euclidean distances between a chosen tip and inputted coordinates.

    Parameters
    ----------
    coord : array
        List of N coordinate points, together with branch and parent labels.
    tip : array
        Coordinates & branch and parent labels of a chosen tip.

    Returns
    -------
    distance : array
        List of distances.
    """
    
    diff = np.add(coord[:,:2],-tip[:2])
    distance = LA.norm(diff,axis=1)
    return distance
    
#%%
# this function now incorporates the rules for branching and elongation, as well as annihilation of generated tips
# putting the annihilation here reduces the simulation time drastically!
# but the annihilation is for now only implemented after elongation steps, so it still needs to be updated for branching!

def branching(prob, node, angle, coords, coords_other, min_branch, rad_termin, lstep=1, xmin=0, xmax = 200, ymin=0, ymax=200):
    """
    This function incorporates the rules for branching and elongation, 
    as well as annihilation of generated tips.
    
    Parameters
    ----------
    prob : float
        Probability of branching.
    node : array
        Coordinates (first 2 indices) & branch and parent labels (last 2 indices) of all active tips.
    angle : array
        Angles (in radians) of all active tips.
    coords : array
        List of ALL coordinate points, together with branch and parent labels for the entire network.
    coords_other: array
        List of all coordinate points from the neighboring network
    min_branch : float
        Minimal angular displacement during a branching event.
    rad_termin : float
        Radius of annihilation/termination for the active tips.
    lstep : float, optional
        Elementary step size. The default is 1.
    xmin : float, optional
        Simulation box left boundary. The default is 0.
    xmax : float, optional
        Simulation box right boundary. The default is 200.
    ymin : float, optional
        Simulation box bottom boundary. The default is 0.
    ymax : float, optional
        Simulation box top boundary. The default is 200.

    Returns
    -------
    d : dict
        List of coordinates (node) & angle values (angle) of all active tips 
        (as well as their parent and branch labels).
    """
    
    skip = 0 
    skipp = 0
    # draw as many random numbers between 0 and 1 as there are active tips to decide on their next jumps:
    rnr = np.random.rand(len(angle))
    # copy the branching probability into a vector of size len(rnr) 
    prob_copy = [prob]*len(rnr)
    # subtract the random probabilities (in the list rnr) from the vector of branching probabilities
    # (the reason here is to see for which active tip the random probability is smaller than the branching probability)
    prob_diff = np.add(np.array(rnr),-np.array(prob_copy))

    # scan the list 'prob_diff': for negative entries, branching occurs; for positive, elongation occurs:
    # if an entry in prob_diff is <=0, that means branching wins (the random probability is smaller)
    for j in range(len(prob_diff)):

        # first check annihilation at the box boundaries:
        if not (ymax>node[j+skip-skipp][1]>ymin) or not (xmax>node[j+skip-skipp][0]>xmin):
            node = np.delete(node,j+skip-skipp,0)
            angle = np.delete(angle,j+skip-skipp,0)
            skipp += 1  
            continue 
        
        coord_temp = np.append(coords,np.array(node),axis=0)
        # highest tree/branch label of existing active tips:
        topnr = max(coord_temp[:,-1])
        
        # if subtraction is non-positive, then we branch:
        if prob_diff[j]<=0:
            # determine two random branching angles between pi/10 and pi/2 (note that we can change these min/max angles!):
            ang_branch1 = np.random.uniform(min_branch*3,np.pi/2)
            ang_branch2 = np.random.uniform(min_branch*3,np.pi/2)   
            # add a new branch changing the coordinates with the random angle ang_branch1: 
            angle = np.insert(angle,j+skip-skipp+1,angle[j+skip-skipp]+ang_branch1)
            node = np.insert(node,j+skip-skipp+1,[node[j+skip-skipp][0]+lstep*np.cos(angle[j+skip-skipp+1]), \
                             node[j+skip-skipp][1]+lstep*np.sin(angle[j+skip-skipp+1]),node[j+skip-skipp][3],topnr+2],axis=0)
            # change the angle and coordinates of the remaining branch with the random angle ang_branch2: 
            angle[j+skip-skipp] = angle[j+skip-skipp]-ang_branch2
            node[j+skip-skipp] = [node[j+skip-skipp][0]+lstep*np.cos(angle[j+skip-skipp]), \
                            node[j+skip-skipp][1]+lstep*np.sin(angle[j+skip-skipp]),node[j+skip-skipp][3],topnr+1]
            
            # annihilation of the tip during branching:
            # only take the last time points from the parent coordinates:
            parent_indx = np.where(coords[:,-1]==node[j+skip-skipp,-2])[0]
            
            # coordinates after excluding the indices belonging to parent:
            # only take the last 2 time points from the parent coordinates to prevent immature annihilation:
            excluded_coords = np.delete(coords,parent_indx[-2:],0)
            # calculate the distances between the new tip position and remaining coordinates of the network:
            if len(excluded_coords)>0:
                # here we can set if the network "sees" the neighboring network or not:
                coords_combine = np.concatenate((excluded_coords,coords_other))
                tip_distances = distance(coords_combine,node[j+skip-skipp])
                tip_distances2 = distance(coords_combine,node[j+skip-skipp+1])

                if len(np.where(tip_distances<rad_termin)[0])>0:
                    node = np.delete(node,j+skip-skipp,0)
                    angle = np.delete(angle,j+skip-skipp,0)
                    skipp += 1
                if len(np.where(tip_distances2<rad_termin)[0])>0:
                    node = np.delete(node,j+skip-skipp+1,0)
                    angle = np.delete(angle,j+skip-skipp+1,0)
                    skipp += 1
                    
            skip += 1

        # if subtraction gives a positive entry (random probability higher than branching prob), then elongate:
        else:
            # determine a random elongation angle
            rd_elong = np.random.uniform(-1,1)
            ang_elong = rd_elong*min_branch
            # change the angle and coordinates of active tips
            angle[j+skip-skipp] += ang_elong
            node[j+skip-skipp] = [node[j+skip-skipp][0]+lstep*np.cos(angle[j+skip-skipp]),\
                            node[j+skip-skipp][1]+lstep*np.sin(angle[j+skip-skipp]),node[j+skip-skipp][2],node[j+skip-skipp][3]]  
                
            
            # annihilation of the tip during elongation:
            # first exclude the indices of the same branch:
            self_indx = np.where(coords[:,-1]==node[j+skip-skipp,-1])[0]
            sibling_indx = np.setdiff1d(np.where(coords[:,-2]==node[j+skip-skipp,-2])[0],self_indx)
            # only take the last time point from the sibling and parent coordinates:
            parent_indx = np.where(coords[:,-1]==node[j+skip-skipp,-2])[0]
            # all indices to omit during annihilation:
            if len(parent_indx)>0 and len(sibling_indx)>0:
                omit_indices = np.concatenate(([parent_indx[-1],sibling_indx[-1]],self_indx))
            elif len(parent_indx)==0 and len(sibling_indx)>0:
                omit_indices = np.concatenate(([sibling_indx[-1]],self_indx))
            elif len(parent_indx)>0 and len(sibling_indx)==0:
                omit_indices = np.concatenate(([parent_indx[-1]],self_indx))
            else:
                omit_indices = self_indx
            
            # coordinates after excluding the indices belonging to self, sibling and parent:
            excluded_coords = np.delete(coords,omit_indices,0)
            # calculate the distances between the new tip position and remaining coordinates of the network:
            if len(excluded_coords)>0:
                # here we can set if the network "sees" the neighboring network or not:
                coords_combine = np.concatenate((excluded_coords,coords_other))
                tip_distances = distance(coords_combine,node[j+skip-skipp])
                    
                # Additional annihilation condition due to stochastic "retraction" events:
                retraction_rate = 10
                retraction_prob = 1-np.exp(-retraction_rate)
                if len(np.where(tip_distances<rad_termin)[0])>0:
                    print('senses')

                    if np.random.uniform(0,1)<retraction_prob:
                                                
                        print('retract')
                        
                        # get indices from the coordinates that belong to the active tip:
                        active_indx = np.where(coords[:,-1]==node[j+skip-skipp,-1])[0]
                        
                        
                        if len(active_indx)>2:
                            previous_indx = active_indx[-1]
                            
                            angle[j+skip-skipp] = angle[j+skip-skipp] + np.random.uniform(-1,1)*min_branch*4
                            node[j+skip-skipp] = [coords[previous_indx][0],coords[previous_indx][1],coords[previous_indx][2],coords[previous_indx][3]]  

                    else:
                        print('died')

                        node = np.delete(node,j+skip-skipp,0)
                        angle = np.delete(angle,j+skip-skipp,0)


                        skipp += 1
                    

    d = dict()
    d['tip'] = node
    d['angle'] = angle
    
    return d
    
#%%

# this function now incorporates both external guidance, as well as self- & hetero-avoidance (or attraction) between different tissues!
# the inputs include the strengths of external guidance fc, self-avoidance fav, and the neighbor-interaction parameter fint
# note that for fint<0 (i.e negative) it'll be a repulsive and for fint>0 it'll be an attractive interaction!

def guidance_avoidance_interaction(node, angle, coord_last, coord_until, coord_other, fc, fs, fint, radavoid):
    """
    This function now incorporates the rules for external guidance, as well as 
    for self-avoidance (or attraction) of branches.

    Parameters
    ----------
    node : array
        Coordinates (first 2 indices) & branch and parent labels (last 2 indices) of all active tips.
    angle : array
        Angles (in radians) of all active tips.
    coord_last : array
        List of ALL coordinate points, together with branch and parent labels for the entire network.
    coord_until : array
        List of ALL coordinate points, together with branch and parent labels for the entire network.
    coord_other: array
        List of all coordinate points from the neighboring network
    fc : float
        Strength of the external guidance/potential (typically between 0 and 0.3 is a good range).
    fs : float
        Strength of self-avoidance / attraction of active tips of the network to its branches 
        (same range as fc). fs<0 for self-repulsion, fs>0 for self-attraction.
    fint : float
        Strength of interaction with the neighboring network
        (Same range as fs). fint<0 for repulsion, fint>0 for attraction with neighbor.
    radavoid : float
        Radius of self-avoidance / attraction.

    Returns
    -------
    d : dict
        List of coordinates (node) & angle values (angle) of all active tips 
        (as well as their parent and branch labels).
    """
    for j in range(len(node)):
        tip = node[j]

        if fc!=0:

            # Here we can decide on the form of the external field, for instance axial vs radial:
            # 1) For the case of external guidance along an axial field (positive x-direction):
            #pol_chem = fc*np.array([0,1])
            # 2) For the case of external guidance with a radial field (with origin located at [0,Lz/2]):
            radial_diff = np.add(tip[:2],-coord_until[0,:2])
            pol_chem = fc*radial_diff/(LA.norm(radial_diff))

            tip[0] += pol_chem[0]
            tip[1] += pol_chem[1]

            for k in range(len(coord_last)):
                # filter only displaced nodes
                if pol_chem[0]!=0 or pol_chem[1]!=0:
                    # calculate distance between the displaced node and its previous instance or its parent
                    if tip[-1]==coord_last[k][-1] or tip[-2]==coord_last[k][-1]:
                        displace_more_chem = np.add(tip[:2],-coord_last[k][:2])
                        normalize_chem = LA.norm(displace_more_chem)                    
                        # update node coordinates s.t. normalized distance from previous instance is = 1                    
                        node[j][0] = coord_last[k][0]+displace_more_chem[0]/normalize_chem
                        node[j][1] = coord_last[k][1]+displace_more_chem[1]/normalize_chem
                        # update the angle of the displaced node
                        ydis = node[j][1]-coord_last[k][1]
                        xdis = node[j][0]-coord_last[k][0]

                        angle[j] = np.arctan2(ydis,xdis)


        # self-avoidance rules (apply only if there is avoidance potential):
        if fs!=0:

            # determine the distances between the active tip and inactive nodes
            dist = np.add(tip,-coord_until)
            # ignore distances between active tip and its own branch
            self_indices = np.where(tip[-1]==coord_until[:,-1])[0]
            # ignore distances above avoidance potential
            far_indices = np.where(LA.norm(dist[:,:2],axis=1)>radavoid)[0]
            omit_indices = np.unique(np.concatenate((self_indices,far_indices)))
            
            dist_consider = np.delete(dist,omit_indices,0)
            
            # sum of the distances within radavoid for the active tip
            dist_sum = np.array(np.sum(dist_consider[:,:2],axis=0))
            # normalized vector and the final displacement vector weighted by a factor 'fav'
            norm_dis = LA.norm(dist_sum)
            if norm_dis > 0:
                displace = np.array(dist_sum/norm_dis)
            else:
                displace = np.array([0,0])

            pol = -fs*displace

            tip[0] += pol[0]
            tip[1] += pol[1]
            
            # here we need to loop again over the displaced active tips to preserve the elementary branch length!
            for k in range(len(coord_last)):
                # filter only displaced nodes
                if pol[0]!=0 or pol[1]!=0:
                    # calculate distance between the displaced node and its previous instance or its parent
                    if tip[-1]==coord_last[k][-1] or tip[-2]==coord_last[k][-1]:
                        displace_more = np.add(tip[:2],-coord_last[k][:2])
                        normalize = LA.norm(displace_more)                    
                        # update node coordinates s.t. normalized distance from previous instance is = 1 
                        # ..we need to watch out here if we want the elementary branch length to be different than = 1 !!
                        node[j][0] = coord_last[k][0]+displace_more[0]/normalize
                        node[j][1] = coord_last[k][1]+displace_more[1]/normalize
                        # update the angle of the displaced node
                        ydis = node[j][1]-coord_last[k][1]
                        xdis = node[j][0]-coord_last[k][0]
                        
                        angle[j] = np.arctan2(ydis,xdis)
                        
        # if the interaction between neighboring tissue is nonzero, consider further displacements:
        if fint!=0:

            # distances between the active tip and coordinates of neighboring tissues:
            dist_neighbor = np.add(tip,-coord_other)
            nearby_neighbor = np.where(LA.norm(dist_neighbor[:,:2],axis=1)<radavoid)[0]
            dist_consider_neighbor = dist_neighbor[nearby_neighbor]

            # sum of the distances with the neighbor
            dist_sum_neighbor = np.array(np.sum(dist_consider_neighbor[:,:2],axis=0))
            # normalized vector and the final displacement vector weighted by a factor 'fav'
            norm_dis_neighbor = LA.norm(dist_sum_neighbor)
            if norm_dis_neighbor > 0:
                displace_neighbor = np.array(dist_sum_neighbor/norm_dis_neighbor)
            else:
                displace_neighbor = np.array([0,0])

            pol_neighbor = -fint*displace_neighbor

            tip[0] += pol_neighbor[0]
            tip[1] += pol_neighbor[1]

            # here we need to loop again over the displaced active tips to preserve the elementary branch length!
            for k in range(len(coord_last)):
                # filter only displaced nodes
                if pol_neighbor[0]!=0 or pol_neighbor[1]!=0:
                    # calculate distance between the displaced node and its previous instance or its parent
                    if tip[-1]==coord_last[k][-1] or tip[-2]==coord_last[k][-1]:
                        displace_more = np.add(tip[:2],-coord_last[k][:2])
                        normalize = LA.norm(displace_more)                    
                        # update node coordinates s.t. normalized distance from previous instance is = 1 
                        # ..we need to watch out here if we want the elementary branch length to be different than = 1 !!
                        node[j][0] = coord_last[k][0]+displace_more[0]/normalize
                        node[j][1] = coord_last[k][1]+displace_more[1]/normalize
                        # update the angle of the displaced node
                        ydis = node[j][1]-coord_last[k][1]
                        xdis = node[j][0]-coord_last[k][0]
                        
                        angle[j] = np.arctan2(ydis,xdis)
                        
    d = dict()
    d['tip'] = node
    d['angle'] = angle
    
    return d
    
 #%%
 
 