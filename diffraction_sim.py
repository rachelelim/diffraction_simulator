#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:02:06 2018

@author: rachellim

Simulates diffraction for an arbitrarily defined Gaussian of energies and 
an arbitrarily defined amount of uniform misorientation

Takes config, redesigned energy loops so it's faster.

Also actually correctly deals with multiples grains and misorientation.

Correct bandpass, takes any of 3 inputs for the grains

Saves as npz
"""

import numpy as np
import random
import yaml
import cPickle as cpl
from matplotlib import pyplot as plt
import time
from numba import jit
import h5py
import os
import argparse

from hexrd import instrument
from hexrd.xrd import material
from hexrd import matrixutil as mutil
from hexrd.fitting import peakfunctions as pk
from hexrd.xrd import rotations as rot
from hexrd import config
from hexrd import imageseries

os.chdir('/Users/rachellim/diffraction_simulator')

import diff_sim_util as diffutil #loads all the simulator functions


# Having some weird behavior in spyder so this is a temp fix for that...
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


#==============================================================================
#%% Input config file
#==============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="batchfitgrains")

    parser.add_argument('cfg_file',
                        help="cfg file", type=str)

    args = parser.parse_args()
    cfg_filename = args.cfg_file


else:
    cfg_filename = '/Users/rachellim/diffraction_simulator/diff_sim_config.yml'

cfg = yaml.load(open(cfg_filename, 'r'))
working_dir = cfg['working_dir']

#==============================================================================
#%% Initialization
#==============================================================================

os.chdir(working_dir)

analysis_name = cfg['analysis_name']

#set experiment parameters
det_file = cfg['experiment']['detector_file']
ome_start = cfg['experiment']['omega']['start']
ome_stop = cfg['experiment']['omega']['stop']
ome_step = cfg['experiment']['omega']['step']
ome_ranges = [(np.radians(ome_start),np.radians(ome_stop)),]

#set grains to diffract
gparm_list,calc_mis = diffutil.init_grain_list(cfg)

#set material parameters
if cfg['material']['materials_file']['filename'] is not None:
    matl = diffutil.load_pdata(cfg['material']['materials_file'],cfg['material']['active_material'])
    matl.beamEnergy = cfg['energy']['nominal']
else:
    mat_name = cfg['material']['material_parameters']['mat_name']
    space_group = cfg['material']['material_parameters']['space_group']
    lparms = cfg['material']['material_parameters']['lparms']
    nominal_energy = cfg['energy']['nominal']
    matl = diffutil.make_matl(mat_name, space_group, lparms, nominal_energy)


# load instrument
instr = diffutil.load_instrument(det_file)
det_params = instr.detector_parameters

#==============================================================================
#%% Energy Initialization
#==============================================================================
#currently making own Gaussian

mono = cfg['energy']['mono']

if mono == 'HEM':
    FWHM_at_50 = 7.7e-4
    FWHM_actual = FWHM_at_50*matl.beamEnergy/50
elif mono == 'HRM': #making some generalizations, talk to REL if trying to use this
    FWHM_actual = 1.4e-4
else:
    print('not a valid mono')

num_bins = cfg['energy']['num_bins'] #odd number is best here, number of bins for energy

energy = np.linspace(matl.beamEnergy-FWHM_actual,matl.beamEnergy+FWHM_actual,num=num_bins)

f = np.repeat(2,len(energy))
#a = plt.figure()
#plt.plot(energy,f,'o-',color='crimson')
##plt.bar(energy,f,width = (np.max(energy)-np.min(energy))/num_bins)
#plt.xlabel('Energy (keV)',fontsize=16)
#plt.ylabel('Intensity (a.u.)', fontsize=16)
#plt.legend(['Function','Binned'])


#==============================================================================
#%% Add Misorientation
#==============================================================================

if calc_mis is True:
    gparm_list = diffutil.calculate_mis(cfg,gparm_list)


#==============================================================================
#%% Simulate diffraction
#==============================================================================

start = time.time()

det_int = {}
diff_coord = {}
diff_angs = {}
for det_id in instr.detectors:
    det_X = det_params[det_id]['detector']['pixels']['columns']
    det_Y = det_params[det_id]['detector']['pixels']['rows']
    diff_coord[det_id] = np.empty([0,2])
    diff_angs[det_id] = np.empty([0])
    det_int[det_id] = np.zeros([int((ome_stop-ome_start)/ome_step)+1,det_Y,det_X],dtype = np.uint16)
    

##simulates for energy bandwidth##
for energy_index in range(0,len(energy)):
    start_energy = time.time()
    matl.beamEnergy = energy[energy_index]
    print('Energy = %.5f keV (%d of %d)' %(matl.beamEnergy, energy_index, num_bins))
    plane_data = matl.planeData
    
    weight = f[energy_index].astype(np.uint16)
    
    rot_start = time.time()
    # for each detector key, returns valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size
    rotation_series = instr.simulate_rotation_series(plane_data,gparm_list,ome_ranges=ome_ranges)
    rot_end = time.time()
    print('\t Rotation Series took %.3f' %(rot_end-rot_start))

    for det_id in instr.detectors:
        
        det_X = det_params[det_id]['detector']['pixels']['columns']
        det_Y = det_params[det_id]['detector']['pixels']['rows']      
        
        XY_pix,ome_angs = diffutil.diffraction_XY(rotation_series,det_id,det_params,ome_ranges,ome_step)

        
        for grain in range(0,len(XY_pix)):
            det_int[det_id][(ome_angs[grain]).astype(np.uint16),(XY_pix[grain][1]).astype(np.uint16),(XY_pix[grain][0]).astype(np.uint16)] += weight
#            diff_coord[det_id] = np.vstack([diff_coord[det_id],XY_pix[grain].T])
#            diff_angs[det_id] = np.hstack([diff_angs[det_id],ome_angs[grain]]) 
   
    end_energy = time.time()
    print ('\t took %.3f seconds' %(end_energy - start_energy))

end = time.time()
print ('All energies took ' + str(end - start) + ' seconds')


#==============================================================================
#%% Save as FrameCache
#==============================================================================

start_write = time.time()
steps = diffutil.make_ome_wedge(ome_start,ome_stop,int((ome_stop-ome_start)/ome_step))

diffutil.save_fc(instr, steps, det_int,analysis_name)
print('Saved to %sfc-%s_det_id.npz' %(working_dir,analysis_name))
end_write = time.time()
print('Saving frame-cache took %.3f seconds' %(end_write-start_write))

#%%
#
#x_y_ome = {}
#for det_id in instr.detectors:
#    x_y_ome[det_id] = np.hstack([diff_coord[det_id],np.expand_dims(diff_angs[det_id],axis=1)])
#
##%% plotting
#for det_id in instr.detectors:
#    pix_size = det_params[det_id]['detector']['pixels']['size']
#    
#    
#    plot_width, plot_height = diffutil.plot_shape(det_params,det_id)
#    
#    XY = diff_coord[det_id]*pix_size #change back to micron coords
#    
#    
#    fig = plt.figure(figsize = (plot_width, plot_height))
#    plt.plot(XY[:,0],XY[:,1],'.',markersize=1)
#    plt.title(det_id,fontsize=20)   
#    plt.show()
#
#
# #%%
#    
#    
#    
#detector_image = plt.figure(figsize = (plot_width*2, plot_height*2))
##    detector_image = plt.figure()
#ax = plt.axes()
#plt.imshow(det_int[det_id],cmap='inferno')
#plt.colorbar()
##plt.ylim(bottom = 1900,top= 2000)
##plt.xlim(left = 850, right = 950)
#plt.title(det_id,fontsize=20)
