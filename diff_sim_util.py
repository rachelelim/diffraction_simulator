#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:07:02 2018

@author: rachellim
"""
import numpy as np
import random
import yaml
import cPickle as cpl
from matplotlib import pyplot as plt
import time
from numba import jit
import h5py

from hexrd import instrument
from hexrd.xrd import material
from hexrd import matrixutil as mutil
from hexrd.fitting import peakfunctions as pk
from hexrd.xrd.symmetry import applySym, quatOfLaueGroup
from hexrd.xrd import rotations as rot
from hexrd import config
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries


def init_grain_list(cfg):
    calc_mis = False

    #set grain parameters to simulate
    if cfg['grains_to_diffract']['microstructure']['file'] is not None:
        micro = h5py.File(cfg['grains_to_diffract']['microstructure']['file'],'r')
        ax_ang = micro['DataContainers']['SyntheticVolumeDataContainer']['CellData']['AxisAngle'][:,:,:]
        exp_map_array = ax_ang[:,:,:,0:3]*np.expand_dims(ax_ang[:,:,:,3],axis=3)
        a = exp_map_array.shape[0]
        b = exp_map_array.shape[1]
        c = exp_map_array.shape[2]
    
        orientation = np.reshape(exp_map_array,(a*b*c,3))
    
        X0s = np.linspace(0,a-1,a)-(a-1)/2.
        Y0s = np.linspace(0,b-1,b)-(b-1)/2.
        Z0s = np.linspace(0,c-1,c)-(c-1)/2.
        Xs, Ys, Zs = np.meshgrid(X0s,Y0s,Z0s)
    
        voxel_size = cfg['grains_to_diffract']['microstructure']['voxel_size']
        centroid = np.stack([Xs.flatten(),Ys.flatten(),Zs.flatten()]).T * voxel_size /1000
        gparm_list = make_gparam_list(orientation, centroid)
        micro.close()
    
    elif cfg['grains_to_diffract']['grains_file'] is not None:
        grains_out = cfg['grains_to_diffract']['grains_file']
        calc_mis = True
    
    else:
        centroid = np.array(cfg['grains_to_diffract']['grain_parameters']['centroid'])
        stretch = np.array(cfg['grains_to_diffract']['grain_parameters']['stretch'])
        orientation = np.array(cfg['grains_to_diffract']['grain_parameters']['orientation'])
    
        if orientation == 'random':
            orientation = mutil.unitVector(np.array([random.random(),random.random(),random.random()]))[0]
        gparm_list = make_gparam_list(orientation, centroid)
        calc_mis = True

    return gparm_list,calc_mis
    

def make_matl(mat_name, sgnum, lparms, energy, hkl_ssq_max=100):
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max
    matl.beamEnergy = energy

    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl


def load_pdata(cpkl, key):
    with file(cpkl, "r") as matf:
        mat_list = cpl.load(matf)
    return dict(zip([i.name for i in mat_list], mat_list))[key].planeData


def load_instrument(yml):
    with file(yml, 'r') as f:
        icfg = yaml.load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


def make_gparam_list(expmap, centroid, stretch=None): #!!! make take strain
    if len(expmap.shape) == 1:
        if stretch is None:
            stretch = np.array([0.999,0.999,0.999])
        gparm_list = np.expand_dims(np.hstack([expmap, centroid, stretch]),axis=0)
    else:
        if stretch is None:
            stretch = np.repeat(np.expand_dims([0.999,0.999,0.999],axis=1),len(expmap),axis=1).T
        gparm_list = np.hstack([expmap, centroid, stretch])
    return gparm_list


def load_gparam_list(filename):
    data = np.loadtxt(filename)
    g_data = data[:,3:15]
    return g_data


def calculate_mis(cfg,gparm_list): #borrowed from the nfutil misorientation calculation,
    # currently hardcoded for one orientation only REL 2018-09-24
    expmap = gparm_list[:,0:3]
    centroid = gparm_list[:,3:6]
    stretch = gparm_list[:,6:]
    #misorientation
    misorientation_bnd = cfg['grains_to_diffract']['mis_bound']#degrees
    misorientation_spacing = cfg['grains_to_diffract']['mis_space'] #degrees


    mis_amt=misorientation_bnd*np.pi/180.
    spacing=misorientation_spacing*np.pi/180.

    ori_pts = np.arange(-mis_amt, (mis_amt+(spacing*0.999)),spacing)
    num_ori_grid_pts=ori_pts.shape[0]**3

    num_oris=1#expmap.shape[0] !!!fix this later

    XsO, YsO, ZsO = np.meshgrid(ori_pts, ori_pts, ori_pts)

    grid0 = np.vstack([XsO.flatten(), YsO.flatten(), ZsO.flatten()]).T


    exp_maps_expanded=np.zeros([num_ori_grid_pts*num_oris,3])

    for ii in np.arange(num_oris):
        pts_to_use=np.arange(num_ori_grid_pts)+ii*num_ori_grid_pts  
        exp_maps_expanded[pts_to_use,:]=grid0+np.r_[expmap ]

        exp_maps=exp_maps_expanded
        n_grains=exp_maps.shape[0]
        rMat_c = rot.rotMatOfExpMap(exp_maps.T)
        phi, n = rot.angleAxisOfRotMat(rMat_c)

        exp_map = (phi*n).T
        centroids = np.repeat(centroid,exp_map.shape[0],axis=0)
        stretchs = np.repeat(stretch,exp_map.shape[0],axis=0)

        gparm_list = np.hstack([exp_map, centroids, stretchs])
    return gparm_list


def make_ome_wedge(omin,omax,nsteps):
    a = np.linspace(omin,omax,nsteps+1)
    om = np.zeros((nsteps,2))
    om[:,0] = a[:-1]
    om[:,1] = a[1:]
    return om



def diffraction_XY(rotation_series,det_id,det_params,ome_range,ome_step):
    valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size = rotation_series[det_id]
    
    det_X = det_params[det_id]['detector']['pixels']['columns']
    det_Y = det_params[det_id]['detector']['pixels']['rows']
    pix_size = det_params[det_id]['detector']['pixels']['size']
    
    #bin data --> bins[i-1] < x <= bins[i]
    X_bins = np.arange(-np.float(det_X)/2+0.5,np.float(det_X)/2+0.5,1)
    Y_bins = np.arange(-np.float(det_Y)/2+0.5,np.float(det_Y)/2+0.5,1)
    ome_bins = np.arange(ome_range[0][0],ome_range[0][1],np.radians(ome_step))
    
    
    XY_pix = []
    XY_bin_pix = []
    ome_bin_ang = []
    for grain in range(0,len(valid_xys)):
        XY_pix = valid_xys[grain]/pix_size #change XY from microns to pixels
    

        XY_bin_pix.append(np.vstack([np.digitize(XY_pix[:,0],X_bins,right=True),
                    np.digitize(XY_pix[:,1],Y_bins,right=True)]))
        ome_bin_ang.append(np.digitize(valid_angs[grain][:,2],ome_bins,right=True))
    
    
    return XY_bin_pix,ome_bin_ang #tth, eta, ome



def plot_shape(det_params,det_id):
    pix_size = det_params[det_id]['detector']['pixels']['size']
    det_X = det_params[det_id]['detector']['pixels']['columns'] * pix_size[0]
    det_Y = det_params[det_id]['detector']['pixels']['rows'] * pix_size[1]
    plot_height = 6.

    plot_width = (np.float(det_X)/np.float(det_Y))*plot_height
    return plot_width, plot_height


def make_ims(diff_coord_array,om):
    m = dict(omega=om)
    ims = imageseries.open(None, 'array', data = diff_coord_array, meta = m)
    oms = OmegaImageSeries(ims)
    return oms


def save_fc(instr, steps, det_int,analysis_name):
    ims = {}
    for det_id in instr.detectors:
        ims = make_ims(det_int[det_id][:-1,:,:],steps)
        imageseries.write(ims,'dummy','frame-cache',cache_file = 'fc-%s_%s.npz' %(analysis_name,det_id),threshold=0)
    pass
