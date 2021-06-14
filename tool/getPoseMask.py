from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import numpy as np
import pickle
import pdb
import glob


import scipy.io
import scipy.stats
import skimage.morphology
from skimage.morphology import square, dilation, erosion

def isvalid(p):
    return p[0] != -1 and p[1] != -1

def getPoseMask(peaks, height, width, radius=4, var=4, mode='Solid'):

    peaks = np.transpose(peaks,(1,0))

    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [9,10], \
                         [10,11], [12,13], [13,14], \
                         [9,12], [12,6], [9,3], [1,2], [1,15], [1,16], [15,17], [16,18], [17,17], [18,18]] 

    indices = []
    values = []
    for limb in limbSeq:

        p0 = peaks[limb[0] -1]
        p1 = peaks[limb[1] -1]
        # if 0!=len(p0) and 0!=len(p1):
        if isvalid(p0) and isvalid(p1): 
            r0 = p0[0]
            c0 = p0[1]
            r1 = p1[0]
            c1 = p1[1]
            ind, val = getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
            ind, val = getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
            indices.extend(ind)
            values.extend(val)
        
            distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
            sampleN = int(distance/radius)
            if sampleN>1:
                for i in range(1,sampleN):
                    r = r0 + (r1-r0)*i/sampleN
                    c = c0 + (c1-c0)*i/sampleN
                    ind, val = getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                    indices.extend(ind)
                    values.extend(val)

    shape = [height, width, 1]
    ## Fill body
    dense = np.squeeze(sparse2dense(indices, values, shape))
    dense = dilation(dense, square(5))
    dense = erosion(dense, square(5))

    dense = np.expand_dims(dense, axis=0)
    #print(dense.sum())
    return dense


def getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    Ratio_0_4 = 1.0/scipy.stats.norm(0, 4).pdf(0)
    Gaussian_0_4 = scipy.stats.norm(0, 4)
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    values = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2+j**2))
            if r+i>=0 and r+i<height and c+j>=0 and c+j<width:
                if 'Solid'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    values.append(1)
                elif 'Gaussian'==mode and distance<=radius:
                    indices.append([r+i, c+j, k])
                    if 4==var:
                        values.append( Gaussian_0_4.pdf(distance) * Ratio_0_4  )
                    else:
                        assert 'Only define Ratio_0_4  Gaussian_0_4 ...'
    return indices, values

def sparse2dense(indices, values, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = values[i]
    return dense

def get_valid_peaks(all_peaks, subsets):
    try:
        subsets = subsets.tolist()
        valid_idx = -1
        valid_score = -1
        for i, subset in enumerate(subsets):
            score = subset[-2]

            if score > valid_score:
                valid_idx = i
                valid_score = score
        if valid_idx>=0:
            peaks = []
            cand_id_list = subsets[valid_idx][:18]
            for ap in all_peaks:
                valid_p = []
                for p in ap:
                    if p[-1] in cand_id_list:
                        valid_p = p
                peaks.append(valid_p)

            return all_peaks
        else:
            return None
    except:
        # pdb.set_trace()
        return None
