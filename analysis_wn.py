# -*- coding: utf-8 -*-
"""
Created on 11/16/2016

analysis_wn: Test analysis script for data acquired by Martin Haesemeyer for gaussian white noise experiments
             aggregated in one Hdf5 file
@author: Martin Haesemeyer
"""

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import core
import pandas
import os
from scipy.stats import linregress

mh_datarate = 250  # acquisition and datarate in Hz
mh_pixelscale = 1 / 9  # pixelsize in mm
# start data 30 ms earlier than actual start since turn often happens
# before the instant speed bout call assigns the start
preStartms = 30


if __name__ == '__main__':
    sv = ''
    while sv != 'y' and sv != 'n':
        sv = input('Save figures? [y/n]')
    sv = sv == 'y'
    fname = core.UiGetFile(filetypes=[('Gauss white noise data', 'wn_gauss_data.mat')], diagTitle='Load data files')
    dfile = h5py.File(fname, 'r')

    for k in dfile.keys:
        exp_data = np.array(dfile[k])
        basename = k[:k.find('.mat')]
        x_c = exp_data[:, 0]
        y_c = exp_data[:, 1]
        inmiddle = exp_data[:, 4].astype(bool)
        heading = exp_data[:, 2]
        phase = exp_data[:, 3]
        x_f, y_f = core.SmoothenTrack(x_c.copy(), y_c.copy(), 11)
        ispeed = core.ComputeInstantSpeed(x_f, y_f, mh_datarate)
        frameTime = np.arange(x_c.size) / mh_datarate
        bouts = core.DetectBouts(ispeed, 20, mh_datarate)
