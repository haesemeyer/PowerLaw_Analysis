# -*- coding: utf-8 -*-
"""
Created on 11/14/2016

analysis_IHB: Test analysis script for data acquired by Isaac H Bianco and saved in MATLAB v7.3 files
@author: Martin Haesemeyer
"""

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import h5py
import core

ihb_datarate = 700  # acquisition and datarate in Hz
ihb_pixelscale = 1 / 24.8  # pixelsize in mm

if __name__ == '__main__':
    fnames = core.UiGetFile(filetypes=[('Matlab file', '.mat')], diagTitle='Load data files')
    for name in fnames:
        dfile = h5py.File(name, 'r')
        assert 'martindata' in dfile.keys()
        exp_data = np.array(dfile['martindata'])
        assert exp_data.shape[0] == 8
        x_c = exp_data[1, :]
        y_c = exp_data[2, :]
        inmiddle = exp_data[3, :]
        heading = exp_data[4, :]
        looming = exp_data[5, :]
        escape = exp_data[6, :]
        hunting = exp_data[7, :]
        x_f, y_f = core.SmoothenTrack(x_c.copy(), y_c.copy(), 22)
        ispeed = core.ComputeInstantSpeed(x_f, y_f, ihb_datarate)
        frameTime = np.arange(x_c.size) / ihb_datarate
        bouts = core.DetectBouts(ispeed, 50, 500)

        # plot dish occupancy as well as small data-slice for quality control
        with sns.axes_style('whitegrid'):
            fig, ax = pl.subplots()
            ax.plot(x_c * ihb_pixelscale, y_c * ihb_pixelscale)
            ax.set_xlabel('X position [mm]')
            ax.set_ylabel('Y position [mm]')

            select = slice(700*60, 700*80)
            fig, (ax_x, ax_y, ax_s) = pl.subplots(nrows=3)
            ax_x.plot(frameTime[select], x_c[select] * ihb_pixelscale, label='Raw')
            ax_x.plot(frameTime[select], x_f[select] * ihb_pixelscale, label='Filtered')
            ax_x.set_ylabel('X position [mm]')
            ax_x.set_xlabel('Time [s]')
            ax_x.legend()
            ax_y.plot(frameTime[select], y_c[select] * ihb_pixelscale, label='Raw')
            ax_y.plot(frameTime[select], y_f[select] * ihb_pixelscale, label='Filtered')
            ax_y.set_ylabel('Y position [mm]')
            ax_y.set_xlabel('Time [s]')
            ax_y.legend()
            ax_s.plot(frameTime[select], ispeed[select] * ihb_pixelscale)
            ax_s.set_ylabel('Instant speed [mm/s]')
            ax_s.set_xlabel('Time [s]')
            fig.tight_layout()
