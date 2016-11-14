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

# bout category dictionaries
cdict = {"exclude": -1, "regular": 0, "hunting": 1, "escape": 2}
cat_decode = {v: k for k, v in cdict.items()}  # inverse dictionary to later get our names back easily

if __name__ == '__main__':
    fnames = core.UiGetFile(filetypes=[('Matlab file', '.mat')], diagTitle='Load data files')
    for name in fnames:
        dfile = h5py.File(name, 'r')
        assert 'martindata' in dfile.keys()
        exp_data = np.array(dfile['martindata'])
        assert exp_data.shape[0] == 8
        x_c = exp_data[1, :]
        y_c = exp_data[2, :]
        inmiddle = exp_data[3, :].astype(bool)
        heading = exp_data[4, :]
        looming = exp_data[5, :].astype(bool)
        escape = exp_data[6, :].astype(bool)
        hunting = exp_data[7, :].astype(bool)
        x_f, y_f = core.SmoothenTrack(x_c.copy(), y_c.copy(), 22)
        ispeed = core.ComputeInstantSpeed(x_f, y_f, ihb_datarate)
        frameTime = np.arange(x_c.size) / ihb_datarate
        bouts = core.DetectBouts(ispeed, 50, 500, speedThresholdAbsolute=40, maxFramesAtPeak=10)

        # plot dish occupancy as well as small data-slice for quality control
        with sns.axes_style('whitegrid'):
            fig, ax = pl.subplots()
            ax.plot(x_c[inmiddle] * ihb_pixelscale, y_c[inmiddle] * ihb_pixelscale, lw=1)
            ax.plot(x_c[np.logical_not(inmiddle)] * ihb_pixelscale, y_c[np.logical_not(inmiddle)] * ihb_pixelscale,
                    'r', lw=0.5)
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
            bs = bouts[:, 0].astype(int)
            bs = bs[bs >= select.start]
            bs = bs[bs < select.stop]
            be = bouts[:, 2].astype(int)
            be = be[be >= select.start]
            be = be[be < select.stop]
            ax_s.plot(frameTime[bs], ispeed[bs] * ihb_pixelscale, 'r*')
            ax_s.plot(frameTime[be], ispeed[be] * ihb_pixelscale, 'k*')
            ax_s.set_ylabel('Instant speed [mm/s]')
            ax_s.set_xlabel('Time [s]')
            fig.tight_layout()

        # for each bout compute the distance between start and endpoint as well as the heading change
        bstarts = bouts[:, 0].astype(int)
        bends = bouts[:, 2].astype(int)
        boutRadius = np.sqrt((x_c[bends] - x_c[bstarts])**2 + (y_c[bends] - y_c[bstarts])**2)
        boutTheta = np.abs(core.AssignDeltaAnglesToBouts(bouts, heading)[0])
        # for each bout assign a category: -1 not in middle, 0 regular, 1 hunting, 2 escape
        boutCategory = np.zeros(bouts.shape[0])
        escape_frames = np.nonzero(escape)[0]
        for i, (bs, be) in enumerate(zip(bstarts, bends)):
            if not inmiddle[bs]:
                boutCategory[i] = cdict["exclude"]
            else:
                # NOTE: Escapes are identified by a single 1 but this does not align with our bout calls
                if np.min(np.abs(bs - escape_frames)) <= 25:
                    boutCategory[i] = cdict["escape"]
                elif hunting[bs:be].sum() > 0:
                    boutCategory[i] = cdict["hunting"]
                else:
                    boutCategory[i] = cdict["regular"]

        # plot bout radius vs. bout theta for our categories
        with sns.axes_style('white'):
            fig, ax = pl.subplots()
            ax.scatter(boutRadius[boutCategory == 0]*ihb_pixelscale, boutTheta[boutCategory == 0], c='b', s=10,
                       alpha=0.7, label='Regular')
            ax.scatter(boutRadius[boutCategory == 1]*ihb_pixelscale, boutTheta[boutCategory == 1], c='g', s=10,
                       alpha=0.7, label='Hunt')
            ax.scatter(boutRadius[boutCategory == 2]*ihb_pixelscale, boutTheta[boutCategory == 2], c='r', s=10,
                       alpha=0.7, label='Escape')
            ax.legend()
            ax.set_xlim(0)
            ax.set_ylim(0, 180)
            sns.despine(fig, ax)
            ax.set_xlabel('Bout displacement [mm]')
            ax.set_ylabel('Bout delta-angle [degrees]')

        # for each bout compute spline fit with overhang and then compute angular speed as well as curvature
        overhang = 300  # overhang used to ensure that the data used does not suffer from edge effects
        ang_speeds = np.array([])
        curvatures = np.array([])
        categories = np.array([])
        for b, categ in zip(bouts, boutCategory):
            # compute starts and ends of our stretch and only use if fully within dataset
            s = int(b[0] - overhang)
            if s < 0:
                continue
            e = int(b[2] + overhang)
            if e >= x_f.size:
                continue
            xb = x_f[s:e]
            yb = y_f[s:e]
            tck, u = core.spline_fit(xb * ihb_pixelscale, yb * ihb_pixelscale)
            a_spd = core.compute_angSpeed(tck, u, ihb_datarate)
            curve = core.compute_curvature(tck, u)
            # strip fit-overhang frames - but start data 20 ms earlier than actual start since turn often happens
            # before the instant speed bout call assigns the start
            start = overhang - int(20 / 1000 * ihb_datarate)
            end = -1 * (overhang - 1)
            a_spd = a_spd[start:end]
            curve = curve[start:end]
            ct = np.full_like(curve, categ)
            ang_speeds = np.r_[ang_speeds, a_spd]
            curvatures = np.r_[curvatures, curve]
            categories = np.r_[categories, ct]
