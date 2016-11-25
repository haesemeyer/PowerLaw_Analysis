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

# bout category dictionaries
cdict = {"exclude": -1, "slow straight": 0, "fast straight": 1, "slow turn": 2, "fast turn": 3}
cat_decode = {v: k for k, v in cdict.items()}  # inverse dictionary to later get our names back easily


if __name__ == '__main__':
    sv = ''
    while sv != 'y' and sv != 'n':
        sv = input('Save figures? [y/n]')
    sv = sv == 'y'
    fname = core.UiGetFile(diagTitle='Load data files')
    dfile = h5py.File(fname[0], 'r')

    all_fits = []  # list of log/log fits for each experiment
    bout_curves = []  # list of 2-element tuples with curvature for each bout and it's category
    bout_aspeeds = []  # list of 2-element tuples with angular speeds for each bouts and it's category

    for eid, k in enumerate(dfile.keys()):
        exp_data = np.array(dfile[k])
        basename = k[:k.find('.wns')]
        x_c = exp_data[:, 0]
        y_c = exp_data[:, 1]
        inmiddle = exp_data[:, 4].astype(bool)
        heading = exp_data[:, 2]
        phase = exp_data[:, 3]
        x_f, y_f = core.SmoothenTrack(x_c.copy(), y_c.copy(), 11)
        ispeed = core.ComputeInstantSpeed(x_f, y_f, mh_datarate)
        frameTime = np.arange(x_c.size) / mh_datarate
        bouts = core.DetectBouts(ispeed, 20, mh_datarate)

        # plot dish occupancy as well as small data-slice for quality control
        if eid % 5 == 0:
            with sns.axes_style('whitegrid'):
                fig, ax = pl.subplots()
                ax.plot(x_c[inmiddle] * mh_pixelscale, y_c[inmiddle] * mh_pixelscale, lw=1)
                ax.plot(x_c[np.logical_not(inmiddle)] * mh_pixelscale, y_c[np.logical_not(inmiddle)] * mh_pixelscale,
                        'r', lw=0.5)
                ax.set_xlabel('X position [mm]')
                ax.set_ylabel('Y position [mm]')
                ax.set_title(basename)

                select = slice(mh_datarate * 60 * 10, int(mh_datarate * 60 * 10.5))
                fig, (ax_x, ax_y, ax_s) = pl.subplots(nrows=3)
                ax_x.plot(frameTime[select], x_c[select] * mh_pixelscale, label='Raw')
                ax_x.plot(frameTime[select], x_f[select] * mh_pixelscale, label='Filtered')
                ax_x.set_ylabel('X position [mm]')
                ax_x.set_xlabel('Time [s]')
                ax_x.legend()
                ax_x.set_title(basename)
                ax_y.plot(frameTime[select], y_c[select] * mh_pixelscale, label='Raw')
                ax_y.plot(frameTime[select], y_f[select] * mh_pixelscale, label='Filtered')
                ax_y.set_ylabel('Y position [mm]')
                ax_y.set_xlabel('Time [s]')
                ax_y.legend()
                ax_s.plot(frameTime[select], ispeed[select] * mh_pixelscale)
                bs = bouts[:, 0].astype(int)
                bs = bs[bs >= select.start]
                bs = bs[bs < select.stop]
                be = bouts[:, 2].astype(int)
                be = be[be >= select.start]
                be = be[be < select.stop]
                ax_s.plot(frameTime[bs], ispeed[bs] * mh_pixelscale, 'r*')
                ax_s.plot(frameTime[be], ispeed[be] * mh_pixelscale, 'k*')
                ax_s.set_ylabel('Instant speed [mm/s]')
                ax_s.set_xlabel('Time [s]')
                fig.tight_layout()

        # for each bout compute the distance between start and endpoint as well as the heading change
        bstarts = bouts[:, 0].astype(int)
        bends = bouts[:, 2].astype(int)
        boutRadius = np.sqrt((x_c[bends] - x_c[bstarts]) ** 2 + (y_c[bends] - y_c[bstarts]) ** 2)
        boutTheta = np.abs(core.AssignDeltaAnglesToBouts(bouts, heading)[0])
        boutCategory = np.zeros(bouts.shape[0])
        for i, (bs, be, r, t) in enumerate(zip(bstarts, bends, boutRadius, boutTheta)):
            if not inmiddle[bs]:
                boutCategory[i] = cdict["exclude"]
            else:
                if r <= 21.2 and np.abs(t) <= 5:  # slow straight
                    boutCategory[i] = cdict["slow straight"]
                elif r > 21.2 and np.abs(t) <= 5:  # fast (top 25%) straight
                    boutCategory[i] = cdict["fast straight"]
                elif r <= 21.2 and np.abs(t) > 5:  # slow turn
                    boutCategory[i] = cdict["slow turn"]
                elif r > 21.2 and np.abs(t) > 5:  # fast turn
                    boutCategory[i] = cdict["fast turn"]
                else:
                    boutCategory[i] = cdict["exclude"]
        # for each bout compute spline fit with overhang and then compute angular speed as well as curvature
        overhang = 100  # overhang used to ensure that the data used does not suffer from edge effects
        ang_speeds = np.array([])
        curvatures = np.array([])
        categories = np.array([])
        relTimes = np.array([])  # relative time within bout: 0-0.5 is before peak speed, 0.5-1 after till end of bout
        for b, categ in zip(bouts, boutCategory):
            if b[1] < b[0]:
                print('Strange bout call. Peak before start. Skipped', flush=True)
                continue
            if b[1] == b[0]:
                print('Peak at bout start. Skipped bout from analysis.', flush=True)
                continue
            # compute starts and ends of our stretch and only use if fully within dataset
            s = int(b[0] - overhang)
            if s < 0:
                continue
            e = int(b[2] + overhang)
            if e >= x_f.size:
                continue
            xb = x_f[s:e]
            yb = y_f[s:e]
            tck, u = core.spline_fit(xb * mh_pixelscale, yb * mh_pixelscale)
            a_spd = core.compute_angSpeed(tck, u, mh_datarate)
            curve = core.compute_curvature(tck, u)
            # strip fit-overhang frames
            start = overhang - int(preStartms / 1000 * mh_datarate)
            end = -1 * (overhang - 1)
            peak_frame = int(b[1] - b[0]) + int(preStartms / 1000 * mh_datarate)
            a_spd = a_spd[start:end]
            curve = curve[start:end]
            ct = np.full_like(curve, categ)
            # create our relative time vector
            rel_time = np.r_[np.linspace(0, 0.5, peak_frame, False), np.linspace(0.5, 1, curve.size - peak_frame)]
            ang_speeds = np.r_[ang_speeds, a_spd]
            curvatures = np.r_[curvatures, curve]
            categories = np.r_[categories, ct]
            relTimes = np.r_[relTimes, rel_time]
            bout_curves.append((core.cut_and_pad(curve, int(130 / 1000 * mh_datarate)), categ))
            bout_aspeeds.append((core.cut_and_pad(a_spd, int(130 / 1000 * mh_datarate)), categ))

        # remove nan-values
        nan_vals = np.logical_or(np.isnan(ang_speeds), np.isnan(curvatures))
        ang_speeds = ang_speeds[np.logical_not(nan_vals)]
        curvatures = curvatures[np.logical_not(nan_vals)]
        categories = categories[np.logical_not(nan_vals)]
        relTimes = relTimes[np.logical_not(nan_vals)]
        # compute linear fits and add to lists
        ss_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict["slow straight"],
                                cdict["slow straight"], basename)
        fs_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict["fast straight"],
                                cdict["fast straight"], basename)
        st_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict["slow turn"],
                                cdict["slow turn"], basename)
        ft_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict["fast turn"],
                                cdict["fast turn"], basename)
        all_fits.append(ss_fit)
        all_fits.append(fs_fit)
        all_fits.append(st_fit)
        all_fits.append(ft_fit)
        # plot overview scatter across different bout categories as well as linear fit
        xmin = -6
        xmax = 6
        if eid % 5 == 0:
            with sns.axes_style('white'):
                fig, axes = pl.subplots(nrows=2, ncols=2, sharey=True)
                axes = axes.ravel()
                ss_fit.PlotFit(axes[0], 'b')
                axes[0].set_ylabel('log10(Angular speed)')
                axes[0].set_xlabel('log10(Curvature)')
                axes[0].set_xlim(xmin, xmax)
                axes[0].set_title("Slow straight")
                sns.despine(ax=axes[0])
                fs_fit.PlotFit(axes[1], 'g')
                axes[1].set_ylabel('log10(Angular speed)')
                axes[1].set_xlabel('log10(Curvature)')
                axes[1].set_xlim(xmin, xmax)
                axes[1].set_title("Fast straight")
                sns.despine(ax=axes[1])
                st_fit.PlotFit(axes[2], 'm')
                axes[2].set_xlabel('log10(Curvature)')
                axes[2].set_xlim(xmin, xmax)
                axes[2].set_title("Slow turn")
                sns.despine(ax=axes[2])
                ft_fit.PlotFit(axes[3], 'r')
                axes[3].set_ylabel('log10(Angular speed)')
                axes[3].set_xlabel('log10(Curvature)')
                axes[3].set_xlim(xmin, xmax)
                axes[3].set_title("Fast turn")
                sns.despine(ax=axes[3])
                fig.tight_layout()
                if sv:
                    fig.savefig(basename + '_scatterFits.png', type='png')

    # collect aggregate data and plot
    slopes = pandas.DataFrame({cat_decode[k]: [ft.slope for ft in all_fits if ft.category == k]
                               for k in cat_decode if k != -1})
    intercepts = pandas.DataFrame({cat_decode[k]: [ft.intercept for ft in all_fits if ft.category == k]
                                   for k in cat_decode if k != -1})
    r_sq = pandas.DataFrame({cat_decode[k]: [ft.rsquared for ft in all_fits if ft.category == k]
                             for k in cat_decode if k != -1})
    with sns.axes_style('whitegrid'):
        fig, (ax_s, ax_k, ax_r) = pl.subplots(ncols=3)
        sns.boxplot(data=slopes, ax=ax_s, whis=np.inf, palette='muted')
        sns.swarmplot(data=slopes, ax=ax_s, color='k', size=4)
        ax_s.set_ylabel('Slope $\\beta$')
        ax_s.set_ylim(0.5, 1)
        sns.boxplot(data=intercepts, ax=ax_k, whis=np.inf, palette='muted')
        sns.swarmplot(data=intercepts, ax=ax_k, color='k', size=4)
        ax_k.set_ylabel('Intercept $k$')
        sns.boxplot(data=r_sq, ax=ax_r, whis=np.inf, palette='muted')
        sns.swarmplot(data=r_sq, ax=ax_r, color='k', size=4)
        ax_r.set_ylabel('$R^2$')
        ax_r.set_ylim(0, 1)
        fig.tight_layout()
        if sv:
            fig.savefig('slope_rsquared_overview.png', type='png')
