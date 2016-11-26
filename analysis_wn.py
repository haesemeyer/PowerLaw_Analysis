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

        # for each bout compute the heading change
        bstarts = bouts[:, 0].astype(int)
        bends = bouts[:, 2].astype(int)
        boutSpeed = bouts[:, -1]  # peak speed of bouts
        boutTheta = np.abs(core.AssignDeltaAnglesToBouts(bouts, heading)[0])
        boutCategory = np.zeros(bouts.shape[0])
        for i, (bs, be, r, t) in enumerate(zip(bstarts, bends, boutSpeed, boutTheta)):
            if not inmiddle[bs]:
                boutCategory[i] = cdict["exclude"]
            else:
                if r <= 149 and np.abs(t) <= 5:  # slow straight
                    boutCategory[i] = cdict["slow straight"]
                elif r > 149 and np.abs(t) <= 5:  # fast (top 50%) straight
                    boutCategory[i] = cdict["fast straight"]
                elif r <= 149 and np.abs(t) > 5:  # slow turn
                    boutCategory[i] = cdict["slow turn"]
                elif r > 149 and np.abs(t) > 5:  # fast turn
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
                axes[1].set_xlabel('log10(Curvature)')
                axes[1].set_xlim(xmin, xmax)
                axes[1].set_title("Fast straight")
                sns.despine(ax=axes[1])
                st_fit.PlotFit(axes[2], 'm')
                axes[2].set_ylabel('log10(Angular speed)')
                axes[2].set_xlabel('log10(Curvature)')
                axes[2].set_xlim(xmin, xmax)
                axes[2].set_title("Slow turn")
                sns.despine(ax=axes[2])
                ft_fit.PlotFit(axes[3], 'r')
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
    plot_order = [cat_decode[k] for k in cat_decode if k != -1]
    with sns.axes_style('whitegrid'):
        fig, (ax_s, ax_k, ax_r) = pl.subplots(ncols=3)
        sns.boxplot(data=slopes, ax=ax_s, whis=np.inf, palette='muted', order=plot_order)
        sns.swarmplot(data=slopes, ax=ax_s, color='k', size=4, order=plot_order)
        ax_s.set_ylabel('Slope $\\beta$')
        ax_s.set_ylim(0.5, 1)
        sns.boxplot(data=intercepts, ax=ax_k, whis=np.inf, palette='muted', order=plot_order)
        sns.swarmplot(data=intercepts, ax=ax_k, color='k', size=4, order=plot_order)
        ax_k.set_ylabel('Intercept $k$')
        sns.boxplot(data=r_sq, ax=ax_r, whis=np.inf, palette='muted', order=plot_order)
        sns.swarmplot(data=r_sq, ax=ax_r, color='k', size=4, order=plot_order)
        ax_r.set_ylabel('$R^2$')
        ax_r.set_ylim(0, 1)
        fig.tight_layout()
        if sv:
            fig.savefig('slope_rsquared_overview.png', type='png')

    # for each of our categories, plot the average development of angular speed and curvature
    bc_ss = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'slow straight'])
    ba_ss = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'slow straight'])
    bc_fs = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'fast straight'])
    ba_fs = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'fast straight'])
    bc_st = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'slow turn'])
    ba_st = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'slow turn'])
    bc_ft = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'fast turn'])
    ba_ft = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'fast turn'])
    plotTime = (np.arange(bc_ss.shape[1])-int(preStartms / 1000 * mh_datarate)) / mh_datarate * 1000
    with sns.axes_style('whitegrid'):
        fig, (ax_c, ax_a) = pl.subplots(nrows=2, sharex=True)
        sns.tsplot(data=bc_ss, time=plotTime, estimator=np.nanmean, ci=95, color='b', ax=ax_c)
        sns.tsplot(data=bc_fs, time=plotTime, estimator=np.nanmean, ci=95, color='g', ax=ax_c)
        sns.tsplot(data=bc_st, time=plotTime, estimator=np.nanmean, ci=95, color='m', ax=ax_c)
        sns.tsplot(data=bc_ft, time=plotTime, estimator=np.nanmean, ci=95, color='r', ax=ax_c)
        ax_c.set_ylabel('log10(Curvature)')
        sns.tsplot(data=ba_ss, time=plotTime, estimator=np.nanmean, ci=95, color='b', ax=ax_a)
        sns.tsplot(data=ba_fs, time=plotTime, estimator=np.nanmean, ci=95, color='g', ax=ax_a)
        sns.tsplot(data=ba_st, time=plotTime, estimator=np.nanmean, ci=95, color='m', ax=ax_a)
        sns.tsplot(data=ba_ft, time=plotTime, estimator=np.nanmean, ci=95, color='r', ax=ax_a)
        ax_a.set_ylabel('log10(Angular speed)')
        ax_a.set_xlabel('Time [ms]')
        fig.tight_layout()
        if sv:
            fig.savefig('Speed_Curve_Bout_development.png', type='png')

    # analyze beta and k according to relative bout time
    rt_edges = np.linspace(0, 1, 5)
    rt_centers = rt_edges[:-1] + np.diff(rt_edges)/2
    ss_timed_beta = np.zeros((slopes.shape[0], rt_centers.size))
    ss_timed_k = np.zeros_like(ss_timed_beta)
    fs_timed_beta = np.zeros_like(ss_timed_beta)
    fs_timed_k = np.zeros_like(ss_timed_beta)
    st_timed_beta = np.zeros_like(ss_timed_beta)
    st_timed_k = np.zeros_like(ss_timed_beta)
    ft_timed_beta = np.zeros_like(ss_timed_beta)
    ft_timed_k = np.zeros_like(ss_timed_beta)
    ss_count = fs_count = st_count = ft_count = 0
    for ft in all_fits:
        if ft.category == cdict["slow straight"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                ss_timed_beta[ss_count, j] = beta
                ss_timed_k[ss_count, j] = k
            ss_count += 1
        elif ft.category == cdict["fast straight"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                fs_timed_beta[fs_count, j] = beta
                fs_timed_k[fs_count, j] = k
            fs_count += 1
        elif ft.category == cdict["slow turn"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                st_timed_beta[st_count, j] = beta
                st_timed_k[st_count, j] = k
            st_count += 1
        elif ft.category == cdict["fast turn"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                ft_timed_beta[ft_count, j] = beta
                ft_timed_k[ft_count, j] = k
            ft_count += 1

    with sns.axes_style('whitegrid'):
        fig, (ax_b, ax_k) = pl.subplots(nrows=2, sharex=True)
        sns.tsplot(data=ss_timed_beta, time=rt_centers, color='b', ax=ax_b, interpolate=False, ci=95)
        sns.tsplot(data=fs_timed_beta, time=rt_centers, color='g', ax=ax_b, interpolate=False, ci=95)
        sns.tsplot(data=st_timed_beta, time=rt_centers, color='m', ax=ax_b, interpolate=False, ci=95)
        sns.tsplot(data=ft_timed_beta, time=rt_centers, color='r', ax=ax_b, interpolate=False, ci=95)
        ax_b.plot([0.5, 0.5], ax_b.get_ylim(), 'k--')
        ax_b.set_ylabel('Slope $\\beta$')
        sns.tsplot(data=ss_timed_k, time=rt_centers, color='b', ax=ax_k, interpolate=False, ci=95)
        sns.tsplot(data=fs_timed_k, time=rt_centers, color='g', ax=ax_k, interpolate=False, ci=95)
        sns.tsplot(data=st_timed_k, time=rt_centers, color='m', ax=ax_k, interpolate=False, ci=95)
        sns.tsplot(data=ft_timed_k, time=rt_centers, color='r', ax=ax_k, interpolate=False, ci=95)
        ax_k.plot([0.5, 0.5], ax_k.get_ylim(), 'k--')
        ax_k.set_ylabel('Intercept $k$')
        ax_k.set_xlabel('Relative bout time [AU]')
        fig.tight_layout()
        if sv:
            fig.savefig('Beta_K_development.png', type='png')
