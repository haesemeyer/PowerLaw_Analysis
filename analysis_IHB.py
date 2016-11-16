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
import pandas
import os
from scipy.stats import linregress

ihb_datarate = 700  # acquisition and datarate in Hz
ihb_pixelscale = 1 / 24.8  # pixelsize in mm
# start data 20 ms earlier than actual start since turn often happens
# before the instant speed bout call assigns the start
preStartms = 30

# bout category dictionaries
cdict = {"exclude": -1, "regular": 0, "hunting": 1, "escape": 2}
cat_decode = {v: k for k, v in cdict.items()}  # inverse dictionary to later get our names back easily

if __name__ == '__main__':
    sv = ''
    while sv != 'y' and sv != 'n':
        sv = input('Save figures? [y/n]')
    sv = sv == 'y'
    fnames = core.UiGetFile(filetypes=[('Matlab file', '.mat')], diagTitle='Load data files')
    all_fits = []  # list of log/log fits for each experiment
    bout_curves = []  # list of 2-element tuples with curvature for each bout and it's category
    bout_aspeeds = []  # list of 2-element tuples with angular speeds for each bouts and it's category

    for eid, name in enumerate(fnames):
        basename = os.path.basename(name)
        basename = basename[:basename.find('.mat')]
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
        x_f, y_f = core.SmoothenTrack(x_c.copy(), y_c.copy(), 21)
        ispeed = core.ComputeInstantSpeed(x_f, y_f, ihb_datarate)
        frameTime = np.arange(x_c.size) / ihb_datarate
        bouts = core.DetectBouts(ispeed, 50, 500, speedThresholdAbsolute=35, maxFramesAtPeak=10)

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
            if sv:
                fig.savefig(basename + '_scatterBoutChars.png', type='png')

        # for each bout compute spline fit with overhang and then compute angular speed as well as curvature
        overhang = 300  # overhang used to ensure that the data used does not suffer from edge effects
        ang_speeds = np.array([])
        curvatures = np.array([])
        categories = np.array([])
        relTimes = np.array([])  # relative time within bout: 0-0.5 is before peak speed, 0.5-1 after till end of bout
        for b, categ in zip(bouts, boutCategory):
            if b[1] <= b[0]:
                print('Strange bout call. Peak before start. Skipped', flush=True)
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
            tck, u = core.spline_fit(xb * ihb_pixelscale, yb * ihb_pixelscale)
            a_spd = core.compute_angSpeed(tck, u, ihb_datarate)
            curve = core.compute_curvature(tck, u)
            # strip fit-overhang frames
            start = overhang - int(preStartms / 1000 * ihb_datarate)
            end = -1 * (overhang - 1)
            peak_frame = int(b[1] - b[0]) + int(preStartms / 1000 * ihb_datarate)
            a_spd = a_spd[start:end]
            curve = curve[start:end]
            ct = np.full_like(curve, categ)
            # create our relative time vector
            rel_time = np.r_[np.linspace(0, 0.5, peak_frame, False), np.linspace(0.5, 1, curve.size - peak_frame)]
            ang_speeds = np.r_[ang_speeds, a_spd]
            curvatures = np.r_[curvatures, curve]
            categories = np.r_[categories, ct]
            relTimes = np.r_[relTimes, rel_time]
            bout_curves.append((core.cut_and_pad(curve, int(130 / 1000 * ihb_datarate)), categ))
            bout_aspeeds.append((core.cut_and_pad(a_spd, int(130 / 1000 * ihb_datarate)), categ))

        # remove nan-values
        nan_vals = np.logical_or(np.isnan(ang_speeds), np.isnan(curvatures))
        ang_speeds = ang_speeds[np.logical_not(nan_vals)]
        curvatures = curvatures[np.logical_not(nan_vals)]
        categories = categories[np.logical_not(nan_vals)]
        relTimes = relTimes[np.logical_not(nan_vals)]
        # compute linear fits and add to lists
        reg_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict['regular'], cdict['regular'],
                                 basename)
        hnt_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict['hunting'], cdict['hunting'],
                                 basename)
        esc_fit = core.LogLogFit(curvatures, ang_speeds, relTimes, categories == cdict['escape'], cdict['escape'],
                                 basename)
        all_fits.append(reg_fit)
        all_fits.append(hnt_fit)
        all_fits.append(esc_fit)
        # plot overview scatter across different bout categories as well as linear fit
        xmin = -6
        xmax = 6
        with sns.axes_style('white'):
            fig, axes = pl.subplots(ncols=3, sharey=True)
            reg_fit.PlotFit(axes[0], 'b')
            axes[0].set_ylabel('log10(Angular speed)')
            axes[0].set_xlabel('log10(Curvature)')
            axes[0].set_xlim(xmin, xmax)
            axes[0].set_title("Regular bouts")
            sns.despine(ax=axes[0])
            hnt_fit.PlotFit(axes[1], 'g')
            axes[1].set_xlabel('log10(Curvature)')
            axes[1].set_xlim(xmin, xmax)
            axes[1].set_title("Hunting bouts")
            sns.despine(ax=axes[1])
            esc_fit.PlotFit(axes[2], 'r')
            axes[2].set_xlabel('log10(Curvature)')
            axes[2].set_xlim(xmin, xmax)
            axes[2].set_title("Escapes")
            sns.despine(ax=axes[2])
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
        ax_s.set_ylim(0.5, 0.75)
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

    # for each of our categories, plot the average development of angular speed and curvature
    bc_reg = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'regular'])
    ba_reg = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'regular'])
    bc_hnt = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'hunting'])
    ba_hnt = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'hunting'])
    bc_esc = np.vstack([np.log10(bc[0]) for bc in bout_curves if cat_decode[bc[1]] == 'escape'])
    ba_esc = np.vstack([np.log10(ac[0]) for ac in bout_aspeeds if cat_decode[ac[1]] == 'escape'])
    plotTime = (np.arange(bc_reg.shape[1])-int(preStartms / 1000 * ihb_datarate)) / ihb_datarate * 1000
    with sns.axes_style('whitegrid'):
        fig, (ax_c, ax_a) = pl.subplots(nrows=2, sharex=True)
        sns.tsplot(data=bc_reg, time=plotTime, estimator=np.nanmean, ci=95, color='b', ax=ax_c)
        sns.tsplot(data=bc_hnt, time=plotTime, estimator=np.nanmean, ci=95, color='g', ax=ax_c)
        sns.tsplot(data=bc_esc, time=plotTime, estimator=np.nanmean, ci=95, color='r', ax=ax_c)
        ax_c.set_ylabel('log10(Curvature)')
        sns.tsplot(data=ba_reg, time=plotTime, estimator=np.nanmean, ci=95, color='b', ax=ax_a)
        sns.tsplot(data=ba_hnt, time=plotTime, estimator=np.nanmean, ci=95, color='g', ax=ax_a)
        sns.tsplot(data=ba_esc, time=plotTime, estimator=np.nanmean, ci=95, color='r', ax=ax_a)
        ax_a.set_ylabel('log10(Angular speed)')
        ax_a.set_xlabel('Time [ms]')
        fig.tight_layout()
        if sv:
            fig.savefig('Speed_Curve_Bout_development.png', type='png')

    # analyze beta and k according to relative bout time
    rt_edges = np.linspace(0, 1, 5)
    rt_centers = rt_edges[:-1] + np.diff(rt_edges)/2
    reg_timed_beta = np.zeros((len(fnames), rt_centers.size))
    reg_timed_k = np.zeros_like(reg_timed_beta)
    hnt_timed_beta = np.zeros_like(reg_timed_beta)
    hnt_timed_k = np.zeros_like(reg_timed_beta)
    esc_timed_beta = np.zeros_like(reg_timed_beta)
    esc_timed_k = np.zeros_like(reg_timed_beta)
    reg_count = hnt_count = esc_count = 0
    for ft in all_fits:
        if ft.category == cdict["regular"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                reg_timed_beta[reg_count, j] = beta
                reg_timed_k[reg_count, j] = k
            reg_count += 1
        elif ft.category == cdict["hunting"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                hnt_timed_beta[hnt_count, j] = beta
                hnt_timed_k[hnt_count, j] = k
            hnt_count += 1
        elif ft.category == cdict["escape"]:
            for j in range(rt_centers.size):
                take = np.logical_and(ft.relativeTime >= rt_edges[j], ft.relativeTime < rt_edges[j+1])
                beta, k = linregress(ft.logCurvature[take], ft.logAngularSpeed[take])[:2]
                esc_timed_beta[esc_count, j] = beta
                esc_timed_k[esc_count, j] = k
            esc_count += 1

    with sns.axes_style('whitegrid'):
        fig, (ax_b, ax_k) = pl.subplots(nrows=2, sharex=True)
        sns.tsplot(data=reg_timed_beta, time=rt_centers, color='b', ax=ax_b, interpolate=False, ci=95)
        sns.tsplot(data=hnt_timed_beta, time=rt_centers, color='g', ax=ax_b, interpolate=False, ci=95)
        sns.tsplot(data=esc_timed_beta, time=rt_centers, color='r', ax=ax_b, interpolate=False, ci=95)
        ax_b.plot([0.5, 0.5], ax_b.get_ylim(), 'k--')
        ax_b.set_ylabel('Slope $\\beta$')
        sns.tsplot(data=reg_timed_k, time=rt_centers, color='b', ax=ax_k, interpolate=False, ci=95)
        sns.tsplot(data=hnt_timed_k, time=rt_centers, color='g', ax=ax_k, interpolate=False, ci=95)
        sns.tsplot(data=esc_timed_k, time=rt_centers, color='r', ax=ax_k, interpolate=False, ci=95)
        ax_k.plot([0.5, 0.5], ax_k.get_ylim(), 'k--')
        ax_k.set_ylabel('Intercept $k$')
        ax_k.set_xlabel('Relative bout time [AU]')
        fig.tight_layout()
        if sv:
            fig.savefig('Beta_K_development.png', type='png')
