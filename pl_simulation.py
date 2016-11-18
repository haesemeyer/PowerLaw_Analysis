# -*- coding: utf-8 -*-
"""
Created on 11/17/2016

pl_simulation: A script to test whether the power-law naturally emerges due to sampling noise

@author: Martin Haesemeyer
"""

import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
import core
from scipy.stats import linregress


def trajectory(t):
    """
    For a given time t returns the exact trajectory position
    """
    # x(t) = np.sin(t)*t*0.001
    # y(t) = np.cos(t)*t*0.0005
    # dx_dt = 0.001*(np.cos(t)*t + np.sin(t))
    # dy_dt = 0.0005*(np.cos(t) - np.sin(t)*t)
    # d2x_dt2 = 0.001*(2*np.cos(t)-np.sin(t)*t)
    # d2y_dt2 = 0.0005*(-2*np.sin(t) - np.cos(t)*t)
    expand = t * 0.001
    return np.sin(t)*expand, np.cos(t)*expand*0.5


def velocity(t):
    """
    For a given time t returns the exact velocity
    """
    dx_dt = 0.001 * (np.cos(t) * t + np.sin(t))
    dy_dt = 0.0005 * (np.cos(t) - np.sin(t) * t)
    return np.sqrt(dx_dt**2 + dy_dt**2)


def curvature(t):
    """
    For a given time t returns the exact curvature
    """
    dx_dt = 0.001 * (np.cos(t) * t + np.sin(t))
    dy_dt = 0.0005*(np.cos(t) - np.sin(t)*t)
    d2x_dt2 = 0.001 * (2 * np.cos(t) - np.sin(t) * t)
    d2y_dt2 = 0.0005 * (-2 * np.sin(t) - np.cos(t) * t)
    c = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / velocity(t)**3
    return np.abs(c)


def angular_speed(t):
    """
    For a given time t returns the exact angular speed
    """
    dx_dt = 0.001 * (np.cos(t) * t + np.sin(t))
    dy_dt = 0.0005 * (np.cos(t) - np.sin(t) * t)
    angs = np.arctan2(dy_dt, dx_dt)
    d_angs = np.r_[np.abs(np.diff(angs)), 0]
    d_angs[d_angs > np.pi] = 2 * np.pi - d_angs[d_angs > np.pi]
    return d_angs


def fit_plot(x, y, ax, c='k'):
    if ax is None:
        fig, ax = pl.subplots()
    ax.scatter(x, y, c=c, s=5, alpha=0.5)
    no_nan = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    slope, inter, r = linregress(x[no_nan], y[no_nan])[0:3]
    xmin = x[no_nan].min()
    xmax = x[no_nan].max()
    y1 = xmin*slope + inter
    y2 = xmax*slope + inter
    ax.plot([xmin, xmax], [y1, y2], color=c, ls="--")
    ax.text(xmin, y2, "$R^2$ = " + str(round(r**2, 2)))

if __name__ == "__main__":
    realtime = np.linspace(0, 100, 100000)
    # plot our trajectory and exact measures
    with sns.axes_style('white'):
        fig, axes = pl.subplots(2, 2)
        x, y = trajectory(realtime)
        axes[0, 0].plot(x, y)
        axes[0, 0].set_title('Trajectory')
        axes[0, 1].plot(realtime, np.log10(velocity(realtime)))
        axes[0, 1].set_title('log10(Tangential velocity)')
        axes[1, 0].plot(realtime, np.log10(curvature(realtime)))
        axes[1, 0].set_title('log10(Curvature)')
        axes[1, 1].plot(realtime, np.log10(angular_speed(realtime)))
        axes[1, 1].set_title('log10(Angular speed)')
        sns.despine()
        fig.tight_layout()

    # scatter plot of angular speed versus curvature
    l10_curve = np.log10(curvature(realtime)[:-1])
    l10_as = np.log10(angular_speed(realtime)[:-1])
    nan_vals = np.logical_or(np.isnan(l10_curve), np.isnan(l10_as))
    l10_curve = l10_curve[np.logical_not(nan_vals)]
    l10_as = l10_as[np.logical_not(nan_vals)]
    with sns.axes_style('white'):
        sns.jointplot(l10_curve, l10_as, kind='regplot').set_axis_labels('log10(Curvature)', 'log10(Angular speed)')

    # our function generated trajectory has a "natural" relationship between curvature and angular speed as also shown
    # in the plot above. To break this up, we will finely resample the trajectory adjusting our velocity such that it
    # is roughly k/C(t). We do this via timewarping to sample our trajectory such that we sample more finely (reduce
    # velocity) wherever curvature is high.

    def timewarp(timepoints):
        assert type(timepoints) == np.ndarray
        true_curve = curvature(timepoints)
        l_table = 5 / true_curve
        warped = np.full_like(l_table, l_table[0])
        for i in range(1, l_table.size):
            warped[i] = warped[i-1] + l_table[i]
        return warped / warped.max() * timepoints.max()

    sample_times = timewarp(realtime)
    # use spline-fitting to obtain angular speed and curvature on the noiseless trace
    tjx, tjy = trajectory(sample_times)
    tck, u = core.spline_fit(tjx, tjy)
    a_spd = core.compute_angSpeed(tck, u, 1)
    curve = core.compute_curvature(tck, u)
    t_vel = core.compute_tangVelocity(tck, u, 1)
    # plot our resampled trajectory and measures
    with sns.axes_style('white'):
        fig, axes = pl.subplots(2, 2)
        axes[0, 0].plot(tjx, tjy)
        axes[0, 0].set_title('Trajectory - Resampled')
        axes[0, 1].plot(realtime, np.log10(t_vel))
        axes[0, 1].set_title('log10(Tangential velocity)')
        axes[1, 0].plot(realtime, np.log10(curve))
        axes[1, 0].set_title('log10(Curvature) - Resampled')
        axes[1, 1].plot(realtime, np.log10(a_spd))
        axes[1, 1].set_title('log10(Angular speed) - Resampled')
        sns.despine()
        fig.tight_layout()
    with sns.axes_style('white'):
        sns.jointplot(np.log10(curve[:-1]), np.log10(a_spd[:-1]), kind='regplot')\
            .set_axis_labels('log10(Curvature)', 'log10(Angular speed)')
