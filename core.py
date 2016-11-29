# -*- coding: utf-8 -*-
"""
Created on 11/14/2016

core: Contains analysis support functions, mostly from mhba_basic
@author: Martin Haesemeyer
"""

import tkinter
from tkinter import filedialog
import numpy as np
from peakfinder import peakdet
from scipy.signal import filtfilt
from scipy import interpolate
from scipy.stats import linregress
import matplotlib.pyplot as pl
import seaborn as sns
import matplotlib.cm as cm
import h5py


def UiGetFile(filetypes=[('Matlab file', '.mat')], diagTitle="Load files", multiple=True):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    options = {'filetypes': filetypes, 'multiple': multiple, 'title': diagTitle}
    tkinter.Tk().withdraw()  # Close the root window
    return filedialog.askopenfilename(**options)


def DetectBouts(instantSpeed, minFramesPerBout, frameRate, **kwargs):
    """
    Detect bouts based on emprically determined characterstics within a trace
    of instant speeds.
    """
    # bouts should have a clear peak, hence the peakspeed should not be
    # maintained for >~20ms
    if 'maxFramesAtPeak' in kwargs:
        maxFramesAtPeak = kwargs['maxFramesAtPeak']
    else:
        maxFramesAtPeak = 7
    # the speed threshold is determined mostly empirically and probably needs
    # to be changed if our framerate deviates much from 250 - this depends on
    # how aggressively the position traces are smoothed in relation to the
    # framerate
    if 'speedThresholdAbsolute' in kwargs:
        speedThresholdAbsolute = kwargs['speedThresholdAbsolute']
    else:
        speedThresholdAbsolute = 18

    # threshold speeds
    spds = instantSpeed.copy()
    spds[spds < speedThresholdAbsolute] = 0

    # master indexer
    i = 0
    bouts = np.zeros((1, 5))

    l = np.size(spds)

    while i < l:
        if np.isnan(spds[i]) or spds[i] == 0:
            i += 1
        else:
            # we found a potential "start of bout" - loop till end collecting
            # all speeds
            bStart = i
            b = spds[i]
            i += 1
            while i < l and spds[i] > 0:
                b = np.append(b, spds[i])
                i += 1
            peak = b[b == b.max()]
            bEnd = i - 1  # since we loop until speed==0 the last part of the bout is in the frame before this frame
            # look at our criteria to decide whether we take this bout or not
            if peak.size <= maxFramesAtPeak and b.size >= minFramesPerBout:
                # if fish initiate bouts before coming to rest (as during faster
                # OMR presentation) we may have a "super bout" consisting of
                # multiple concatenated bouts - try using peakfinder to determine
                # if this is the case
                # these criteria alone lead to way too many bout-splits
                # a) only bouts which are at least 2*minFramesPerBout should
                # be split and each split should still retain at least minFram...
                # b) the minima that are used to split the bout should be rather
                # small - smaller than 1.5*speedThreshold... and smaller
                # than half of the average speed of its surrounding peaks to
                # make sure that the fish was actually approaching rest rather
                # than just slowing down its swim a bit

                pmaxs, pmins = peakdet(b)
                try:
                    peakLoc = pmaxs[:, 0]
                    peakMag = pmaxs[:, 1]
                except IndexError:  # rare case: Absolutely no peak found?
                    peakLoc = np.zeros(1)
                if b.size > 2 * minFramesPerBout and peakLoc.size > 1:
                    # find minima in between peaks and make sure that they are
                    # small enough
                    possibleMinima = np.array([], dtype=float)
                    for p in range(np.size(peakLoc) - 1):
                        minval = b[peakLoc[p]:peakLoc[p + 1]].min()
                        if minval < speedThresholdAbsolute * 1.5 and minval < (peakMag[p] + peakMag[p + 1]) / 4:
                            mn = np.nonzero(b[peakLoc[p]:peakLoc[p + 1]] == minval) + peakLoc[p]
                            possibleMinima = np.append(possibleMinima, mn)
                    # for p in range...
                    # each minimum splits the bout - make sure that on both
                    # sides of each there is still at least minFramesPerBout left
                    if possibleMinima.size > 0:
                        lm = 0
                        for p in range(np.size(possibleMinima)):
                            if possibleMinima[p] - lm < minFramesPerBout:
                                possibleMinima[p] = float('NaN')
                            else:
                                lm = possibleMinima[p]
                        # for p in range...
                        possibleMinima = possibleMinima[np.nonzero(~np.isnan(possibleMinima))]
                        # if(have possible minima)
                else:
                    possibleMinima = np.array([])

                if possibleMinima.size > 0:
                    # we should split our bout
                    nBouts = np.size(possibleMinima) + 1
                    allSplitBouts = np.zeros((nBouts, 5))
                    # update quantities to be relative to experiment start
                    # rather than bout start
                    minima = possibleMinima + bStart
                    allSplitStarts = np.zeros(nBouts, dtype=int)
                    allSplitEnds = np.zeros_like(allSplitStarts)
                    allSplitStarts[0] = bStart
                    allSplitStarts[1::] = minima + 1
                    allSplitEnds[0:-1] = minima
                    allSplitEnds[-1] = bEnd
                    # find the summed displacement and peakframe for each
                    # sub-bout
                    allSplitDisp = np.zeros(nBouts)
                    allPeaks = np.zeros(nBouts)
                    allPeakMags = np.zeros(nBouts)
                    for k in range(nBouts):
                        allSplitDisp[k] = np.sum(spds[allSplitStarts[k]:allSplitEnds[k]], dtype=float) / frameRate
                        pk = np.argmax(spds[allSplitStarts[k]:allSplitEnds[k]])
                        allPeaks[k] = pk + allSplitStarts[k]
                        assert allPeaks[k] >= allSplitStarts[k]
                        assert allPeaks[k] <= allSplitEnds[k]
                    # assign all
                    allSplitBouts[:, 0] = allSplitStarts
                    allSplitBouts[:, 1] = allPeaks
                    allSplitBouts[:, 2] = allSplitEnds
                    allSplitBouts[:, 3] = allSplitDisp
                    allSplitBouts[:, 4] = allPeakMags
                    # add our bouts
                    bouts = np.vstack((bouts, allSplitBouts))
                else:
                    # we have a valid, singular bout
                    peakframe = np.argmax(b)
                    # put peakframe into context
                    peakframe = bStart + peakframe
                    # a bout should not start on the peakframe
                    if bStart == peakframe:
                        bStart -= 1
                    bt = np.r_[bStart, peakframe, bEnd, np.sum(b, dtype=float) / frameRate, b.max()]
                    bouts = np.vstack((bouts, bt))
    # outer while loop
    if bouts.size > 1:
        bouts = bouts[1::, :]
    return bouts


def SmoothenTrack(xpos, ypos, window_len):
    """
    Smoothens a fish track with a specified smoothing window, removing all NaN
    from the traces before smoothing
       xpos: The xposition trace
       ypos: The yposition trace
       window_len: The size of the smoothing window
       RETURNS:
          [0]: The smoothed x-position trace
          [1]: The smoothed y-position trace
    """
    # remove all NaN frames smoothing across the gap
    nonanX = xpos[np.logical_not(np.isnan(xpos))]
    nonanY = ypos[np.logical_not(np.isnan(ypos))]
    nonanX = filtfilt(np.ones(window_len)/window_len, 1, nonanX)
    nonanY = filtfilt(np.ones(window_len)/window_len, 1, nonanY)
    # put back in original trace
    xpos[np.logical_not(np.isnan(xpos))] = nonanX
    ypos[np.logical_not(np.isnan(ypos))] = nonanY
    return xpos, ypos


def ComputeInstantSpeed(xpos, ypos, frameRate=250):
    """
    Takes a (smoothened) x and y position trace and computes the instant speed
    in pixels per second
       xpos: The x-position trace
       ypos: The y-position trace
       frameRate: The acquisition framerate for normalization
       RETURNS:
           The instant speed of the fish
    """
    dx = np.r_[0, np.diff(xpos)]
    dy = np.r_[0, np.diff(ypos)]
    return np.sqrt(dx**2 + dy**2) * frameRate


def AssignDeltaAnglesToBouts(bouts, ang):
    """
    Uses the bouts identified in bouts together with an angle trace to
    determine the heading angle of the fish before bout start and after
    completion in order to determine the turn angle of the bout.
    """
    # takes a bout structure and an angle trace and uses pre/post bout averaging
    # of angles to compute a deltaAngle for each bout
    # returns deltaAngles for each bout

    # we have to overcome the following problem: Angles are a circular quantity.
    # This means that if the heading angle fluctuates around 0 it would actually
    # get averaged to 180 ( (0+360)/2 ) instead of 0.
    # Therefore, angles should be added as vector sums. Therefore for all pre-
    # and post angles we compute the x and y vector component using cosine and
    # sine respectively. These are added together and the pre- and post angles
    # are computed using atan2. Subsequently the delta angle is computed based
    # on the pre and post angles.
    m = bouts.shape[0]
    dAngles = np.zeros(m)
    preAngle = np.zeros(m)
    postAngle = np.zeros(m)

    mal = 50  # the maximum number of frames chosen across which we average

    angles = np.deg2rad(ang - 180)  # renorm and copy

    for i in range(m):
        bStart = bouts[i, 0].astype(int)
        bEnd = bouts[i, 2].astype(int)
        if i == 0:  # for first bout we take whole past trace for start angle computation
            if bStart <= mal:
                sf = 1
            else:
                sf = bStart - mal
        else:
            bPrevEnd = bouts[i - 1, 2].astype(int)
            if bStart - bPrevEnd <= mal:
                sf = bPrevEnd
            else:
                sf = bStart - mal
        # if(i==0)
        preBoutX = np.nansum(np.cos(angles[sf:bStart]))
        preBoutY = np.nansum(np.sin(angles[sf:bStart]))

        if i == m - 1:  # for last bout we take trace up to end of experiment
            if angles.size - bEnd <= mal:
                ef = np.size(angles) - 1
            else:
                ef = bEnd + mal
        else:
            bNextStart = bouts[i + 1, 0].astype(int)
            if bNextStart - bEnd <= mal:
                ef = bNextStart
            else:
                ef = bEnd + mal
                if ef >= angles.size:
                    ef = np.size(angles) - 1
        postBoutX = np.nansum(np.cos(angles[bEnd:ef]))
        postBoutY = np.nansum(np.sin(angles[bEnd:ef]))
        preAngle[i] = np.rad2deg(np.arctan2(preBoutY, preBoutX))
        postAngle[i] = np.rad2deg(np.arctan2(postBoutY, postBoutX))
        dAngles[i] = ComputeMinorDeltaAngle(preAngle[i], postAngle[i])
    return dAngles, preAngle, postAngle


def ComputeMinorDeltaAngle(angleFirst, angleSecond):
    """
    Computes the minimum angle of rotation going from angleFirst to
    angleSecond
    """

    if angleFirst.size > 1 or angleSecond.size > 1:
        # we have vector data - make sure it conforms
        m1 = np.shape(angleFirst)
        m2 = np.shape(angleSecond)

        if np.size(m1) > 1 or np.size(m2) > 1:
            raise ValueError('Function only accepts vector or scalar data')

        if m1 != m2:
            raise ValueError('Inputs to the function must have the same dimensions')

    if angleFirst.size == 1:
        # scalar data
        dangle = angleSecond - angleFirst
        if dangle > 180:
            delta = dangle - 360
        else:
            if dangle < -180:
                delta = 360 + dangle
            else:
                delta = dangle
    else:
        # vector data
        dangle = angleSecond - angleFirst
        delta = dangle
        delta[dangle > 180] = dangle[dangle > 180] - 360
        delta[dangle < -180] = dangle[dangle < -180] + 360
    return delta


def spline_fit(x, y):
    """
    Compute smoothened spline fit
    """
    return interpolate.splprep([x, y], u=np.arange(x.size), s=0.003)[:2]


def compute_fitCoords(tck, u):
    """
    Compute coordinates of the given spline fit
    """
    return interpolate.splev(u, tck)


def compute_tangVelocity(tck, u, frameRate):
    """
    Computes the tangential velocity
    """
    dx, dy = interpolate.splev(u, tck, der=1)
    return np.sqrt(dx**2 + dy**2) * frameRate


def compute_angSpeed(tck, u, frameRate):
    """
    Compute the angular  speed of the tangent on the curve in
    radians per second
    """
    dx, dy = interpolate.splev(u, tck, der=1)
    angs = np.arctan2(dy, dx)
    d_angs = np.r_[np.abs(np.diff(angs)), 0]
    d_angs[d_angs > np.pi] = 2 * np.pi - d_angs[d_angs > np.pi]
    return d_angs * frameRate


def compute_curvature(tck, u):
    """
    Compute the curvature of a trajectory at each given point
    """
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    curves = (dx * ddy - dy * ddx) / ((dx * dx + dy * dy) ** (3 / 2))
    return np.abs(curves)


def compute_plFit(cu, av, take):
    """
    Computes the linear fit between log10(cu) and log10(av)
    Args:
        cu: The curvature values
        av: The angular velocity values
        take: Logical identifying which cu and av should be part of the calculation

    Returns:
        [0]: The power law exponent (slope of the fit)
        [1]: The intercept
        [2]: The r-value
    """
    cut = cu[take]
    avt = av[take]
    keep = np.logical_and(cut > 0, avt > 0)
    return linregress(np.log10(cut[keep]), np.log10(avt[keep]))[0:3]


def cut_and_pad(trace, size):
    """
    Cuts trace to the desired size if too long and pads at the end with nan if too short
    """
    if trace.size > size:
        return trace[:size].copy()
    else:
        retval = np.full(size, np.nan)
        retval[:trace.size] = trace
        return retval


class Experiment:
    """
    Describes one generic power law experiment
    """
    def __init__(self, key, filename, datarate, pixelsize):
        """
        Creates a new Experiment
        :param key: The data key in the hdf5 file
        :param filename: The name of the hdf5 file containing the experiment
        """
        self.key = key
        self.filename = filename
        self.fits = []
        self.datarate = datarate
        self.pixelsize = pixelsize
        self.bout_curves = []
        self.bout_aspeeds = []
        self.bout_tang_vels = []
        self.bouts = []
        self.bout_categories = []
        # determine filter window size (based on empirical tests)
        if self.datarate == 250:
            self.filter_window = 11
        elif self.datarate == 700:
            self.filter_window = 21
        else:
            self.filter_window = min(int(11/250*self.datarate), 21)
        # bout category dictionaries - these defaults will likely be overridden by derived classes
        self.cdict = {"exclude": -1, "take": 0}
        self.cat_decode = {v: k for k, v in self.cdict.items()}  # inverse dictionary to later get our names back easily

    def load_data(self, file=None):
        """
        Load data from file and process according to experiment type
        :param file: If provided is assumed to be a dictionary of data objects such as an hdf5 file
        :return: The extracted experiment data
        """
        if file is None:
            dfile = h5py.File(self.filename, 'r')
            exp_data = np.array(dfile[self.key])
            dfile.close()
        else:
            exp_data = np.array(file[self.key])
        return self._extract_data(exp_data)

    def _extract_data(self, data):
        """
        Subclass specific procedure to extract data
        for given experiment type
        :param data: The raw data array of the Experiment
        :return: x, y, heading, valid, (exp_specific)
        """
        return None, None, None, None

    def _detect_bouts(self, instantSpeed):
        """
        Bout detection. Subclasses should override to tune parameters
        :param instantSpeed: The instant speed trace
        :return: Bouts matrix
        """
        generic_minframes = 70 / 1000 * self.datarate  # min 70ms per bout
        generic_spdThreshold = 0.05 * self.datarate
        return DetectBouts(instantSpeed, generic_minframes, self.datarate, speedThresholdAbsolute=generic_spdThreshold)

    def _compute_fits(self, overhang, pre_start_ms, x_f, y_f):
        """
        Compute curvature and angular speed and tangential velocity for each bout and create log-log-fits
        :param overhang: Number of frames to include around bout in spline fit to avoid edge effects
        :param pre_start_ms: Number of ms before bout start call to include in data
        :param x_f: The filtered x-coordinates across the experiment
        :param y_f: The filtered y-coordinates across the experiment
        """
        self.bout_curves = []
        self.bout_aspeeds = []
        self.bout_tang_vels = []
        self.fits = []
        ang_speeds = np.array([])
        curvatures = np.array([])
        categories = np.array([])
        relTimes = np.array([])  # relative time within bout: 0-0.5 is before peak speed, 0.5-1 after till end of bout
        for b, categ in zip(self.bouts, self.bout_categories):
            if b[1] < b[0] or b[1] == b[0]:
                # skip odd bout calls
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
            tck, u = spline_fit(xb * self.pixelsize, yb * self.pixelsize)
            a_spd = compute_angSpeed(tck, u, self.datarate)
            curve = compute_curvature(tck, u)
            tangv = compute_tangVelocity(tck, u, self.datarate)
            # strip fit-overhang frames
            start = overhang - int(pre_start_ms / 1000 * self.datarate)
            end = -1 * (overhang - 1)
            peak_frame = int(b[1] - b[0]) + int(pre_start_ms / 1000 * self.datarate)
            a_spd = a_spd[start:end]
            curve = curve[start:end]
            tangv = tangv[start:end]
            ct = np.full_like(curve, categ)
            # create our relative time vector
            rel_time = np.r_[np.linspace(0, 0.5, peak_frame, False), np.linspace(0.5, 1, curve.size - peak_frame)]
            ang_speeds = np.r_[ang_speeds, a_spd]
            curvatures = np.r_[curvatures, curve]
            categories = np.r_[categories, ct]
            relTimes = np.r_[relTimes, rel_time]
            self.bout_curves.append((cut_and_pad(curve, int(130 / 1000 * self.datarate)), categ))
            self.bout_aspeeds.append((cut_and_pad(a_spd, int(130 / 1000 * self.datarate)), categ))
            self.bout_tang_vels.append((cut_and_pad(tangv, int(130 / 1000 * self.datarate)), categ))
        # remove nan-values
        nan_vals = np.logical_or(np.isnan(ang_speeds), np.isnan(curvatures))
        ang_speeds = ang_speeds[np.logical_not(nan_vals)]
        curvatures = curvatures[np.logical_not(nan_vals)]
        categories = categories[np.logical_not(nan_vals)]
        relTimes = relTimes[np.logical_not(nan_vals)]
        # compute linear fits and add to lists
        for categ in self.cat_decode.keys():
            if categ == -1:
                # don't add fit for bouts in exclude category
                continue
            ft = LogLogFit(curvatures, ang_speeds, relTimes, categories == categ, categ, self.filename+'/'+self.key)
            self.fits.append(ft)

    def plot_birdnest(self, plotSplinefit=False, start=0, end=None):
        """
        Makes birdnest plot of experiment trajectory
        :param plotSplinefit: Whether to overlay spline fit onto plot (slow)
        :param start: First frame to include in plot
        :param end: Last frame to include (or None for end)
        :return: figure, axes
        """
        # we don't save trajectory data so reload
        if start < 0 or (end is not None and end < 0):
            raise ValueError("Start and end have to be >= 0")
        xc, yc, h, in_middle = self.load_data()[:4]
        if start >= xc.size:
            raise ValueError("Start is beyond experiment length")
        if end is None:
            end = xc.size
        if end > xc.size:
            raise ValueError("End is beyond experiment length")
        xf, yf = SmoothenTrack(xc, yc, self.filter_window)
        if plotSplinefit:
            tck, u = spline_fit(xf, yf)
            xs, ys = compute_fitCoords(tck, u)
            xs = xs[start:end]
            ys = ys[start:end]
        xf = xf[start:end]
        yf = yf[start:end]
        valid = in_middle[start:end] > 0
        with sns.axes_style("white"):
            fig, ax = pl.subplots()
            ax.plot(xf[valid]*self.pixelsize, yf[valid]*self.pixelsize)
            ax.plot(xf[np.logical_not(valid)]*self.pixelsize, yf[np.logical_not(valid)]*self.pixelsize, 'r')
            if plotSplinefit:
                ax.plot(xs[valid]*self.pixelsize, ys[valid]*self.pixelsize, 'g')
            ax.set_xlabel("Position [mm]")
            ax.set_ylabel("Position [mm]")
            sns.despine(fig, ax)
        return fig, ax

    def plot_boutCalls(self, start=0, end=None):
        """
        Quality control plot of bout calling algorithm and trajectory smoothing
        :param start: First frame to include in plot
        :param end: Last frame to include (or None for end)
        :return: figure, axes tuple
        """
        # we don't save trajectory data so reload
        if start < 0 or (end is not None and end < 0):
            raise ValueError("Start and end have to be >= 0")
        xc, yc = self.load_data()[:2]
        if start >= xc.size:
            raise ValueError("Start is beyond experiment length")
        if end is None:
            end = xc.size
        if end > xc.size:
            raise ValueError("End is beyond experiment length")
        frameTime = np.arange(xc.size) / self.datarate
        xf, yf = SmoothenTrack(xc.copy(), yc.copy(), self.filter_window)
        ispd = ComputeInstantSpeed(xf, yf, self.datarate)
        select = slice(start, end)
        with sns.axes_style("white"):
            fig, (ax_x, ax_y, ax_s) = pl.subplots(nrows=3, sharex=True)
            ax_x.plot(frameTime[select], xc[select] * self.pixelsize, label='Raw')
            ax_x.plot(frameTime[select], xf[select] * self.pixelsize, label='Filtered')
            ax_x.set_ylabel('X position [mm]')
            ax_x.legend()
            ax_x.set_title(self.filename + '/' + self.key)
            sns.despine(ax=ax_x)
            ax_y.plot(frameTime[select], yc[select] * self.pixelsize, label='Raw')
            ax_y.plot(frameTime[select], yf[select] * self.pixelsize, label='Filtered')
            ax_y.set_ylabel('Y position [mm]')
            ax_y.legend()
            sns.despine(ax=ax_y)
            ax_s.plot(frameTime[select], ispd[select] * self.pixelsize)
            bs = self.bouts[:, 0].astype(int)
            bs = bs[bs >= select.start]
            bs = bs[bs < select.stop]
            be = self.bouts[:, 2].astype(int)
            be = be[be >= select.start]
            be = be[be < select.stop]
            ax_s.plot(frameTime[bs], ispd[bs] * self.pixelsize, 'r*')
            ax_s.plot(frameTime[be], ispd[be] * self.pixelsize, 'k*')
            ax_s.set_ylabel('Instant speed [mm/s]')
            ax_s.set_xlabel('Time [s]')
            sns.despine(ax=ax_s)
            fig.tight_layout()
        return fig, (ax_x, ax_y, ax_s)

    def plot_fits(self):
        """
        Plots scatter plot and fit line for each fit
        :return: figure and axes array
        """
        with sns.axes_style('whitegrid'):
            cols = sns.color_palette("deep", len(self.fits))
            fig, axes = pl.subplots(ncols=len(self.fits), sharey=True, sharex=True)
            for i, f in enumerate(sorted(self.fits, key=lambda x: x.category)):
                f.PlotFit(axes[i], color=cols[i])
                if i == 0:
                    axes[i].set_ylabel('log10(Angular speed)')
                axes[i].set_xlabel('log10(Curvature)')
                axes[i].set_title(self.cat_decode[f.category])
                sns.despine(ax=axes[i])
            fig.tight_layout()
        return fig, axes


class AFAP_Experiment(Experiment):
    """
    Describes an IHB Afap experiment
    """

    def __init__(self, key, filename):
        """
        Creates a new AFAP_Experiment
        :param key: The key in the hdf5 dictionary at which this experiments data is stored
        :param filename: The name of the hdf5 file containing this experiment
        """
        super().__init__(key, filename, 700, 1/24.8)
        # override default dictionaries
        self.cdict = {"exclude": -1, "regular": 0, "hunting": 1, "escape": 2}
        self.cat_decode = {v: k for k, v in self.cdict.items()}
        # load data from file and process
        x_c, y_c, heading, inmiddle, looming, escape, hunting = self.load_data()
        x_f, y_f = SmoothenTrack(x_c.copy(), y_c.copy(), 21)
        ispeed = ComputeInstantSpeed(x_f, y_f, self.datarate)
        # detect and store bouts
        self.bouts = self._detect_bouts(ispeed)
        # for each bout compute the distance between start and endpoint as well as the heading change
        bstarts = self.bouts[:, 0].astype(int)
        bends = self.bouts[:, 2].astype(int)
        self.bout_displacements = self.bouts[:, -2]
        self.bout_thetas = np.abs(AssignDeltaAnglesToBouts(self.bouts, heading)[0])
        escape_frames = np.nonzero(escape)[0]
        self.bout_categories = np.zeros(self.bouts.shape[0], dtype=np.int32)
        for i, (bs, be) in enumerate(zip(bstarts, bends)):
            if not inmiddle[bs]:
                self.bout_categories[i] = self.cdict["exclude"]
            else:
                # NOTE: Escapes are identified by a single 1 but this does not align with our bout calls
                if np.min(np.abs(bs - escape_frames)) <= 25:
                    self.bout_categories[i] = self.cdict["escape"]
                elif hunting[bs:be].sum() > 0:
                    self.bout_categories[i] = self.cdict["hunting"]
                else:
                    self.bout_categories[i] = self.cdict["regular"]
        self._compute_fits(300, 30, x_f, y_f)

    def _detect_bouts(self, instantSpeed):
        """
        Bout detection.
        :param instantSpeed: The instant speed trace
        :return: Bouts matrix
        """
        return DetectBouts(instantSpeed, 50, self.datarate, speedThresholdAbsolute=35, maxFramesAtPeak=10)

    def _extract_data(self, data):
        """
        Extracts the raw data of AFAP experiments
        :param data: The raw data array of the Experiment
        :return: x, y, heading, inmiddle, looming, escape, hunting
        """
        x = data[1, :]
        y = data[2, :]
        inmiddle = data[3, :].astype(bool)
        heading = data[4, :]
        looming = data[5, :].astype(bool)
        escape = data[6, :].astype(bool)
        hunting = data[7, :].astype(bool)
        return x, y, heading, inmiddle, looming, escape, hunting

    def plot_boutScatter(self):
        """
        Plots a scatter plot of total bout turn angle versus total bout displacement for
        the different categories
        :return: figure and axis
        """
        with sns.axes_style('whitegrid'):
            fig, ax = pl.subplots()
            cols = sns.color_palette("deep", len(self.cdict) - 1)
            for k in sorted(self.cat_decode.keys()):
                if k == -1:
                    continue
                ax.scatter(self.bout_displacements[self.bout_categories == k] * self.pixelsize,
                           self.bout_thetas[self.bout_categories == k], c=cols[k], s=10, alpha=0.7,
                           label=self.cat_decode[k])
            ax.legend()
            ax.set_xlim(0)
            ax.set_ylim(0, 180)
            sns.despine(fig, ax)
            ax.set_xlabel('Bout displacement [mm]')
            ax.set_ylabel('Bout delta-angle [degrees]')
        return fig, ax

    @staticmethod
    def load_experiments():
        """
        Presents dialog to user to load data files and extract data
        :return: List of AFAP experiments
        """
        fnames = UiGetFile(diagTitle="Load AFAP files")
        exps = []
        for f in fnames:
            exps.append(AFAP_Experiment("martindata", f))
        return exps


class LogLogFit:
    """
    Creates a log(curvature) log(angular velocity) fit and stores the retrieved information
    """
    def __init__(self, cu, av, rt, take, category_flag, name):
        """
        Creates a new LogLogFit object
        :param cu: The curvature trace
        :param av: The angular velocity trace
        :param rt: The relative time within bout
        :param take: Which elements of the cu, av and rt traces to consider
        :param category_flag: The category assigned to this fit
        :param name: Experiment name
        """
        self.category = category_flag
        self.name = name
        cut = cu[take]
        avt = av[take]
        rtt = rt[take]
        keep = np.logical_and(cut > 0, avt > 0)
        self.logCurvature = np.log10(cut[keep])
        self.logAngularSpeed = np.log10(avt[keep])
        self.relativeTime = rtt[keep]
        self.slope, self.intercept, self.rvalue, self.pvalue = linregress(self.logCurvature, self.logAngularSpeed)[0:4]
        self.rsquared = self.rvalue**2

    def PlotFit(self,  ax, color):
        """
        Plots a log-log scatter plot and a line corresponding to the fit
        :param ax: Axis to plot on. Will be created if None
        :param color: Line plot color
        :return: The axis object
        """
        if ax is None:
            fig, ax = pl.subplots()
        xmin = self.logCurvature.min()
        xmax = self.logCurvature.max()
        y1 = xmin*self.slope + self.intercept
        y2 = xmax*self.slope + self.intercept
        cmap = sns.diverging_palette(240, 10, s=75, l=40, center="dark", as_cmap=True)
        ax.scatter(self.logCurvature, self.logAngularSpeed, s=5, alpha=0.3, c=cmap(self.relativeTime))
        ax.plot([xmin, xmax], [y1, y2], c=color, lw=1, ls='--')
        sm = cm.ScalarMappable(cmap=cmap)
        sm._A = []
        pl.colorbar(sm, ax=ax)
        return ax, cmap
