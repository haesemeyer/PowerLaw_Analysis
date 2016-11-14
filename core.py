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


def UiGetFile(filetypes=[('Matlab file', '.mat'), ('Data file', '.pickle')], diagTitle="Load files"):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    options = {'filetypes': filetypes, 'multiple': True, 'title': diagTitle}
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
                        mag = np.max(spds[allSplitStarts[k]:allSplitEnds[k]])
                        allPeakMags[k] = mag
                        pk = np.nonzero(spds[allSplitStarts[k]:allSplitEnds[k]] == mag)
                        allPeaks[k] = pk[0][0]
                    allPeaks = allPeaks + bStart
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
                    peakframe = np.nonzero(b == b.max())
                    peakframe = peakframe[0][0]
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
    m = np.shape(bouts)[0]
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
