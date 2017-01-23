# -*- coding: utf-8 -*-
"""
Created on 01/23/2017

analysis_mc: Test analysis script for data acquired by Martin Haesemeyer for methyl cellulose
             experiments comparing power law characteristics in fish-water as well as
             low percentage methyl cellulose resampling bout parameters to match swim
             kinematics in the two conditions
@author: Martin Haesemeyer
"""

import core
import matplotlib.pyplot as pl

if __name__ == '__main__':
    sv = ''
    while sv != 'y' and sv != 'n':
        sv = input('Save figures? [y/n]')
    sv = sv == 'y'

    A = core.Analyzer(core.SMCExperiment, sv)

    # create some overview and quality control plots on every 5th experiment

    for i, e in enumerate(A.experiments):
        if i % 5 == 0:
            e.plot_birdnest()
            e.plot_boutCalls(250*600, 250*620)
            e.plot_boutScatter()
            e.plot_fits()

    # instead of plotting aggregate data via analyzer object, resample bouts
    # in order to match peak-speed distribution since bouts are slower
    # overall in methyl cellulose