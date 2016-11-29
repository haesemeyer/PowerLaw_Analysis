# -*- coding: utf-8 -*-
"""
Created on 11/16/2016

analysis_wn: Test analysis script for data acquired by Martin Haesemeyer for gaussian white noise experiments
             aggregated in one Hdf5 file
@author: Martin Haesemeyer
"""

import core
import matplotlib.pyplot as pl

if __name__ == '__main__':
    sv = ''
    while sv != 'y' and sv != 'n':
        sv = input('Save figures? [y/n]')
    sv = sv == 'y'

    A = core.Analyzer(core.WN_Experiment, sv)

    # create some overview and quality control plots on every 5th experiment

    for i, e in enumerate(A.experiments):
        if i % 5 == 0:
            e.plot_birdnest()
            e.plot_boutCalls(250*600, 250*620)
            e.plot_boutScatter()
            e.plot_fits()

    # plot main analysis figures
    A.fit_boxplot()
    A.plot_fitDevelopment()
    A.plot_plCHaracteristics()

    pl.show()
