# -*- coding: utf-8 -*-
"""
Created on 11/14/2016

analysis_IHB: Test analysis script for data acquired by Isaac H Bianco and saved in MATLAB v7.3 files
@author: Martin Haesemeyer
"""


import core
import matplotlib.pyplot as pl

if __name__ == '__main__':
    sv = ''
    while sv != 'y' and sv != 'n':
        sv = input('Save figures? [y/n]')
    sv = sv == 'y'

    A = core.Analyzer(core.AFAP_Experiment, sv)

    # create some overview and quality control plots on every 5th experiment

    for i, e in enumerate(A.experiments):
        if i % 5 == 0:
            e.plot_birdnest()
            e.plot_boutCalls(700*60, 700*80)
            e.plot_boutScatter()
            e.plot_fits()

    # plot main analysis figures
    A.fit_boxplot()
    A.plot_fitDevelopment()
    A.plot_plCHaracteristics()

    pl.show()

