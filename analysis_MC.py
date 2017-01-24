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
import numpy as np

class Resampled:
    """
    Dummy container to use Analyzer's plotting functions
    """
    def __init__(self, analyzer):
        self.fits = []
        self.cat_decode = analyzer.cat_decode
        self.analyzer = analyzer

    def _save_figure(self, filename, figure):
        """
        Save a figure object to file
        :param filename: The filename without extension
        :param figure: The figure object to save
        :return: None
        """
        if self.analyzer.save_plots:
            figure.savefig(filename+'.'+self.analyzer.save_type, type=self.analyzer.save_type)

def fit_resample(analyzer):
    is_fw_valid = analyzer.all_categories == 0
    is_mc_valid = analyzer.all_categories == 1
    pspd_fw = analyzer.all_pSpeeds[is_fw_valid]
    pspd_mc = analyzer.all_pSpeeds[is_mc_valid]
    eid_fw = analyzer.all_eid[is_fw_valid]
    eid_mc = analyzer.all_eid[is_mc_valid]
    bix_fw = analyzer.all_bout_ix[is_fw_valid]
    bix_mc = analyzer.all_bout_ix[is_mc_valid]

    #for each experiment create synthetic resampled experiment with the same
    # number of bouts as the original experiment - for synthetic MC experiments
    # simply resample bouts from methyl-cellulose experiments according to
    # methyl-cellulose experiments
    resampled = Resampled(analyzer)
    for e in analyzer.experiments:
        assert (0 in e.bout_categories) != (1 in e.bout_categories)
        is_fw_exp = 0 in e.bout_categories
        n_bouts = np.sum(np.logical_or(e.bout_categories==0, e.bout_categories==1))
        if is_fw_exp:
            ix_resam = core.emp_resample(pspd_fw, pspd_mc, n_bouts, 15)
            eid = eid_fw[ix_resam]
            bix = bix_fw[ix_resam]
        else:
            ix_resam = core.emp_resample(pspd_mc, pspd_mc, n_bouts, 15)
            eid = eid_mc[ix_resam]
            bix = bix_mc[ix_resam]
        ang_speeds = np.array([])
        curvatures = np.array([])
        categories = np.array([])
        relTimes = np.array([]) 
        # loop over experiments and individual bouts that were selected
        for ei in np.unique(eid):
            for bi in bix[eid == ei]:
                if bi in analyzer.experiments[ei].raw_curves:
                    # this bout index has valid parameters in this experiment
                    cu = analyzer.experiments[ei].raw_curves[bi]
                    av = analyzer.experiments[ei].raw_aspeeds[bi]
                    rt = analyzer.experiments[ei].raw_relTimes[bi]
                    c = analyzer.experiments[ei].raw_categories[bi]
                    ang_speeds = np.r_[ang_speeds, av]
                    curvatures = np.r_[curvatures, cu]
                    categories = np.r_[categories, c]
                    relTimes = np.r_[relTimes, rt]
        ft = core.LogLogFit(curvatures, ang_speeds, relTimes, categories != is_fw_exp,
                int(is_fw_exp), e.filename+'/resampled')
        resampled.fits.append(ft)
    return resampled


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
    resampled = fit_resample(A)
    core.Analyzer.fit_boxplot(resampled)