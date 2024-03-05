#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example script, showing the rootplotting interface. 

@file:   example.py
@date:   25 April 2017
@author: Andreas Søgaard 
@email:  andreas.sogaard@cern.ch
"""

# Basic
import sys, glob

# Scientific import(s)
import ROOT
import numpy as np

# Local import(s)
from tools import *
import ap


# Main function
def main ():
    

    # Setup.
    # – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – 

    # Load data
    files  = glob.glob('data/objdef_MC_3610*.root')  + glob.glob('data/objdef_MC_3054*.root') 

    if len(files) == 0:
        warning("No files found. Try to run:")
        warning(" $ source getSomeData.sh")
        return

    tree   = 'BoostedJet+ISRgamma/Nominal/EventSelection/Pass/NumLargeRadiusJets/Postcut'
    prefix = 'Jet_'

    data = loadData(files, tree)
    info = loadData(files, 'BoostedJet+ISRgamma/Nominal/outputTree', stop=1)

    # Rename variables
    data.dtype.names = [name.replace(prefix, '') for name in data.dtype.names]

    # Scaling by cross section
    xsec = loadXsec('sampleInfo.csv')
    lumi = 36.1

    # Append new DSID field
    data = append_fields(data, 'DSID', np.zeros((data.size,)), dtypes=int)
    for idx in info['id']:    
        msk = (data['id'] == idx) # Get mask of all 'data' entries with same id, i.e. from same file
        DSID = info['DSID'][idx]  # Get DSID for this file
        data['weight'][msk] *= xsec[DSID] # Scale by cross section x filter eff. for this DSID
        data['DSID']  [msk] = DSID # Store DSID
        pass
    data['weight'] *= lumi # Scale all events (MC) by luminosity
    
    msk_incl = (data['DSID'] >= 361039) & (data['DSID'] <= 361062)
    msk_W    = (data['DSID'] >= 305435) & (data['DSID'] <= 305439)
    msk_Z    = (data['DSID'] >= 305440) & (data['DSID'] <= 305445)

    
    # Plotting
    # – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – – 

    bins = np.linspace(50, 300, 25 + 1, endpoint=True)
    
    # Setup canvas
    c = ap.canvas(num_pads=2)
    p0, p1 = c.pads()

    # Add stacked backgrounds
    h_W    = c.stack(data['m'][msk_W],    bins=bins, weights=data['weight'][msk_W],    fillcolor=ROOT.kAzure + 2, label='W(qq) + #gamma', scale=1)
    h_Z    = c.stack(data['m'][msk_Z],    bins=bins, weights=data['weight'][msk_Z],    fillcolor=ROOT.kAzure + 3, label='Z(qq) + #gamma', scale=1)
    h_bkg  = c.stack(data['m'][msk_incl], bins=bins, weights=data['weight'][msk_incl], fillcolor=ROOT.kAzure + 7, label='Incl. #gamma')
    
    # Draw stats. error of stacked sum
    h_sum  = c.getStackSum()
    c.hist(h_sum, fillstyle=3245, fillcolor=ROOT.kGray+2, linecolor=ROOT.kGray + 3, label='Stats. uncert.', option='E2')
    
    # Add (pseudo-) data
    np.random.seed(21)
    h_data = c.plot (data['m'], bins=bins, weights=data['weight'] * (1. + np.random.randn(len(data['m'])) * 1.), markersize=0.8, label='Data', scale=1)

    # Draw error- and ratio plots
    h_err   = c.ratio_plot((h_sum,  h_sum), option='E2')
    h_ratio = c.ratio_plot((h_data, h_sum), markersize=0.8)
    
    # Overlay
    o = ap.overlay(c)
    o.graph(np.square(bins), bins=bins, linecolor=ROOT.kRed, linewidth=2, option='L')
    o.label('Overlay plot')

    # Add labels and text
    c.xlabel('Signal jet mass [GeV]')
    c.ylabel('Events')
    p1.ylabel('Data / Bkg.')
    c.text(["#sqrt{s} = 13 TeV,  L = 36.1 fb^{-1}",
            "Trimmed anti-k_{t}^{R=1.0} jets"], 
           qualifier='Simulation Internal')

    # Add line(s) and limit(s)
    p1.ylines([0.9, 1.0, 1.1])
    p1.ylim(0.8, 1.2)

    # Configure axis padding and -scale
    c.padding(0.35)
    c.log(True)

    # Draw legend
    c.legend()

    # Save and show plot
    c.save('test.pdf')
    c.show()
    return


# Main function call.
if __name__ == '__main__':
    main()
    pass
