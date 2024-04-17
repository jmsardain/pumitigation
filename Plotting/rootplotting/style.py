# -*- coding: utf-8 -*-

""" Style sheet. 

@file:   style.py
@date:   25 April 2017
@author: Andreas Søgaard 
@email:  andreas.sogaard@cern.ch
"""

import ROOT
from array import array

# Global style variables.
font = 43

fontSizeS = 20 
fontSizeM = 20
fontSizeL = 21

kMyBlue  = 1701;
myBlue   = ROOT.TColor(kMyBlue,   0./255.,  30./255.,  59./255.)
kMyRed   = 1702;
myRed    = ROOT.TColor(kMyRed,  205./255.,   0./255.,  55./255.)
kMyGreen = ROOT.kGreen + 2
kMyLightGreen = ROOT.kGreen - 10

colours       = [ROOT.kViolet + 7, ROOT.kAzure + 7, ROOT.kTeal,     ROOT.kSpring - 2, ROOT.kOrange - 3, ROOT.kPink    ]
colours_light = [ROOT.kViolet - 9, ROOT.kAzure + 6, ROOT.kTeal - 4, ROOT.kSpring - 4, ROOT.kOrange - 2, ROOT.kPink - 4]

# Custom style definition.
AStyle = ROOT.TStyle('AStyle', "AStyle")

# -- Canvas colours
AStyle.SetFrameBorderMode(0)
AStyle.SetFrameFillColor(0)
AStyle.SetCanvasBorderMode(0)
AStyle.SetCanvasColor(0)
AStyle.SetPadBorderMode(0)
AStyle.SetPadColor(0)
AStyle.SetStatColor(0)

# -- Canvas size and margins
AStyle.SetPadRightMargin (0.05)
AStyle.SetPadBottomMargin(0.15)
AStyle.SetPadLeftMargin  (0.15)
AStyle.SetPadTopMargin   (0.06) # 0.05
AStyle.SetTitleOffset(1.2, 'x')
AStyle.SetTitleOffset(2.0, 'y')
AStyle.SetTitleOffset(1.6, 'z')

# -- Fonts
AStyle.SetTextFont(font)

AStyle.SetTextSize(fontSizeS)

for coord in ['x', 'y', 'z']:
    AStyle.SetLabelFont  (font,      coord)
    AStyle.SetTitleFont  (font,      coord)
    AStyle.SetLabelSize  (fontSizeM, coord)
    AStyle.SetTitleSize  (fontSizeM, coord)
    AStyle.SetLabelOffset(0.01, coord)
    pass

AStyle.SetLegendFont(font)
AStyle.SetLegendTextSize(fontSizeS)

# -- Histograms
AStyle.SetMarkerStyle(20)
AStyle.SetMarkerSize(0.9)
AStyle.SetHistLineWidth(2)
AStyle.SetLineStyleString(2,"[12 12]") # postscript dashes

# -- Canvas
AStyle.SetOptTitle(0)
AStyle.SetOptStat(0)
AStyle.SetOptFit(0)

AStyle.SetPadTickX(1)
AStyle.SetPadTickY(1)
AStyle.SetLegendBorderSize(0)

# -- Error bars
#AStyle.SetErrorX(0)

# Colour palette.
def set_palette(name='palette', ncontours=999):
    """Set a color palette from a given RGB list
    stops, red, green and blue should all be lists of the same length
    see set_decent_colors for an example"""

    stops = [0.00, 1.00]
    red   = [0.98,  0./255.]
    green = [0.98, 30./255.]
    blue  = [0.98, 59./255.]

    s = array('d', stops)
    r = array('d', red)
    g = array('d', green)
    b = array('d', blue)

    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)
    return

set_palette()

# Set (and force) style.
# --------------------------------------------------------------------

ROOT.gROOT.SetStyle("AStyle")
ROOT.gROOT.ForceStyle()
