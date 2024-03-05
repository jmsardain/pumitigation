from ROOT import TFile, TH1D, TH2D, TH2F, TH1F, TCanvas, TGraph, TMultiGraph, TLegend, TColor, TLatex
from root_numpy import fill_hist  as fh
from root_numpy import array2hist as a2h
from root_numpy import hist2array as h2a
from root_numpy import tree2array
from rootplotting import ap
from rootplotting.tools import *
ROOT.gStyle.SetPalette(ROOT.kBird)
import pandas as pd
import uproot as ur


file = ROOT.TFile.Open('/home/jmsardain/JetCalib/PUMitigation/final/plotting/output.root')

   fileDataDisabled = ROOT.TFile.Open("/data/jmsardain/LJP/Plots/Tile/2DPlots/Slices2D_DeltaR_kt_binned_pt_"+str(i)+"_Disabled Tile.root")
    hData      = fileDataRun2.Get("Data full Run2")
    hDisabled  = fileDataDisabled.Get("Disabled Tile")



    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]
    c = ap.canvas(num_pads=1, batch=True)


    # h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    ratio   = hData.Clone()
    for ibinx in range(1, ratio.GetNbinsX()+1):
        for ibiny in range(1, ratio.GetNbinsY()+1):
            ratio.SetBinContent(ibinx, ibiny, 0)

    for ibinx in range(1, ratio.GetNbinsX()+1):
        for ibiny in range(1, ratio.GetNbinsY()+1):

            val = 0
            if  hDisabled.GetBinContent(ibinx, ibiny) != 0:
                val = hData.GetBinContent(ibinx, ibiny) / hDisabled.GetBinContent(ibinx, ibiny)
                print(val)
            else:
                val = 0
            ratio.SetBinContent(ibinx, ibiny, val)
    ratio.GetZaxis().SetRangeUser(0.8, 1.2)

    c.hist2d(ratio, option='AXIS')
    c.hist2d(ratio, option='COLZ')
    # c.hist2d(ratio, option='AXIS')
    c.save("./2DPlots/ratio_data_dead_pt_"+str(i)+".png")
~
~
~
~
