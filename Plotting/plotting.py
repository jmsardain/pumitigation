import ROOT
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mode, iqr
import math
import argparse
from rootplotting import ap
from rootplotting.tools import *
from root_numpy import fill_hist
from array import array

def BinLogX(h):
    axis = h.GetXaxis()
    bins = axis.GetNbins()

    a = axis.GetXmin()
    b = axis.GetXmax()
    width = (b-a) / bins
    newbins = np.zeros([bins + 1])
    for i in range(bins+1):
        newbins[i] = pow(10, a + i * width)

    axis.Set(bins, newbins)
    del newbins
    pass

def BinLogY(h):
    axis = h.GetYaxis()
    bins = h.GetNbinsY()

    a = axis.GetXmin()
    b = axis.GetXmax()
    width = (b-a) / bins
    newbins1 = np.zeros([bins + 1])
    for i in range(bins+1):
        newbins1[i] = pow(10, a + i * width)

    axis.Set(bins, newbins1)
    del newbins1
    pass

def HistClassification(score_signal, score_pu):
    colours = [ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026')]
    c = ap.canvas(num_pads=2, batch=True)
    p0, p1 = c.pads()
    xaxis = np.linspace(0, 1, 100 + 1, endpoint=True)
    hSignal  = c.hist(score_signal,  bins=xaxis, option='HIST', label='Signal',  linecolor=colours[0])
    hPU      = c.hist(score_pu,      bins=xaxis, option='HIST', label='PU',      linecolor=colours[1])

    # p1.ylim(0., 2.)
    c.ratio_plot((hSignal,  hSignal),  option='E2',      linecolor=colours[0]) #, oob=True)
    c.ratio_plot((hSignal,   hPU),  option='HIST',    linecolor=colours[1]) #, oob=True)
    p1.yline(1.0)

    c.xlabel('Scores')
    c.ylabel('Number of clusters')
    c.legend(xmin=0.7, xmax=0.9)
    c.log()
    c.text(["#sqrt{s} = 13 TeV, Pythia 8, dijet",
            "anti-k_{T} R = 0.4 EMTopo jets",
            "p_{T}^{JES} > 20 GeV, |y^{JES} < 2|",
            ], qualifier='Simulation Preliminary')

    return c

def Hist1D_Pt(jetarea, jetraw, jetdnn, jetpu, truthjet, raw_signal_jet):

    #colours = [ROOT.kAzure + 7, ROOT.kOrange-3, ROOT.kYellow+3, ROOT.kBlue]
    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]
    c = ap.canvas(num_pads=2, batch=True)
    p0, p1 = c.pads()
    xaxis = np.logspace(np.log10(1), np.log10(500), 100 + 1, endpoint=True)

    hJetArea  = c.hist(jetraw,  bins=xaxis, option='HIST', label='E_{jet}^{area}',  linecolor=colours[0])
    hJetRaw   = c.hist(jetarea, bins=xaxis, option='HIST', label='E_{jet}^{const}', linecolor=colours[1])
    hJetDNN   = c.hist(jetdnn,  bins=xaxis, option='HIST', label='E_{jet}^{DNN}',   linecolor=colours[2])
    hJetPU    = c.hist(jetpu,   bins=xaxis, option='HIST', label='E_{jet}^{PU}',    linecolor=colours[3])
    # hJetTrue  = c.hist(truthjet,bins=xaxis, option='HIST', label='E_{jet}^{true}',    linecolor=colours[4])
    hJetSig   = c.hist(raw_signal_jet,bins=xaxis, option='HIST', label='#Sigma E_{clus}^{label=1}',    linecolor=colours[4])
    p1.ylim(0., 2.)
    c.ratio_plot((hJetArea,  hJetArea),  option='E2',      linecolor=colours[0]) #, oob=True)
    c.ratio_plot((hJetRaw,   hJetArea),  option='HIST',    linecolor=colours[1]) #, oob=True)
    c.ratio_plot((hJetDNN,   hJetArea),  option='HIST',    linecolor=colours[2]) #, oob=True)
    c.ratio_plot((hJetPU,    hJetArea),  option='HIST',    linecolor=colours[3]) #, oob=True)
    # c.ratio_plot((hJetTrue,  hJetArea),  option='HIST',    linecolor=colours[4]) #, oob=True)
    c.ratio_plot((hJetSig,  hJetArea),  option='HIST',    linecolor=colours[4]) #, oob=True)

    p1.yline(1.0)
    p1.logx()
    c.xlabel('p_{T}^{true jet} [GeV]')
    c.ylabel('Number of jets')
    p1.ylabel('Ratio to E_{jet}^{area}')

    c.legend(xmin=0.7, xmax=0.9)
    c.log()
    c.logx()
    c.text(["#sqrt{s} = 13 TeV, Pythia 8, dijet",
            "anti-k_{T} R = 0.4 EMTopo jets",
            "p_{T}^{JES} > 20 GeV, |y^{JES} < 2|",
            ], qualifier='Simulation Preliminary')

    return c

def Hist1D(jet_values, sum_cluster_values):
    colours = [ROOT.kAzure + 7, ROOT.kOrange-3, ROOT.kYellow+3, ROOT.kBlue]

    c = ap.canvas(num_pads=2, batch=True)
    p0, p1 = c.pads()
    xaxis = np.logspace(np.log10(1), np.log10(1000), 100 + 1, endpoint=True)

    hJetRawE = c.hist(jet_values, bins=xaxis, option='HIST', label='E_{jet}^{const}',     linecolor=colours[0])
    hSumClusE  = c.hist(sum_cluster_values,  bins=xaxis, option='HIST', label='#Sigma E_{clus}^{EM}',  linecolor=colours[1])
    p1.ylim(0., 2.)
    c.ratio_plot((hJetRawE,  hJetRawE),  option='E2',      linecolor=colours[0]) #, oob=True)
    c.ratio_plot((hSumClusE,   hJetRawE),  option='HIST',    linecolor=colours[1]) #, oob=True)

    p1.yline(1.0)
    p1.logx()
    c.xlabel('Energy [GeV]')
    c.ylabel('Number of jets')
    p1.ylabel('Sum / Jet')

    c.legend(xmin=0.7, xmax=0.9)
    c.log()
    c.logx()
    c.text(["#sqrt{s} = 13 TeV, Pythia 8, dijet",
            "anti-k_{T} R = 0.4 EMTopo jets",
            "p_{T}^{JES} > 20 GeV, |y^{JES} < 2|",
            ], qualifier='Simulation Preliminary')

    return c

def Plots2D(valueX, ratio_num, ratio_den, xlabel, ylabel):
    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()

    xaxis = np.logspace(np.log10(1), np.log10(500), 100 + 1, endpoint=True)
    yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

    # h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1]])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    h1          = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    ratio = [x / y for x, y in zip(ratio_num, ratio_den)]
    mesh = np.vstack((valueX, ratio)).T
    fill_hist(h1, mesh)

    c.hist2d(h1_backdrop, option='AXIS')
    c.hist2d(h1,          option='COLZ')
    c.hist2d(h1_backdrop, option='AXIS')

    c.logx()
    c.xlabel(xlabel)
    c.ylabel(ylabel)
    c.text(["#sqrt{s} = 13 TeV, Pythia 8, dijet",
            "anti-k_{T} R = 0.4 EMTopo jets",
            "p_{T}^{JES} > 20 GeV, |y^{JES} < 2|",
            ], qualifier='Simulation Preliminary')


    return c

def Profile2DPlots(truthJetPt, JetPUE, JetAreaE, JetTrue, weighted_score, xlabel, ylabel):


    xaxis = np.logspace(np.log10(1), np.log10(500), 100 + 1, endpoint=True)
    yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

    NBINS_PT = 49
    JetPt_bins = array('d', xaxis)

    # h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], 0.75* yaxis[-1] ])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    h1_backdrop = ROOT.TH2F('', "", 1, np.array([xaxis[0], xaxis[-1]]), 1, np.array([yaxis[0], yaxis[-1]])) # + 0.55 * (yaxis[-1] - yaxis[0])]))
    h = ROOT.TProfile2D("", "", len(JetPt_bins)-1, JetPt_bins, 100, 0, 2)
    h1 = ROOT.TProfile2D("", "", len(JetPt_bins)-1, JetPt_bins, 100, 0, 2)
    ratio = [x / y for x, y in zip(JetPUE, JetAreaE)]
    ratio1 = [x / y for x, y in zip(JetPUE, JetTrue)]
    # mesh = np.vstack((truthJetPt, ratio)).T
    # fill_hist(h, mesh)

    for pt, div, score in zip(truthJetPt, ratio, weighted_score):
        h.Fill(pt, div, score)
    for pt, div, score in zip(truthJetPt, ratio1, weighted_score):
        h1.Fill(pt, div, score)
        # print(pt)
        # print(div)
        # print(score)

    c = ROOT.TCanvas("", "", 500, 500)
    c.SetRightMargin(0.2)
    c.SetLogx()
    h.GetYaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetTitle('p_{T}^{true jet} [GeV]')
    h.GetYaxis().SetTitle('E_{jet}^{PU} / E_{jet}^{area}')
    h.GetZaxis().SetTitle('Jet score')
    h1_backdrop.Draw("axis")
    h.Draw("colz")
    # c.SetLogz()
    c.SaveAs("./plots/Profile2DPlots_GNN_Area_truthPt.png")

    c1 = ROOT.TCanvas("", "", 500, 500)
    c1.SetRightMargin(0.2)
    c1.SetLogx()
    h1.GetYaxis().SetTitleOffset(1.2)
    h1.GetXaxis().SetTitle('p_{T}^{true jet} [GeV]')
    h1.GetYaxis().SetTitle('E_{jet}^{PU} / E_{jet}^{true}')
    h1.GetZaxis().SetTitle('Jet score')
    h1_backdrop.Draw("axis")
    h1.Draw("colz")
    # c1.SetLogz()
    c1.SaveAs("./plots/Profile2DPlots_GNN_truthE_truthPt.png")


def GetMedianAndIQR(truthJetPt, JetTruthE, JetDNNE, JetPUE, denominator, label):

    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]

    ratio_SigE_Area = [x / y for x, y in zip(JetTruthE, denominator)]
    ratio_DNN_Area = [x / y for x, y in zip(JetDNNE, denominator)]
    ratio_PU_Area  = [x / y for x, y in zip(JetPUE, denominator)]

    c = ap.canvas(batch=True, size=(600,600))
    # c.pads()[0]._bare().SetRightMargin(0.2)

    c1 = ap.canvas(batch=True, size=(600,600))
    # c1.pads()[0]._bare().SetRightMargin(0.2)

    ## define axes
    xaxis = np.logspace(np.log10(1), np.log10(500), 100 + 1, endpoint=True)
    yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

    ## create 2D histos
    hratio_Sig_Area_2D = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    hratio_DNN_Area_2D = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    hratio_PU_Area_2D = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    ## create 1D histo that will have the final results
    hratio_Sig_Area_median = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    hratio_DNN_Area_median = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    hratio_PU_Area_median  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    hratio_Sig_Area_iqr = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    hratio_DNN_Area_iqr = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    hratio_PU_Area_iqr  = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    ## fill histos
    mesh1a = np.vstack((truthJetPt, ratio_DNN_Area)).T
    mesh1b = np.vstack((truthJetPt, ratio_PU_Area)).T
    mesh1c = np.vstack((truthJetPt, ratio_SigE_Area)).T
    fill_hist(hratio_DNN_Area_2D, mesh1a)
    fill_hist(hratio_PU_Area_2D, mesh1b)
    fill_hist(hratio_Sig_Area_2D, mesh1c)

    for ibinx in range(1, hratio_DNN_Area_2D.GetNbinsX()+1):
        median_binX = []
        for ibiny in range(1, hratio_DNN_Area_2D.GetNbinsY()+1):
            n = int(hratio_DNN_Area_2D.GetBinContent(ibinx, ibiny))
            for _ in range(n):
                median_binX.append(hratio_DNN_Area_2D.GetYaxis().GetBinCenter(ibiny))
                pass
        if not median_binX:
            continue
        calcMedian =  np.median(median_binX)
        calcIQR =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
        hratio_DNN_Area_median.SetBinContent(ibinx, calcMedian)
        hratio_DNN_Area_median.SetBinError(ibinx, 0)
        hratio_DNN_Area_iqr.SetBinContent(ibinx, calcIQR)
        hratio_DNN_Area_iqr.SetBinError(ibinx, 0)
    #
    for ibinx in range(1, hratio_PU_Area_2D.GetNbinsX()+1):
        median_binX = []
        for ibiny in range(1, hratio_PU_Area_2D.GetNbinsY()+1):
            n = int(hratio_PU_Area_2D.GetBinContent(ibinx, ibiny))
            for _ in range(n):
                median_binX.append(hratio_PU_Area_2D.GetYaxis().GetBinCenter(ibiny))
                pass
        if not median_binX:
            continue
        calcMedian =  np.median(median_binX)
        calcIQR =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
        hratio_PU_Area_median.SetBinContent(ibinx, calcMedian)
        hratio_PU_Area_median.SetBinError(ibinx, 0)
        hratio_PU_Area_iqr.SetBinContent(ibinx, calcIQR)
        hratio_PU_Area_iqr.SetBinError(ibinx, 0)

    for ibinx in range(1, hratio_Sig_Area_2D.GetNbinsX()+1):
        median_binX = []
        for ibiny in range(1, hratio_Sig_Area_2D.GetNbinsY()+1):
            n = int(hratio_Sig_Area_2D.GetBinContent(ibinx, ibiny))
            for _ in range(n):
                median_binX.append(hratio_Sig_Area_2D.GetYaxis().GetBinCenter(ibiny))
                pass
        if not median_binX:
            continue
        calcMedian =  np.median(median_binX)
        calcIQR =  iqr(median_binX, rng=(16, 84)) / (2 * np.median(median_binX))
        hratio_Sig_Area_median.SetBinContent(ibinx, calcMedian)
        hratio_Sig_Area_median.SetBinError(ibinx, 0)
        hratio_Sig_Area_iqr.SetBinContent(ibinx, calcIQR)
        hratio_Sig_Area_iqr.SetBinError(ibinx, 0)

    c.hist(hratio_DNN_Area_median, markercolor=colours[0], linecolor=colours[0], label="E_{jet}^{DNN} / E_{jet}^{"+label+"}")
    c.hist(hratio_PU_Area_median,  markercolor=colours[1], linecolor=colours[1], label="E_{jet}^{PU} / E_{jet}^{"+label+"}")
    c.hist(hratio_Sig_Area_median,  markercolor=colours[2], linecolor=colours[2], label="#Sigma E_{clus}^{label=1} / E_{jet}^{"+label+"}")
    c.logx()
    c.legend(xmin=0.7, xmax=0.9)
    c.xlabel("p_{T}^{true jet} [GeV]")
    c.ylabel("Jet energy response, R_{E}")
    c.text(["#sqrt{s} = 13 TeV, Pythia 8, dijet",
            "anti-k_{T} R = 0.4 EMTopo jets",
            "p_{T}^{JES} > 20 GeV, |y^{JES} < 2|",
            ], qualifier='Simulation Preliminary')

    c1.hist(hratio_DNN_Area_iqr, markercolor=colours[0], linecolor=colours[0], label="E_{jet}^{DNN} / E_{jet}^{"+label+"}")
    c1.hist(hratio_PU_Area_iqr,  markercolor=colours[1], linecolor=colours[1], label="E_{jet}^{PU} / E_{jet}^{"+label+"}")
    c1.hist(hratio_Sig_Area_iqr,  markercolor=colours[2], linecolor=colours[2], label="#Sigma E_{clus}^{label=1} / E_{jet}^{"+label+"}")
    c1.logx()
    c1.legend(xmin=0.7, xmax=0.9)
    c1.xlabel("p_{T}^{true jet} [GeV]")
    c1.ylabel("Jet energy resolution, #sigma(r_{E})")
    c1.text(["#sqrt{s} = 13 TeV, Pythia 8, dijet",
            "anti-k_{T} R = 0.4 EMTopo jets",
            "p_{T}^{JES} > 20 GeV, |y^{JES} < 2|",
            ], qualifier='Simulation Preliminary')
    return c, c1

def PlotFracEnergy_Score(h2D_Inc, h2D_Sig, h2D_PU):

    # c1 = ap.canvas(batch=True, size=(600,600))
    # c1.pads()[0]._bare().SetRightMargin(0.2)
    # c1.pads()[0]._bare().SetLogz()
    #
    # c2 = ap.canvas(batch=True, size=(600,600))
    # c2.pads()[0]._bare().SetRightMargin(0.2)
    # c2.pads()[0]._bare().SetLogz()
    #
    # c3 = ap.canvas(batch=True, size=(600,600))
    # c3.pads()[0]._bare().SetRightMargin(0.2)
    # c3.pads()[0]._bare().SetLogz()

    h1_backdrop = ROOT.TH2F('', "", 1, np.array([0, 1]), 1, np.array([0, 1])) # + 0.55 * (yaxis[-1] - yaxis[0])]))

    c = ROOT.TCanvas("", "", 500, 500)
    c.SetRightMargin(0.2)
    h2D_Inc.GetYaxis().SetTitleOffset(1.2)
    h2D_Inc.GetXaxis().SetTitle('Score')
    h2D_Inc.GetYaxis().SetTitle('Fraction of energy')
    h1_backdrop.Draw("axis")
    h2D_Inc.Draw("colz")
    c.SaveAs("./plots/EnergyFraction_vs_Score_inclusive.png")

    c1 = ROOT.TCanvas("", "", 500, 500)
    c1.SetRightMargin(0.2)
    h2D_Sig.GetYaxis().SetTitleOffset(1.2)
    h2D_Sig.GetXaxis().SetTitle('Score')
    h2D_Sig.GetYaxis().SetTitle('Fraction of energy')
    h1_backdrop.Draw("axis")
    h2D_Sig.Draw("colz")
    c1.SaveAs("./plots/EnergyFraction_vs_Score_signal.png")

    c2 = ROOT.TCanvas("", "", 500, 500)
    c2.SetRightMargin(0.2)
    h2D_PU.GetYaxis().SetTitleOffset(1.2)
    h2D_PU.GetXaxis().SetTitle('Score')
    h2D_PU.GetYaxis().SetTitle('Fraction of energy')
    h1_backdrop.Draw("axis")
    h2D_PU.Draw("colz")
    c2.SaveAs("./plots/EnergyFraction_vs_Score_pu.png")

def main():
    ROOT.gStyle.SetPalette(ROOT.kBird)

    df = pd.read_csv('/home/jmsardain/JetCalib/PUMitigation/final/ckpts/GATNet/out_GATNet_.csv')
    grouped = df.groupby(by=['eventNumber', 'jetCnt'])

    total_jetRawE = []
    total_clusterE = []

    total_JetAreaE = []
    total_JetRawE = []
    total_JetDNNE = []
    total_JetPUE = []
    total_JetTruthE = []
    total_JetRawE_signal = []

    total_JetAreaPt = []
    total_JetRawPt = []
    total_JetDNNPt = []
    total_JetPUPt = []
    total_JetTruthPt = []
    total_JetRawPt_signal = []

    total_score_signal = []
    total_score_pu     = []

    total_weighted_score = []

    ## fraction of energy
    total_fracE_inclusive = []
    total_fracE_signal    = []
    total_fracE_pu        = []
    total_score_inclusive = []
    total_score_signal    = []
    total_score_pu        = []


    h2D_Inc = ROOT.TH2D("", "", 100, 0, 1, 100, 0, 1)
    h2D_Sig = ROOT.TH2D("", "", 100, 0, 1, 100, 0, 1)
    h2D_PU = ROOT.TH2D("", "", 100, 0, 1, 100, 0, 1)

    for (eventNumber, jetCnt), group_df in grouped:
        # print(eventNumber)
        # if eventNumber!=46000020.0 : continue
        clusterPt_list = []
        clusterEta_list = []
        clusterPhi_list = []
        clusterE_list = []
        clusterEDNN_list = []
        scores_list = []
        label_list = []
        jetRawE_list = []
        tlv_Raw_list = []
        tlv_DNN_list = []
        tlv_Recalc_list = []
        jetAreaE_list = []
        jetAreaPt_list = []
        jetTruePt_list = []
        jetTrueE_list = []
        tlv_signal_raw_list = []

        # fracE_inclusive = []
        # fracE_signal    = []
        # fracE_pu        = []
        # score_inclusive = []
        # score_signal = []
        # score_pu = []

        hInc = ROOT.TH1D ("", "", 100, 0, 1)
        hSig = ROOT.TH1D ("", "", 100, 0, 1)
        hPU = ROOT.TH1D ("", "", 100, 0, 1)
        for index, row in group_df.iterrows():
            clusterPt_list.append(row['clusterPt'])
            clusterEta_list.append(row['clusterEta'])
            clusterPhi_list.append(row['clusterPhi'])
            clusterE_list.append(row['clusterE'])
            clusterEDNN_list.append(row['clusterEDNN'])
            scores_list.append(row['score'])
            label_list.append(row['labels'])
            jetRawE_list.append(row['jetRawE'])
            jetAreaE_list.append(row['jetAreaE'])
            jetAreaPt_list.append(row['jetAreaPt'])
            jetTruePt_list.append(row['truthJetPt'])
            jetTrueE_list.append(row['truthJetE'])

            if row['labels'] == 1:
                total_score_signal.append(row['score'])
            if row['labels'] == 0:
                total_score_pu.append(row['score'])



        clusterE_raw_sum = np.sum(clusterE_list)
        clusterEDNN_sum = np.sum(clusterEDNN_list)
        # print(clusterEDNN_sum)
        for clusterPt, clusterEta, clusterPhi, clusterE, clusterEDNN, score, label, rawjetE in zip(clusterPt_list, clusterEta_list, clusterPhi_list, clusterE_list, clusterEDNN_list, scores_list, label_list, jetRawE_list):
            ## for signal clusters
            tlv_signal_raw = ROOT.TLorentzVector()
            tlv_signal_truth = ROOT.TLorentzVector()
            if label == 1:
                tlv_signal_raw.SetPtEtaPhiE(clusterPt, clusterEta, clusterPhi, clusterE)
                tlv_signal_raw_list.append(tlv_signal_raw)

            ## for all clusters
            ## Get constituent level
            tlv = ROOT.TLorentzVector()
            tlv.SetPtEtaPhiE(clusterPt, clusterEta, clusterPhi, clusterE)
            tlv_Raw_list.append(tlv)
            ## Get calibrated without pumitigation level
            tlv_dnn = ROOT.TLorentzVector()
            tlv_dnn.SetPtEtaPhiE(clusterPt * (clusterEDNN / clusterE), clusterEta, clusterPhi, clusterEDNN)
            tlv_DNN_list.append(tlv_dnn)
            ## Get calibrated with pumitigation level
            # newScore = score
            newScore = 1 / (1 + np.exp(-10 * (score - 0.5)))
            # newScore = 1 / (1 + np.exp(-20 * (score - 0.8)))
            tlv_dnn_pu = ROOT.TLorentzVector()
            tlv_dnn_pu.SetPtEtaPhiE(clusterPt * (clusterEDNN *  newScore / clusterE), clusterEta, clusterPhi, clusterE * newScore)
            tlv_Recalc_list.append(tlv_dnn_pu)

            ## frac energy
            hInc.Fill(score, clusterEDNN / rawjetE)
            if label == 1:
                hSig.Fill(score, clusterEDNN / rawjetE)
            if label == 0:
                hPU.Fill(score, clusterEDNN / rawjetE)


        # hInc.Scale(1/hInc.Integral())
        # hSig.Scale(1/hInc.Integral())
        # hPU.Scale(1/hInc.Integral())

        for i in range(1, hInc.GetNbinsX()+1):
            # print(hInc.GetBinCenter(i))
            if hInc.GetBinContent(i) > 0:
                h2D_Inc.Fill(hInc.GetBinCenter(i), hInc.GetBinContent(i))
                # print("inclusive {}".format(hInc.GetBinContent(i)))
            if hSig.GetBinContent(i) > 0:
                h2D_Sig.Fill(hInc.GetBinCenter(i), hSig.GetBinContent(i))
                # print("signal {}".format(hInc.GetBinContent(i)))
            if hPU.GetBinContent(i) > 0:
                h2D_PU.Fill(hInc.GetBinCenter(i), hPU.GetBinContent(i))
                # print(hInc.GetBinContent(i))



        #
        # total_fracE_inclusive.append(fracE_inclusive)
        # total_fracE_signal.append(fracE_signal)
        # total_fracE_pu .append(fracE_pu)
        # total_score_inclusive.append(score_inclusive)
        # total_score_signal.append(score_signal)
        # total_score_pu.append(score_pu)

        jetRawE_value = np.sum(jetRawE_list) / len(jetRawE_list)
        jetAreaE_value = np.sum(jetAreaE_list) / len(jetAreaE_list)
        jetAreaPt_value = np.sum(jetAreaPt_list) / len(jetAreaPt_list)
        jetTruePt_value = np.sum(jetTruePt_list) / len(jetTruePt_list)
        jetTrueE_value = np.sum(jetTrueE_list) / len(jetTrueE_list)

        total_jetRawE.append(jetRawE_value)
        total_clusterE.append(clusterE_raw_sum)
        weighted_scores_values_temp = [x * y for x, y in zip(clusterE_list, scores_list)]
        weighted_scores_values = [x / y for x, y in zip(weighted_scores_values_temp, jetRawE_list)]
        total_weighted_score.append(sum(weighted_scores_values))

        total_JetAreaE.append(jetAreaE_value)
        total_JetRawE_signal.append(sum(tlv.E() for tlv in tlv_signal_raw_list))
        total_JetRawE.append(sum(tlv.E() for tlv in tlv_Raw_list))
        total_JetDNNE.append(sum(tlv.E() for tlv in tlv_DNN_list))
        total_JetPUE.append(sum(tlv.E() for tlv in tlv_Recalc_list))
        total_JetTruthE.append(jetTrueE_value)

        total_JetAreaPt.append(jetAreaPt_value)
        total_JetRawPt_signal.append(sum(tlv.Pt() for tlv in tlv_signal_raw_list))
        total_JetRawPt.append(sum(tlv.Pt() for tlv in tlv_Raw_list))
        total_JetDNNPt.append(sum(tlv.Pt() for tlv in tlv_DNN_list))
        total_JetPUPt.append(sum(tlv.Pt() for tlv in tlv_Recalc_list))
        total_JetTruthPt.append(jetTruePt_value)

    # total_fracE_inclusive = np.array(total_fracE_inclusive).flatten()
    # total_fracE_signal = np.array(total_fracE_signal).flatten()
    # total_fracE_pu = np.array(total_fracE_pu).flatten()
    # total_score_inclusive = np.array(total_score_inclusive).flatten()
    # total_score_signal = np.array(total_score_signal).flatten()
    # total_score_pu = np.array(total_score_pu).flatten()

    c = HistClassification(total_score_signal, total_score_pu)
    c.save("./plots/Classification.png")

    c = Hist1D(total_jetRawE, total_clusterE)
    c.save("./plots/Comparison_SumClus_JetRawE.png")

    c = Hist1D_Pt(total_JetAreaPt, total_JetRawPt, total_JetDNNPt, total_JetPUPt, total_JetTruthPt, total_JetRawPt_signal)
    c.save("./plots/Comparison_Pt.png")

    ## energy fraction versus score
    ## inclusive
    PlotFracEnergy_Score(h2D_Inc, h2D_Sig, h2D_PU)


    ## division by TruthJetE
    c = Plots2D(total_JetTruthPt, total_JetAreaE, total_JetTruthE, xlabel='p_{T}^{true} [GeV]', ylabel='E_{jet}^{area} / E_{jet}^{true}')
    c.save("./plots/2DPlots_Area_truthE_truthPt.png")

    c = Plots2D(total_JetTruthPt, total_JetDNNE, total_JetTruthE, xlabel='p_{T}^{true} [GeV]', ylabel='E_{jet}^{DNN} / E_{jet}^{true}')
    c.save("./plots/2DPlots_Calib_truthE_truthPt.png")

    c = Plots2D(total_JetTruthPt, total_JetPUE, total_JetTruthE, xlabel='p_{T}^{true} [GeV]', ylabel='E_{jet}^{PU} / E_{jet}^{true}')
    c.save("./plots/2DPlots_GNN_truthE_truthPt.png")

    c = Plots2D(total_JetTruthPt, total_JetRawE_signal, total_JetTruthE, xlabel='p_{T}^{true} [GeV]', ylabel='#Sigma E_{clus}^{label=1}  / E_{jet}^{true}')
    c.save("./plots/2DPlots_SignalJet_truthE_truthPt.png")

    ## division by JetArea
    c = Plots2D(total_JetTruthPt, total_JetDNNE, total_JetAreaE, xlabel='p_{T}^{true} [GeV]', ylabel='E_{jet}^{DNN} / E_{jet}^{area}')
    c.save("./plots/2DPlots_Calib_Area_truthPt.png")

    c = Plots2D(total_JetTruthPt, total_JetPUE, total_JetAreaE, xlabel='p_{T}^{true} [GeV]', ylabel='E_{jet}^{PU} / E_{jet}^{area}')
    c.save("./plots/2DPlots_GNN_Area_truthPt.png")

    c = Plots2D(total_JetTruthPt, total_JetRawE_signal, total_JetAreaE, xlabel='p_{T}^{true} [GeV]', ylabel='#Sigma E_{clus}^{label=1}  / E_{jet}^{area}')
    c.save("./plots/2DPlots_SignalJet_Area_truthPt.png")


    Profile2DPlots(total_JetTruthPt, total_JetPUE, total_JetAreaE, total_JetTruthE, total_weighted_score, xlabel='p_{T}^{true} [GeV]', ylabel='E_{jet}^{PU} / E_{jet}^{area}')
    # c.save("./plots/Profile2DPlots_GNN_Area_truthPt.png")
    # c.SaveAs("./plots/Profile2DPlots_GNN_Area_truthPt.png")

    c1, c2 = GetMedianAndIQR(total_JetTruthPt, total_JetRawE_signal, total_JetDNNE, total_JetPUE, total_JetAreaE, label='area')
    c1.save("./plots/Median_usingArea_vs_truthPt.png")
    c2.save("./plots/IQR_usingArea_vs_truthPt.png")

    c1, c2 = GetMedianAndIQR(total_JetTruthPt, total_JetRawE_signal, total_JetDNNE, total_JetPUE, total_JetTruthE, label='true')
    c1.save("./plots/Median_usingTruth_vs_truthPt.png")
    c2.save("./plots/IQR_usingTruth_vs_truthPt.png")
    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
