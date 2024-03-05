from ROOT import TFile, TH1D, TH2D, TH2F, TH1F, TCanvas, TGraph, TMultiGraph, TLegend, TColor, TLatex
from root_numpy import fill_hist
from root_numpy import array2hist as a2h
from root_numpy import hist2array as h2a
from root_numpy import tree2array
from rootplotting import ap
from rootplotting.tools import *
ROOT.gStyle.SetPalette(ROOT.kBird)
import pandas as pd
from scipy.stats import pearsonr, mode, iqr

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


def main():

    # file = ROOT.TFile.Open('/home/jmsardain/JetCalib/PUMitigation/final/plotting/output.root')
    # tree = file["jetInfoTree"]
    df = pd.read_csv('forPlotting.csv', sep=',')

    colours = [ROOT.kAzure + 7, ROOT.TColor.GetColor('#FF8C00'), ROOT.TColor.GetColor('#008026'), ROOT.TColor.GetColor('#24408E'), ROOT.TColor.GetColor('#732982')]
    sumClusterEem    = df["sumClusterE"]
    sumClusterENet   = df["sumClusterENet"]
    jetCalE          = df["jetCalE"]
    truthJetPt       = df["truthJetPt"]
    truthJetE        = df["truthJetE"]

    response_em  = sumClusterEem / truthJetE
    response_net = sumClusterENet / truthJetE
    response_cal = jetCalE / truthJetE

    # xaxis = np.logspace(1.30, 2.47, 15+1) ## meaning from 20 to 300
    xaxis = np.linspace(20, 300, 10+1) ## meaning from 20 to 300
    yaxis = np.linspace(0, 5, 100+1)

    c = ap.canvas(batch=True, size=(600,600))
    c.pads()[0]._bare().SetRightMargin(0.2)
    c.pads()[0]._bare().SetLogz()



    # xaxis = np.linspace(-1,  3, 100 + 1, endpoint=True)
    # yaxis = np.linspace(0, 2,  100 + 1, endpoint=True)

    h2JetEem_truthJetPt = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    h2JetENet_truthJetPt = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)
    h2JetECal_truthJetPt = ROOT.TH2F('', '', len(xaxis) - 1, xaxis, len(yaxis) - 1, yaxis)

    h1JetEem_truthJetPt = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h1JetENet_truthJetPt = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)
    h1JetECal_truthJetPt = ROOT.TH1F('', '', len(xaxis) - 1, xaxis)

    # BinLogX(h2JetEem_truthJetPt)
    # BinLogX(h2JetENet_truthJetPt)
    # BinLogX(h2JetECal_truthJetPt)
    #
    # BinLogX(h1JetEem_truthJetPt)
    # BinLogX(h1JetENet_truthJetPt)
    # BinLogX(h1JetECal_truthJetPt)

    mesh_em  = np.vstack((truthJetPt, response_em)).T
    mesh_net = np.vstack((truthJetPt, response_net)).T
    mesh_cal = np.vstack((truthJetPt, response_cal)).T

    fill_hist(h2JetEem_truthJetPt, mesh_em)
    fill_hist(h2JetENet_truthJetPt, mesh_net)
    fill_hist(h2JetECal_truthJetPt, mesh_cal)

    ## EM scale
    for ibinx in range(1, h2JetEem_truthJetPt.GetNbinsX()+1):
        median_em, median_net, median_cal = [], [], []
        for ibiny in range(1, h2JetEem_truthJetPt.GetNbinsY()+1):
            nem = int(h2JetEem_truthJetPt.GetBinContent(ibinx, ibiny))
            nnet = int(h2JetENet_truthJetPt.GetBinContent(ibinx, ibiny))
            ncal = int(h2JetECal_truthJetPt.GetBinContent(ibinx, ibiny))
            for _ in range(nem): median_em.append(h2JetEem_truthJetPt.GetYaxis().GetBinCenter(ibiny))
            for _ in range(nnet): median_net.append(h2JetENet_truthJetPt.GetYaxis().GetBinCenter(ibiny))
            for _ in range(ncal): median_cal.append(h2JetECal_truthJetPt.GetYaxis().GetBinCenter(ibiny))

        if not median_em: median_em = [-1]*h2JetEem_truthJetPt.GetNbinsX()
        if not median_net: median_net = [-1]*h2JetEem_truthJetPt.GetNbinsX()
        if not median_cal: median_cal = [-1]*h2JetEem_truthJetPt.GetNbinsX()

        val_em  = iqr(median_em, rng=(16, 84)) / (2 * np.median(median_em))
        val_net = iqr(median_net, rng=(16, 84)) / (2 * np.median(median_net))
        val_cal = iqr(median_cal, rng=(16, 84)) / (2 * np.median(median_cal))
        print(median_em)
        print(median_net)
        print(median_cal)
        h1JetEem_truthJetPt.SetBinContent(ibinx, val_em)
        h1JetEem_truthJetPt.SetBinError(ibinx, 0)
        h1JetENet_truthJetPt.SetBinContent(ibinx, val_net)
        h1JetENet_truthJetPt.SetBinError(ibinx, 0)
        h1JetECal_truthJetPt.SetBinContent(ibinx, val_cal)
        h1JetECal_truthJetPt.SetBinError(ibinx, 0)





    c.hist(h1JetEem_truthJetPt, markercolor=colours[0], linecolor=colours[0], label="EM scale", option='L')
    c.hist(h1JetENet_truthJetPt, markercolor=colours[1], linecolor=colours[1], label="GNN", option='L')
    c.hist(h1JetECal_truthJetPt, markercolor=colours[2], linecolor=colours[2], label="Calibrated", option='L')
    # c.logx()
    # c.xlim(2e1, 3e2)

    c.legend()
    c.xlabel("Truth jet p_{T} [GeV]")
    c.ylabel("IQR / (2*median)")
    c.text(["#sqrt{s} = 13 TeV"], qualifier='Simulation Internal')
    c.save("./plots/IQR_JetEnergy.png")

    c1 = ROOT.TCanvas('', '', 500, 500)
    h2JetEem_truthJetPt.Draw('colz')
    c1.SaveAs('plots/h2JetEem_truthJetPt.png')

    c2 = ROOT.TCanvas('', '', 500, 500)
    h2JetENet_truthJetPt.Draw('colz')
    c2.SaveAs('plots/h2JetENet_truthJetPt.png')

    c3 = ROOT.TCanvas('', '', 500, 500)
    h2JetECal_truthJetPt.Draw('colz')
    c3.SaveAs('plots/h2JetECal_truthJetPt.png')


    return

# Main function call.
if __name__ == '__main__':
    main()
    pass
