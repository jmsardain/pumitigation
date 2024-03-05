#define HistoMaker_cxx
#include "HistoMaker.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TColor.h>
#include <TLatex.h>
#include <iostream>
#include "AtlasStyle.h"
#include "AtlasLabels.h"
#include "AtlasUtils.h"

void myTextHere(Double_t x,Double_t y,Color_t color, const char *text,
       float tsize=25,float angle=0.)
 {
   //Double_t tsize=46;
   TLatex l;
   //l.SetTextAlign(12);
   l.SetTextSize(tsize);
   l.SetTextAngle(angle);
   l.SetNDC();
   l.SetTextColor(color);
   l.DrawLatex(x,y,text);
 }

void HistoMaker::Loop(){


  float m_xlabel   = 0.2;
  float m_ylabel   = 0.88;
  float m_interval = 0.06;
  float m_tsize    = 0.03;


  const Int_t NBINS = 99;
  Double_t clusterE_bins[NBINS+1] = {1.00000000e-01, 1.09647820e-01, 1.20226443e-01, 1.31825674e-01, 1.44543977e-01, 1.58489319e-01, 1.73780083e-01, 1.90546072e-01, 2.08929613e-01, 2.29086765e-01, 2.51188643e-01, 2.75422870e-01, 3.01995172e-01, 3.31131121e-01, 3.63078055e-01, 3.98107171e-01, 4.36515832e-01, 4.78630092e-01, 5.24807460e-01, 5.75439937e-01, 6.30957344e-01, 6.91830971e-01, 7.58577575e-01, 8.31763771e-01, 9.12010839e-01, 1.00000000e+00, 1.09647820e+00, 1.20226443e+00, 1.31825674e+00, 1.44543977e+00, 1.58489319e+00, 1.73780083e+00, 1.90546072e+00, 2.08929613e+00, 2.29086765e+00, 2.51188643e+00, 2.75422870e+00, 3.01995172e+00, 3.31131121e+00, 3.63078055e+00, 3.98107171e+00, 4.36515832e+00, 4.78630092e+00, 5.24807460e+00, 5.75439937e+00, 6.30957344e+00, 6.91830971e+00, 7.58577575e+00, 8.31763771e+00, 9.12010839e+00, 1.00000000e+01, 1.09647820e+01, 1.20226443e+01, 1.31825674e+01, 1.44543977e+01, 1.58489319e+01, 1.73780083e+01, 1.90546072e+01, 2.08929613e+01, 2.29086765e+01,2.51188643e+01, 2.75422870e+01, 3.01995172e+01, 3.31131121e+01, 3.63078055e+01, 3.98107171e+01, 4.36515832e+01, 4.78630092e+01, 5.24807460e+01, 5.75439937e+01, 6.30957344e+01, 6.91830971e+01, 7.58577575e+01, 8.31763771e+01, 9.12010839e+01, 1.00000000e+02, 1.09647820e+02, 1.20226443e+02, 1.31825674e+02, 1.44543977e+02, 1.58489319e+02, 1.73780083e+02, 1.90546072e+02, 2.08929613e+02, 2.29086765e+02, 2.51188643e+02, 2.75422870e+02, 3.01995172e+02, 3.31131121e+02, 3.63078055e+02, 3.98107171e+02, 4.36515832e+02, 4.78630092e+02, 5.24807460e+02, 5.75439937e+02, 6.30957344e+02, 6.91830971e+02, 7.58577575e+02, 8.31763771e+02, 9.12010839e+02};

    Do plot of delta rap (clusterEta - truthRap) and clusterEta - jetCalEta for PU and signal
  // all
  TH1D* hClusterE = new TH1D("", "", NBINS, clusterE_bins);
   TH1D* hDeltaRap_cluster_truth = new TH1D("", "", 100, -1, 1); 
   TH1D* hDeltaRap_cluster_cal   = new TH1D("", "", 100, -1, 1); 
    
  // signal
  TH1D* hClusterE_signal = new TH1D("", "", NBINS, clusterE_bins);
    TH1D* hDeltaRap_cluster_truth_signal = new TH1D("", "", 100, -1, 1); 
   TH1D* hDeltaRap_cluster_cal_signal   = new TH1D("", "", 100, -1, 1); 
    
  // PU
  TH1D* hClusterE_pu = new TH1D("", "", NBINS, clusterE_bins);
    TH1D* hDeltaRap_cluster_truth_pu = new TH1D("", "", 100, -1, 1); 
   TH1D* hDeltaRap_cluster_cal_pu   = new TH1D("", "", 100, -1, 1); 

  hClusterE->SetLineColor(kBlack);
  hClusterE_signal->SetLineColor(kRed);
  hClusterE_pu->SetLineColor(kBlue);

hDeltaRap_cluster_truth->SetLineColor(kBlack);
hDeltaRap_cluster_cal->SetLineColor(kBlack);

hDeltaRap_cluster_truth_signal->SetLineColor(kRed);
hDeltaRap_cluster_cal_signal->SetLineColor(kRed);

hDeltaRap_cluster_truth_pu->SetLineColor(kBlue);
hDeltaRap_cluster_cal_pu->SetLineColor(kBlue);
  
  TLegend*l = new TLegend(0.7, 0.7, 0.9, 0.9);
  l->AddEntry(hClusterE, "all", "l");
  l->AddEntry(hClusterE_signal, "signal", "l");
  l->AddEntry(hClusterE_pu, "pu", "l");

  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    // if(jentry> 100000) break;

    if(jentry%100000==0) std::cout << "Entry #" << jentry << std::endl;
    hClusterE->Fill(clusterE);

hDeltaRap_cluster_truth->Fill(clusterEta-truthJetRap);
hDeltaRap_cluster_cal->Fill(clusterEta-jetCalEta);

    if(cluster_ENG_CALIB_TOT<0.0001){
      hClusterE_pu->Fill(clusterE);
hDeltaRap_cluster_truth_pu->Fill(clusterEta-truthJetRap);
hDeltaRap_cluster_cal_pu->Fill(clusterEta-jetCalEta);
    } else {
      hClusterE_signal->Fill(clusterE);
hDeltaRap_cluster_truth_signal->Fill(clusterEta-truthJetRap);
hDeltaRap_cluster_cal_signal->Fill(clusterEta-jetCalEta);
    }
  }

  TCanvas*c = new TCanvas("", "", 500, 500);

  // myText(m_xlabel,ylabel,1,textchannel+"+"+sleveljet+", "+slevelcutptjet+", "+slevelbtag,14);
  hClusterE->Draw();
  hClusterE_signal->Draw("same");
  hClusterE_pu->Draw("same");
  l->Draw("same");
  c->SetLogx();
  c->SetLogy();
  ATLAS_LABEL(m_xlabel,m_ylabel,kBlack);
  myText(m_xlabel+0.13,m_ylabel,kBlack,"Internal");
  myText(m_xlabel,m_ylabel-m_interval,1,"#sqrt{s}=13 TeV, 140.1 fb^{-1}");
  c->SaveAs("./plots/clusterE.png");

TCanvas*c1 = new TCanvas("", "", 500, 500);
hDeltaRap_cluster_truth->GetXaxis()->SetTitle("clusterEta - truthJetRap");
hDeltaRap_cluster_truth->Draw();
hDeltaRap_cluster_truth_signal->Draw("same");
hDeltaRap_cluster_truth_pu->Draw("same");
l->Draw("same");
c1->SetLogy();
ATLAS_LABEL(m_xlabel,m_ylabel,kBlack);
myText(m_xlabel+0.13,m_ylabel,kBlack,"Internal");
myText(m_xlabel,m_ylabel-m_interval,1,"#sqrt{s}=13 TeV, 140.1 fb^{-1}");
c1->SaveAs("./plots/clusterEta_Truth.png");

TCanvas*c2 = new TCanvas("", "", 500, 500);
hDeltaRap_cluster_truth->GetXaxis()->SetTitle("clusterEta - jetCalEta");
hDeltaRap_cluster_cal->Draw();
hDeltaRap_cluster_cal_signal->Draw("same");
hDeltaRap_cluster_cal_pu->Draw("same");
l->Draw("same");
c2->SetLogy();
ATLAS_LABEL(m_xlabel,m_ylabel,kBlack);
myText(m_xlabel+0.13,m_ylabel,kBlack,"Internal");
myText(m_xlabel,m_ylabel-m_interval,1,"#sqrt{s}=13 TeV, 140.1 fb^{-1}");
c2->SaveAs("./plots/clusterEta_Cal.png");


}
