#define Plot_cxx
#include "Plot.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TColor.h>
#include <iostream>


struct JetInfo {
    Double_t sumClusterE;
    Double_t sumClusterENet;
    Double_t sumClusterECalib;
    Double_t sumClusterETruth;
    Double_t jetRawE;
    Double_t jetCalE;
    Double_t truthJetE;
    Double_t labels;
};

void Plot::Loop(){
  if (fChain == 0) return;

  std::map<std::pair<Float_t, Float_t>, JetInfo> jetInfoMap;
  TH1F*hscores_sig = new TH1F("", "", 100, 0, 1);
  TH1F*hscores_pu  = new TH1F("", "", 100, 0, 1);

  const Int_t NBINS = 99;
  Double_t clusterE_bins[NBINS+1] = {1.00000000e-01, 1.09647820e-01, 1.20226443e-01, 1.31825674e-01, 1.44543977e-01, 1.58489319e-01, 1.73780083e-01, 1.90546072e-01, 2.08929613e-01, 2.29086765e-01, 2.51188643e-01, 2.75422870e-01, 3.01995172e-01, 3.31131121e-01, 3.63078055e-01, 3.98107171e-01, 4.36515832e-01, 4.78630092e-01, 5.24807460e-01, 5.75439937e-01, 6.30957344e-01, 6.91830971e-01, 7.58577575e-01, 8.31763771e-01, 9.12010839e-01, 1.00000000e+00, 1.09647820e+00, 1.20226443e+00, 1.31825674e+00, 1.44543977e+00, 1.58489319e+00, 1.73780083e+00, 1.90546072e+00, 2.08929613e+00, 2.29086765e+00, 2.51188643e+00, 2.75422870e+00, 3.01995172e+00, 3.31131121e+00, 3.63078055e+00, 3.98107171e+00, 4.36515832e+00, 4.78630092e+00, 5.24807460e+00, 5.75439937e+00, 6.30957344e+00, 6.91830971e+00, 7.58577575e+00, 8.31763771e+00, 9.12010839e+00, 1.00000000e+01, 1.09647820e+01, 1.20226443e+01, 1.31825674e+01, 1.44543977e+01, 1.58489319e+01, 1.73780083e+01, 1.90546072e+01, 2.08929613e+01, 2.29086765e+01,2.51188643e+01, 2.75422870e+01, 3.01995172e+01, 3.31131121e+01, 3.63078055e+01, 3.98107171e+01, 4.36515832e+01, 4.78630092e+01, 5.24807460e+01, 5.75439937e+01, 6.30957344e+01, 6.91830971e+01, 7.58577575e+01, 8.31763771e+01, 9.12010839e+01, 1.00000000e+02, 1.09647820e+02, 1.20226443e+02, 1.31825674e+02, 1.44543977e+02, 1.58489319e+02, 1.73780083e+02, 1.90546072e+02, 2.08929613e+02, 2.29086765e+02, 2.51188643e+02, 2.75422870e+02, 3.01995172e+02, 3.31131121e+02, 3.63078055e+02, 3.98107171e+02, 4.36515832e+02, 4.78630092e+02, 5.24807460e+02, 5.75439937e+02, 6.30957344e+02, 6.91830971e+02, 7.58577575e+02, 8.31763771e+02, 9.12010839e+02};

  // TH1D* htruthJetE = new TH1D("", "htruthJetE", NBINS, clusterE_bins);
  // TH1D* hjetRawE = new TH1D("", "hjetRawE", NBINS, clusterE_bins);
  // TH1D* hjetCalE = new TH1D("", "hjetCalE", NBINS, clusterE_bins);
  // TH1D* hsumClusterE = new TH1D("", "hsumClusterE", NBINS, clusterE_bins);
  // TH1D* hsumClusterENet = new TH1D("", "hsumClusterENet", NBINS, clusterE_bins);
  // TH1D* hsumClusterECalib = new TH1D("", "hsumClusterECalib", NBINS, clusterE_bins);
  // TH1D* hsumClusterETruth = new TH1D("", "hsumClusterECalib", NBINS, clusterE_bins);

  TH1F*htruthJetE = new TH1F("", "", 100, 0, 1000);
  TH1F*hjetRawE = new TH1F("", "", 100, 0, 1000);
  TH1F*hjetCalE = new TH1F("", "", 100, 0, 1000);
  TH1F*hsumClusterE = new TH1F("", "", 100, 0, 1000);
  TH1F*hsumClusterENet = new TH1F("", "", 100, 0, 1000);
  TH1F*hsumClusterECalib = new TH1F("", "", 100, 0, 1000);
  TH1D* hsumClusterETruth = new TH1D("", "",100, 0, 1000);

  hscores_sig->SetLineColor(TColor::GetColor("#FF8C00"));
  hscores_pu->SetLineColor(TColor::GetColor("#008026"));

  hscores_sig->SetMarkerColor(TColor::GetColor("#FF8C00"));
  hscores_pu->SetMarkerColor(TColor::GetColor("#008026"));

  htruthJetE->SetLineColor(kAzure + 7);
  hjetRawE->SetLineColor(TColor::GetColor("#FF8C00"));
  hjetCalE->SetLineColor(kRed);
  hsumClusterE->SetLineColor(TColor::GetColor("#008026") );
  hsumClusterENet->SetLineColor(TColor::GetColor("#24408E"));
  hsumClusterETruth->SetLineColor(TColor::GetColor("#732982"));

  htruthJetE->SetMarkerColor(kAzure + 7);
  hjetRawE->SetMarkerColor(TColor::GetColor("#FF8C00"));
  hjetCalE->SetMarkerColor(kRed);
  hsumClusterE->SetMarkerColor(TColor::GetColor("#008026") );
  hsumClusterENet->SetMarkerColor(TColor::GetColor("#24408E"));
  hsumClusterETruth->SetMarkerColor(TColor::GetColor("#732982"));

  TLegend*l1 = new TLegend(0.7, 0.7, 0.9 ,0.9);
  l1->AddEntry(htruthJetE, "truthJetE", "l");
  l1->AddEntry(hjetRawE, "jetRawE", "l");
  l1->AddEntry(hjetCalE, "jetCalE", "l");
  l1->AddEntry(hsumClusterE, "sumClusterE", "l");
  l1->AddEntry(hsumClusterENet, "sumClusterENet", "l");
  l1->AddEntry(hsumClusterETruth, "hsumClusterETruth", "l");

  TLegend*l = new TLegend(0.8, 0.8, 0.95 ,0.95);
  l->AddEntry(hscores_sig, "Signal", "l");
  l->AddEntry(hscores_pu, "PU", "l");

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    // if (Cut(ientry) < 0) continue;
    // if(jentry>16) break;
    auto key = std::make_pair(eventNumber, jetCnt);
    if (jetInfoMap.find(key) == jetInfoMap.end()) {

      // jetInfoMap[key] = {clusterE, clusterE*Scores, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, labels};
      jetInfoMap[key] = {clusterE, clusterE*1, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, labels};
    } else {
      jetInfoMap[key].sumClusterE      += clusterE;
      // jetInfoMap[key].sumClusterENet   += clusterE * Scores;
      jetInfoMap[key].sumClusterENet   += clusterE * 1;
      jetInfoMap[key].sumClusterECalib += clusterECalib;
      jetInfoMap[key].sumClusterETruth += cluster_ENG_CALIB_TOT;
      jetInfoMap[key].jetRawE = jetRawE;
      jetInfoMap[key].jetCalE = jetCalE;
      jetInfoMap[key].truthJetE = truthJetE;
    }

    if(labels==0) hscores_sig->Fill(Scores);
    if(labels==1) hscores_pu->Fill(Scores);
  }


  for (const auto& entry : jetInfoMap) {
    //
    htruthJetE->Fill(entry.second.truthJetE);
    hjetRawE->Fill(entry.second.jetRawE);
    hjetCalE->Fill(entry.second.jetCalE);
    hsumClusterE->Fill(entry.second.sumClusterE);
    hsumClusterENet->Fill(entry.second.sumClusterENet);
    hsumClusterECalib->Fill(entry.second.sumClusterECalib);
    hsumClusterETruth->Fill(entry.second.sumClusterETruth);

  }



  TCanvas*c = new TCanvas("", "", 500, 500);
  c->SetLogy();
  hscores_sig->GetYaxis()->SetRangeUser(1e2, 1e6);
  hscores_sig->GetYaxis()->SetTitle("# clusters");
  hscores_sig->GetXaxis()->SetTitle("Score");
  hscores_sig->Draw("HIST");
  hscores_pu->Draw("HISTSAME");
  l->Draw("SAME");
  c->SaveAs("plots/scores.png");

  TCanvas*c1 = new TCanvas("", "", 500, 500);
  c1->SetLogy();
  // c1->SetLogx();
  htruthJetE->GetYaxis()->SetRangeUser(1e0, 1e7);
  htruthJetE->GetXaxis()->SetRangeUser(5, 1000);
  std::cout << hjetRawE->GetMean() << " " << hsumClusterE->GetMean() << std::endl;
  htruthJetE->Draw("HIST");
  hjetRawE->Draw("HISTSAME");
  hjetCalE->Draw("HISTSAME");
  hsumClusterE->Draw("HISTSAME");
  // hsumClusterENet->Draw("HISTSAME");
  hsumClusterETruth->Draw("HISTSAME");
  // hsumClusterECalib->Draw("HISTSAME");
  l1->Draw("SAME");
  c1->SaveAs("plots/energy.png");



}
