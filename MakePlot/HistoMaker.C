#define HistoMaker_cxx
#include "HistoMaker.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TColor.h>
#include <TF1.h>
#include <TLorentzVector.h>
#include <TLegend.h>
#include <iostream>

TH1D* GetTrend(TH1D* h1, TH1D* h1r, TH2D* h2){

  for(int ibinx=1; ibinx<=h2->GetNbinsX()+1; ibinx++){
    TH1D* h1slice = h2->ProjectionY("h1slice"+TString(std::to_string(ibinx)), ibinx, ibinx);
    if(h1slice->GetEntries() > 0){
      double binmax = h1slice->GetMaximumBin();
      double x = h1slice->GetXaxis()->GetBinCenter(binmax);
      h1slice->Fit("gaus", "", "", x-1, x+1);
      TF1* g = (TF1*)h1slice->GetListOfFunctions()->FindObject("gaus");
      h1->SetBinContent(ibinx, g->GetParameter(1));
      h1->SetBinError(ibinx, 0);
      h1r->SetBinContent(ibinx, g->GetParameter(2) / (2*g->GetParameter(1)));
      h1r->SetBinError(ibinx, 0);
    } else {
      h1->SetBinContent(ibinx, -1);
      h1->SetBinError(ibinx, 0);
      h1r->SetBinContent(ibinx, -1);
      h1r->SetBinError(ibinx, 0);
    }
  }
  return h1;
}


struct JetInfo {
    TLorentzVector jetEMpt;
    TLorentzVector jetPUpt;
    Double_t sumClusterE;
    Double_t sumClusterENet;
    Double_t sumClusterECalib;
    Double_t sumClusterETruth;
    Double_t jetRawE;
    Double_t jetCalE;
    Double_t truthJetE;
    Double_t truthJetPt;
    Double_t labels;
};

void HistoMaker::Loop(){

  if (fChain == 0) return;

  std::map<std::pair<Float_t, Float_t>, JetInfo> jetInfoMap;
  TH1F*hscores_sig = new TH1F("", "", 100, 0, 1);
  TH1F*hscores_pu  = new TH1F("", "", 100, 0, 1);

  const Int_t NBINS = 100;
  Double_t clusterE_bins[NBINS+1] = {1.00000000e-01, 1.09647820e-01, 1.20226443e-01, 1.31825674e-01, 1.44543977e-01, 1.58489319e-01, 1.73780083e-01, 1.90546072e-01, 2.08929613e-01, 2.29086765e-01, 2.51188643e-01, 2.75422870e-01, 3.01995172e-01, 3.31131121e-01, 3.63078055e-01, 3.98107171e-01, 4.36515832e-01, 4.78630092e-01, 5.24807460e-01, 5.75439937e-01, 6.30957344e-01, 6.91830971e-01, 7.58577575e-01, 8.31763771e-01, 9.12010839e-01, 1.00000000e+00, 1.09647820e+00, 1.20226443e+00, 1.31825674e+00, 1.44543977e+00, 1.58489319e+00, 1.73780083e+00, 1.90546072e+00, 2.08929613e+00, 2.29086765e+00, 2.51188643e+00, 2.75422870e+00, 3.01995172e+00, 3.31131121e+00, 3.63078055e+00, 3.98107171e+00, 4.36515832e+00, 4.78630092e+00, 5.24807460e+00, 5.75439937e+00, 6.30957344e+00, 6.91830971e+00, 7.58577575e+00, 8.31763771e+00, 9.12010839e+00, 1.00000000e+01, 1.09647820e+01, 1.20226443e+01, 1.31825674e+01, 1.44543977e+01, 1.58489319e+01, 1.73780083e+01, 1.90546072e+01, 2.08929613e+01, 2.29086765e+01, 2.51188643e+01, 2.75422870e+01, 3.01995172e+01, 3.31131121e+01, 3.63078055e+01, 3.98107171e+01, 4.36515832e+01, 4.78630092e+01, 5.24807460e+01, 5.75439937e+01, 6.30957344e+01, 6.91830971e+01, 7.58577575e+01, 8.31763771e+01, 9.12010839e+01, 1.00000000e+02, 1.09647820e+02, 1.20226443e+02, 1.31825674e+02, 1.44543977e+02, 1.58489319e+02, 1.73780083e+02, 1.90546072e+02, 2.08929613e+02, 2.29086765e+02, 2.51188643e+02, 2.75422870e+02, 3.01995172e+02, 3.31131121e+02, 3.63078055e+02, 3.98107171e+02, 4.36515832e+02, 4.78630092e+02, 5.24807460e+02, 5.75439937e+02, 6.30957344e+02, 6.91830971e+02, 7.58577575e+02, 8.31763771e+02, 9.12010839e+02, 1.00000000e+03};


  const Int_t NBINS_response = 18;
  Double_t response_bins[NBINS_response+1] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2};

  const Int_t NBINS_resolution = 18;
  Double_t resolution_bins[NBINS_resolution+1] = {0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36};

  const Int_t NBINS_ptTruth = 25;
  Double_t ptTruth_bins[NBINS_ptTruth+1] = {
    10.0, 12.02264434617413, 14.454397707459272, 17.378008287493753, 20.892961308540396, 25.118864315095795, 30.19951720402016, 36.30780547701014,
    43.65158322401661, 52.48074602497726, 63.09573444801933, 75.85775750291836, 91.20108393559097, 109.64781961431851, 131.82567385564073,
    158.48931924611142, 190.54607179632484, 229.08676527677747, 275.4228703338166, 331.1311214825911, 398.1071705534973, 478.630092322638,
    575.4399373371566, 691.8309709189363, 831.7637711026708, 1000.0
  };

  TH1F*htruthJetE = new TH1F("", "", NBINS, clusterE_bins);
  TH1F*hjetRawE = new TH1F("", "", NBINS, clusterE_bins);
  TH1F*hjetCalE = new TH1F("", "", NBINS, clusterE_bins);
  TH1F*hsumClusterE = new TH1F("", "", NBINS, clusterE_bins);
  TH1F*hsumClusterENet = new TH1F("", "", NBINS, clusterE_bins);
  TH1F*hsumClusterECalib = new TH1F("", "", NBINS, clusterE_bins);
  TH1D* hsumClusterETruth = new TH1D("", "", NBINS, clusterE_bins);

  TH2D* hJetResponse_Cal_pTtruth = new TH2D("", "", NBINS_ptTruth, ptTruth_bins, NBINS_response, response_bins);
  TH2D* hJetResponse_EM_pTtruth = new TH2D("", "", NBINS_ptTruth, ptTruth_bins, NBINS_response,response_bins);
  TH2D* hJetResponse_LCW_pTtruth = new TH2D("", "", NBINS_ptTruth, ptTruth_bins, NBINS_response, response_bins);
  TH2D* hJetResponse_GNN_pTtruth = new TH2D("", "", NBINS_ptTruth, ptTruth_bins, NBINS_response, response_bins);

  TH1D* hJetResponse_Cal_pTtruth_1D = new TH1D("", "", NBINS_ptTruth, ptTruth_bins);
  TH1D* hJetResponse_EM_pTtruth_1D = new TH1D("", "", NBINS_ptTruth, ptTruth_bins);
  TH1D* hJetResponse_LCW_pTtruth_1D = new TH1D("", "", NBINS_ptTruth, ptTruth_bins);
  TH1D* hJetResponse_GNN_pTtruth_1D = new TH1D("", "", NBINS_ptTruth, ptTruth_bins);

  TH1D* hJetResolution_Cal_pTtruth_1D = new TH1D("", "", NBINS_ptTruth, ptTruth_bins);
  TH1D* hJetResolution_EM_pTtruth_1D = new TH1D("", "",  NBINS_ptTruth, ptTruth_bins);
  TH1D* hJetResolution_LCW_pTtruth_1D = new TH1D("", "",  NBINS_ptTruth, ptTruth_bins);
  TH1D* hJetResolution_GNN_pTtruth_1D = new TH1D("", "",  NBINS_ptTruth, ptTruth_bins);

  TH2D* hJetResponse_Cal_Etruth = new TH2D("", "", NBINS, clusterE_bins, NBINS_response, response_bins);
  TH2D* hJetResponse_EM_Etruth = new TH2D("", "", NBINS, clusterE_bins, NBINS_response,response_bins);
  TH2D* hJetResponse_LCW_Etruth = new TH2D("", "", NBINS, clusterE_bins, NBINS_response, response_bins);
  TH2D* hJetResponse_GNN_Etruth = new TH2D("", "", NBINS, clusterE_bins, NBINS_response, response_bins);

  TH1D* hJetResponse_Cal_Etruth_1D = new TH1D("", "", NBINS, clusterE_bins);
  TH1D* hJetResponse_EM_Etruth_1D = new TH1D("", "", NBINS, clusterE_bins);
  TH1D* hJetResponse_LCW_Etruth_1D = new TH1D("", "", NBINS, clusterE_bins);
  TH1D* hJetResponse_GNN_Etruth_1D = new TH1D("", "", NBINS, clusterE_bins);

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

    TLorentzVector jetEMpt;
    TLorentzVector jetPUpt;

    auto key = std::make_pair(eventNumber, jetCnt);
    if (jetInfoMap.find(key) == jetInfoMap.end()) {
      // jetInfoMap[key] = {clusterE, clusterE*Scores, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, labels};
      // jetInfoMap[key] = {clusterE, clusterE*Scores, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, truthJetPt, labels};
      TLorentzVector clusterEM;
      TLorentzVector clusterPU;
      jetEMpt.SetPtEtaPhiE(0.,0.,0.,0.);
      jetPUpt.SetPtEtaPhiE(0.,0.,0.,0.);
      clusterEM.SetPtEtaPhiE(clusterPt, clusterEta, clusterPhi, clusterE);
      clusterPU.SetPtEtaPhiE(clusterPt, clusterEta, clusterPhi, clusterEDNN);
      jetEMpt += clusterEM;
      jetPUpt += clusterPU;

      jetInfoMap[key] = {jetEMpt, jetPUpt, clusterE, clusterEDNN, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, truthJetPt, labels};
    } else {
      TLorentzVector clusterEM;
      TLorentzVector clusterPU;
      clusterEM.SetPtEtaPhiE(clusterPt, clusterEta, clusterPhi, clusterE);
      clusterPU.SetPtEtaPhiE(clusterPt, clusterEta, clusterPhi, clusterEDNN);
      jetEMpt += clusterEM;
      jetPUpt += clusterPU;
      jetInfoMap[key].jetEMpt          += jetEMpt;
      jetInfoMap[key].jetPUpt          += jetPUpt;
      jetInfoMap[key].sumClusterE      += clusterE;
      jetInfoMap[key].sumClusterENet   += clusterEDNN;
      // jetInfoMap[key].sumClusterENet   += clusterE * 1;
      jetInfoMap[key].sumClusterECalib += clusterECalib;
      jetInfoMap[key].sumClusterETruth += cluster_ENG_CALIB_TOT;
      // jetInfoMap[key].jetRawE = TMath::Exp((jetRawE * 1.011533972808067)+4.294673213759488);
      jetInfoMap[key].jetRawE = jetRawE;
      jetInfoMap[key].jetCalE = jetCalE;
      jetInfoMap[key].truthJetE = truthJetE;
      jetInfoMap[key].truthJetPt = truthJetPt;

    }

    if(labels==0) hscores_sig->Fill(Scores);
    if(labels==1) hscores_pu->Fill(Scores);

  }

  // Fill Tree
  TFile *file = new TFile("output.root", "RECREATE");
  TTree *tree = new TTree("jetInfoTree", "Tree containing jet information");
  JetInfo jetInfo;
  double jetEMpt_pt = -999;
  double jetPUpt_pt = -999;

  tree->Branch("jetEMpt", &jetEMpt_pt);
  tree->Branch("jetPUpt", &jetPUpt_pt);
  tree->Branch("sumClusterE", &jetInfo.sumClusterE);
  tree->Branch("sumClusterENet", &jetInfo.sumClusterENet);
  tree->Branch("sumClusterECalib", &jetInfo.sumClusterECalib);
  tree->Branch("sumClusterETruth", &jetInfo.sumClusterETruth);
  tree->Branch("jetRawE", &jetInfo.jetRawE);
  tree->Branch("jetCalE", &jetInfo.jetCalE);
  tree->Branch("truthJetE", &jetInfo.truthJetE);
  tree->Branch("truthJetPt", &jetInfo.truthJetPt);
  tree->Branch("labels", &jetInfo.labels);

  for (const auto& entry : jetInfoMap) {
    //
    jetInfo = entry.second;
    jetEMpt_pt = entry.second.jetEMpt.Pt();
    jetPUpt_pt = entry.second.jetPUpt.Pt();
    tree->Fill();




    htruthJetE->Fill(entry.second.truthJetE);
    hjetRawE->Fill(entry.second.jetRawE);
    hjetCalE->Fill(entry.second.jetCalE);
    hsumClusterE->Fill(entry.second.sumClusterE);
    hsumClusterENet->Fill(entry.second.sumClusterENet);
    hsumClusterECalib->Fill(entry.second.sumClusterECalib);
    hsumClusterETruth->Fill(entry.second.sumClusterETruth);

    double truthpT  = entry.second.truthJetPt;
    double jetResponse_Cal  = entry.second.jetCalE / entry.second.truthJetE;
    double jetResponse_EM   = entry.second.jetRawE / entry.second.truthJetE;
    double jetResponse_LCW   = entry.second.sumClusterECalib / entry.second.truthJetE;
    double jetResponse_GNN  = entry.second.sumClusterENet / entry.second.truthJetE;

    hJetResponse_Cal_pTtruth->Fill(truthpT, jetResponse_Cal);
    hJetResponse_EM_pTtruth->Fill(truthpT, jetResponse_EM);
    hJetResponse_LCW_pTtruth->Fill(truthpT, jetResponse_LCW);
    hJetResponse_GNN_pTtruth->Fill(truthpT, jetResponse_GNN);

    hJetResponse_Cal_Etruth->Fill(entry.second.truthJetE, jetResponse_Cal);
    hJetResponse_EM_Etruth->Fill(entry.second.truthJetE, jetResponse_EM);
    hJetResponse_LCW_Etruth->Fill(entry.second.truthJetE, jetResponse_LCW);
    hJetResponse_GNN_Etruth->Fill(entry.second.truthJetE, jetResponse_GNN);

  }
  file->Write();
  file->Close();
  // truthJetPt
  //
  GetTrend(hJetResponse_Cal_pTtruth_1D, hJetResolution_Cal_pTtruth_1D, hJetResponse_Cal_pTtruth);
  GetTrend(hJetResponse_EM_pTtruth_1D,  hJetResolution_EM_pTtruth_1D,  hJetResponse_EM_pTtruth);
  GetTrend(hJetResponse_LCW_pTtruth_1D, hJetResolution_LCW_pTtruth_1D, hJetResponse_LCW_pTtruth);
  GetTrend(hJetResponse_GNN_pTtruth_1D, hJetResolution_GNN_pTtruth_1D, hJetResponse_GNN_pTtruth);
  // truthJetE
  // GetTrend(hJetResponse_Cal_Etruth_1D, hJetResponse_Cal_Etruth);
  // GetTrend(hJetResponse_EM_Etruth_1D, hJetResponse_EM_Etruth);
  // GetTrend(hJetResponse_LCW_Etruth_1D, hJetResponse_LCW_Etruth);
  // GetTrend(hJetResponse_GNN_Etruth_1D, hJetResponse_GNN_Etruth);

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
  c1->SetLogx();
  htruthJetE->GetYaxis()->SetRangeUser(1e0, 1e7);
  htruthJetE->GetXaxis()->SetRangeUser(5, 1000);
  std::cout << hjetRawE->GetMean() << " " << hsumClusterE->GetMean() << std::endl;
  htruthJetE->GetXaxis()->SetTitle("Energy [GeV]");
  htruthJetE->GetYaxis()->SetTitle("# jets");
  htruthJetE->Draw("HIST");
  hjetRawE->Draw("HISTSAME");
  hjetCalE->Draw("HISTSAME");
  hsumClusterE->Draw("HISTSAME");
  hsumClusterENet->Draw("HISTSAME");
  hsumClusterETruth->Draw("HISTSAME");
  // hsumClusterECalib->Draw("HISTSAME");
  l1->Draw("SAME");
  c1->SaveAs("plots/energy.png");

  TCanvas*c2 = new TCanvas("", "", 500, 500);
  c2->SetRightMargin(0.2);
  c2->SetLogx();
  hJetResponse_Cal_pTtruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_Cal_pTtruth->GetYaxis()->SetTitle("Jet response Cal");
  hJetResponse_Cal_pTtruth->GetXaxis()->SetTitle("p_{T}^{truth} [GeV]");
  hJetResponse_Cal_pTtruth_1D->SetLineColor(kRed);
  hJetResponse_Cal_pTtruth_1D->SetMarkerColor(kRed);
  hJetResponse_Cal_pTtruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_Cal_pTtruth->Draw("colz");
  hJetResponse_Cal_pTtruth_1D->Draw("SAME");
  c2->SaveAs("plots/2D_jetresponseCal_pttruth.png");

  TCanvas*c3 = new TCanvas("", "", 500, 500);
  c3->SetRightMargin(0.2);
  c3->SetLogx();
  hJetResponse_EM_pTtruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_EM_pTtruth->GetYaxis()->SetTitle("Jet response EM");
  hJetResponse_EM_pTtruth->GetXaxis()->SetTitle("p_{T}^{truth} [GeV]");
  hJetResponse_EM_pTtruth_1D->SetLineColor(kRed);
  hJetResponse_EM_pTtruth_1D->SetMarkerColor(kRed);
  hJetResponse_EM_pTtruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_EM_pTtruth->Draw("colz");
  hJetResponse_EM_pTtruth_1D->Draw("SAME");
  c3->SaveAs("plots/2D_jetresponseEM_pttruth.png");

  TCanvas*c4 = new TCanvas("", "", 500, 500);
  c4->SetRightMargin(0.2);
  c4->SetLogx();
  hJetResponse_LCW_pTtruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_LCW_pTtruth->GetYaxis()->SetTitle("Jet response LCW");
  hJetResponse_LCW_pTtruth->GetXaxis()->SetTitle("p_{T}^{truth} [GeV]");
  hJetResponse_LCW_pTtruth_1D->SetLineColor(kRed);
  hJetResponse_LCW_pTtruth_1D->SetMarkerColor(kRed);
  hJetResponse_LCW_pTtruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_LCW_pTtruth->Draw("colz");
  hJetResponse_LCW_pTtruth_1D->Draw("SAME");
  c4->SaveAs("plots/2D_jetresponseLCW_pttruth.png");

  TCanvas*c5 = new TCanvas("", "", 500, 500);
  c5->SetRightMargin(0.2);
  c5->SetLogx();
  hJetResponse_GNN_pTtruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_GNN_pTtruth->GetYaxis()->SetTitle("Jet response GNN");
  hJetResponse_GNN_pTtruth->GetXaxis()->SetTitle("p_{T}^{truth} [GeV]");
  hJetResponse_GNN_pTtruth_1D->SetLineColor(kRed);
  hJetResponse_GNN_pTtruth_1D->SetMarkerColor(kRed);
  hJetResponse_GNN_pTtruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_GNN_pTtruth->Draw("colz");
  hJetResponse_GNN_pTtruth_1D->Draw("same");
  c5->SaveAs("plots/2D_jetresponseGNN_pttruth.png");

  TCanvas*c6 = new TCanvas("", "", 500, 500);
  c6->SetLogx();
  hJetResponse_Cal_pTtruth_1D->SetLineColor(kRed);
  hJetResponse_Cal_pTtruth_1D->SetMarkerColor(kRed);
  hJetResponse_EM_pTtruth_1D->SetLineColor(kBlue);
  hJetResponse_EM_pTtruth_1D->SetMarkerColor(kBlue);
  hJetResponse_LCW_pTtruth_1D->SetLineColor(kBlack);
  hJetResponse_LCW_pTtruth_1D->SetMarkerColor(kBlack);
  hJetResponse_GNN_pTtruth_1D->SetLineColor(kGreen+8);
  hJetResponse_GNN_pTtruth_1D->SetMarkerColor(kGreen+8);
  hJetResponse_Cal_pTtruth_1D->GetXaxis()->SetTitle("p_{T}^{truth} [GeV]");
  hJetResponse_Cal_pTtruth_1D->GetYaxis()->SetTitle("Jet response");
  hJetResponse_Cal_pTtruth_1D->GetYaxis()->SetRangeUser(0, 2);
  hJetResponse_Cal_pTtruth_1D->Draw("same");
  hJetResponse_EM_pTtruth_1D->Draw("same");
  hJetResponse_LCW_pTtruth_1D->Draw("same");
  hJetResponse_GNN_pTtruth_1D->Draw("same");
  TLegend*l2 = new TLegend(0.7, 0.7, 0.9, 0.9);
  l2->AddEntry(hJetResponse_Cal_pTtruth_1D, "Cal", "l");
  l2->AddEntry(hJetResponse_EM_pTtruth_1D, "EM", "l");
  l2->AddEntry(hJetResponse_LCW_pTtruth_1D, "LCW", "l");
  l2->AddEntry(hJetResponse_GNN_pTtruth_1D, "GNN", "l");
  l2->Draw("same");
  c6->SaveAs("plots/2D_jetresponses_pttruth.png");

  TCanvas*c61 = new TCanvas("", "", 500, 500);
  c61->SetLogx();
  hJetResolution_Cal_pTtruth_1D->SetLineColor(kRed);
  hJetResolution_Cal_pTtruth_1D->SetMarkerColor(kRed);
  hJetResolution_EM_pTtruth_1D->SetLineColor(kBlue);
  hJetResolution_EM_pTtruth_1D->SetMarkerColor(kBlue);
  hJetResolution_LCW_pTtruth_1D->SetLineColor(kBlack);
  hJetResolution_LCW_pTtruth_1D->SetMarkerColor(kBlack);
  hJetResolution_GNN_pTtruth_1D->SetLineColor(kGreen+8);
  hJetResolution_GNN_pTtruth_1D->SetMarkerColor(kGreen+8);
  hJetResolution_Cal_pTtruth_1D->GetXaxis()->SetTitle("p_{T}^{truth} [GeV]");
  hJetResolution_Cal_pTtruth_1D->GetYaxis()->SetTitle("IQR / (2 * median)");
  hJetResolution_Cal_pTtruth_1D->GetYaxis()->SetRangeUser(0, 0.3);
  hJetResolution_Cal_pTtruth_1D->Draw("same");
  hJetResolution_EM_pTtruth_1D->Draw("same");
  hJetResolution_LCW_pTtruth_1D->Draw("same");
  hJetResolution_GNN_pTtruth_1D->Draw("same");
  TLegend*l21 = new TLegend(0.7, 0.7, 0.9, 0.9);
  l21->AddEntry(hJetResolution_Cal_pTtruth_1D, "Cal", "l");
  l21->AddEntry(hJetResolution_EM_pTtruth_1D, "EM", "l");
  l21->AddEntry(hJetResolution_LCW_pTtruth_1D, "LCW", "l");
  l21->AddEntry(hJetResolution_GNN_pTtruth_1D, "GNN", "l");
  l21->Draw("same");
  c61->SaveAs("plots/2D_jetresolutions_pttruth.png");


  // truthJetE
  TCanvas*c7 = new TCanvas("", "", 500, 500);
  c7->SetRightMargin(0.2);
  c7->SetLogx();
  hJetResponse_Cal_Etruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_Cal_Etruth->GetYaxis()->SetTitle("Jet response Cal");
  hJetResponse_Cal_Etruth->GetXaxis()->SetTitle("E^{truth} [GeV]");
  hJetResponse_Cal_Etruth_1D->SetLineColor(kRed);
  hJetResponse_Cal_Etruth_1D->SetMarkerColor(kRed);
  hJetResponse_Cal_Etruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_Cal_Etruth->Draw("colz");
  hJetResponse_Cal_Etruth_1D->Draw("SAME");
  c7->SaveAs("plots/2D_jetresponseCal_Etruth.png");

  TCanvas*c8 = new TCanvas("", "", 500, 500);
  c8->SetRightMargin(0.2);
  c8->SetLogx();
  hJetResponse_EM_Etruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_EM_Etruth->GetYaxis()->SetTitle("Jet response EM");
  hJetResponse_EM_Etruth->GetXaxis()->SetTitle("E^{truth} [GeV]");
  hJetResponse_EM_Etruth_1D->SetLineColor(kRed);
  hJetResponse_EM_Etruth_1D->SetMarkerColor(kRed);
  hJetResponse_EM_Etruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_EM_Etruth->Draw("colz");
  hJetResponse_EM_Etruth_1D->Draw("SAME");
  c8->SaveAs("plots/2D_jetresponseEM_Etruth.png");

  TCanvas*c9 = new TCanvas("", "", 500, 500);
  c9->SetRightMargin(0.2);
  c9->SetLogx();
  hJetResponse_LCW_Etruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_LCW_Etruth->GetYaxis()->SetTitle("Jet response LCW");
  hJetResponse_LCW_Etruth->GetXaxis()->SetTitle("E^{truth} [GeV]");
  hJetResponse_LCW_Etruth_1D->SetLineColor(kRed);
  hJetResponse_LCW_Etruth_1D->SetMarkerColor(kRed);
  hJetResponse_LCW_Etruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_LCW_Etruth->Draw("colz");
  hJetResponse_LCW_Etruth_1D->Draw("SAME");
  c9->SaveAs("plots/2D_jetresponseLCW_Etruth.png");

  TCanvas*c10 = new TCanvas("", "", 500, 500);
  c10->SetRightMargin(0.2);
  c10->SetLogx();
  hJetResponse_GNN_Etruth->GetZaxis()->SetTitle("# jets");
  hJetResponse_GNN_Etruth->GetYaxis()->SetTitle("Jet response GNN");
  hJetResponse_GNN_Etruth->GetXaxis()->SetTitle("E^{truth} [GeV]");
  hJetResponse_GNN_Etruth_1D->SetLineColor(kRed);
  hJetResponse_GNN_Etruth_1D->SetMarkerColor(kRed);
  hJetResponse_GNN_Etruth->GetYaxis()->SetRangeUser(0, 3);
  hJetResponse_GNN_Etruth->Draw("colz");
  hJetResponse_GNN_Etruth_1D->Draw("same");
  c10->SaveAs("plots/2D_jetresponseGNN_Etruth.png");

  TCanvas*c11 = new TCanvas("", "", 500, 500);
  c11->SetLogx();
  hJetResponse_Cal_Etruth_1D->SetLineColor(kRed);
  hJetResponse_Cal_Etruth_1D->SetMarkerColor(kRed);
  hJetResponse_EM_Etruth_1D->SetLineColor(kBlue);
  hJetResponse_EM_Etruth_1D->SetMarkerColor(kBlue);
  hJetResponse_LCW_Etruth_1D->SetLineColor(kBlack);
  hJetResponse_LCW_Etruth_1D->SetMarkerColor(kBlack);
  hJetResponse_GNN_Etruth_1D->SetLineColor(kGreen+8);
  hJetResponse_GNN_Etruth_1D->SetMarkerColor(kGreen+8);
  hJetResponse_Cal_Etruth_1D->GetXaxis()->SetTitle("E^{truth} [GeV]");
  hJetResponse_Cal_Etruth_1D->GetYaxis()->SetTitle("Jet response");
  hJetResponse_Cal_Etruth_1D->GetYaxis()->SetRangeUser(0, 2);
  hJetResponse_Cal_Etruth_1D->Draw("same");
  hJetResponse_EM_Etruth_1D->Draw("same");
  hJetResponse_LCW_Etruth_1D->Draw("same");
  hJetResponse_GNN_Etruth_1D->Draw("same");
  TLegend*l3 = new TLegend(0.7, 0.7, 0.9, 0.9);
  l3->AddEntry(hJetResponse_Cal_Etruth_1D, "Cal", "l");
  l3->AddEntry(hJetResponse_EM_Etruth_1D, "EM", "l");
  l3->AddEntry(hJetResponse_LCW_Etruth_1D, "LCW", "l");
  l3->AddEntry(hJetResponse_GNN_Etruth_1D, "GNN", "l");
  l3->Draw("same");
  c11->SaveAs("plots/2D_jetresponses_Etruth.png");


}
