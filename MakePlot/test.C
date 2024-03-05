#define HistoMaker_cxx
#include "HistoMaker.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TColor.h>
#include <TF1.h>
#include <TLegend.h>
#include <iostream>


struct JetInfo {
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

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    auto key = std::make_pair(eventNumber, jetCnt);
    if (jetInfoMap.find(key) == jetInfoMap.end()) {
      // jetInfoMap[key] = {clusterE, clusterE*Scores, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, labels};
      // jetInfoMap[key] = {clusterE, clusterE*Scores, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, truthJetPt, labels};
      jetInfoMap[key] = {clusterE, clusterEDNN, clusterECalib, cluster_ENG_CALIB_TOT, jetRawE, jetCalE, truthJetE, truthJetPt, labels};
    } else {
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


  }

  // Fill Tree
  TFile *file = new TFile("output.root", "RECREATE");
  TTree *tree = new TTree("jetInfoTree", "Tree containing jet information");
  JetInfo jetInfo;
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
    tree->Fill();
  }
  file->Write();
  file->Close();



}
