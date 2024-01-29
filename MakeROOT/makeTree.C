#define makeTree_cxx
#include "makeTree.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <TVector3.h>

void makeTree::Loop(){

  if (fChain == 0) return;


  TFile *outputFile = new TFile("output.root", "RECREATE");
  TTree *outputTree = new TTree("ClusterTree", "ntuple");


  Double_t out_eventNumber;
  Double_t out_clusterE;
  Double_t out_clusterEta;
  Double_t out_cluster_CENTER_LAMBDA;
  Double_t out_cluster_CENTER_MAG;
  Double_t out_cluster_ENG_FRAC_EM;
  Double_t out_cluster_FIRST_ENG_DENS;
  Double_t out_cluster_LATERAL;
  Double_t out_cluster_LONGITUDINAL;
  Double_t out_cluster_PTD;
  Double_t out_cluster_time;
  Double_t out_cluster_ISOLATION;
  Double_t out_cluster_SECOND_TIME;
  Double_t out_cluster_SIGNIFICANCE;
  Double_t out_nPrimVtx;
  Double_t out_avgMu;
  Double_t out_jetCnt;
  Double_t out_clusterPhi;
  Double_t out_zL;
  Double_t out_zT;
  Double_t out_zRel;
  Double_t out_jetCalE;
  Double_t out_jetCalEta;
  Double_t out_jetRawE;
  Double_t out_jetRawPt;
  Double_t out_truthJetE;
  Double_t out_truthJetPt;
  Double_t out_clusterECalib;
  Double_t out_cluster_ENG_CALIB_TOT;

  outputTree->Branch("eventNumber", &out_eventNumber, "eventNumber/D");
  outputTree->Branch("jetCnt", &out_jetCnt, "jetCnt/D");
  outputTree->Branch("clusterE", &out_clusterE, "clusterE/D");
  outputTree->Branch("clusterEta", &out_clusterEta, "clusterEta/D");
  outputTree->Branch("clusterPhi", &out_clusterPhi, "clusterPhi/D");
  outputTree->Branch("cluster_CENTER_LAMBDA", &out_cluster_CENTER_LAMBDA, "cluster_CENTER_LAMBDA/D");
  outputTree->Branch("cluster_CENTER_MAG", &out_cluster_CENTER_MAG, "cluster_CENTER_MAG/D");
  outputTree->Branch("cluster_ENG_FRAC_EM", &out_cluster_ENG_FRAC_EM, "cluster_ENG_FRAC_EM/D");
  outputTree->Branch("cluster_FIRST_ENG_DENS", &out_cluster_FIRST_ENG_DENS, "cluster_FIRST_ENG_DENS/D");
  outputTree->Branch("cluster_LATERAL", &out_cluster_LATERAL, "cluster_LATERAL/D");
  outputTree->Branch("cluster_LONGITUDINAL", &out_cluster_LONGITUDINAL, "cluster_LONGITUDINAL/D");
  outputTree->Branch("cluster_PTD", &out_cluster_PTD, "cluster_PTD/D");
  outputTree->Branch("cluster_time", &out_cluster_time, "cluster_time/D");
  outputTree->Branch("cluster_ISOLATION", &out_cluster_ISOLATION, "cluster_ISOLATION/D");
  outputTree->Branch("cluster_SECOND_TIME", &out_cluster_SECOND_TIME, "cluster_SECOND_TIME/D");
  outputTree->Branch("cluster_SIGNIFICANCE", &out_cluster_SIGNIFICANCE, "cluster_SIGNIFICANCE/D");
  outputTree->Branch("nPrimVtx", &out_nPrimVtx, "nPrimVtx/D");
  outputTree->Branch("avgMu", &out_avgMu, "avgMu/D");
  outputTree->Branch("zL", &out_zL, "zL/D");
  outputTree->Branch("zT", &out_zT, "zT/D");
  outputTree->Branch("zRel", &out_zRel, "zRel/D");
  outputTree->Branch("jetCalE", &out_jetCalE, "jetCalE/D");
  outputTree->Branch("jetCalEta", &out_jetCalEta, "jetCalEta/D");
  outputTree->Branch("jetRawE", &out_jetRawE, "jetRawE/D");
  outputTree->Branch("jetRawPt", &out_jetRawPt, "jetRawPt/D");
  outputTree->Branch("truthJetE", &out_truthJetE, "truthJetE/D");
  outputTree->Branch("truthJetPt", &out_truthJetPt, "truthJetPt/D");
  outputTree->Branch("clusterECalib", &out_clusterECalib, "clusterECalib/D");
  outputTree->Branch("cluster_ENG_CALIB_TOT", &out_cluster_ENG_CALIB_TOT, "cluster_ENG_CALIB_TOT/D");


  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    if(jentry%100000==0) std::cout << "Entry #" << jentry << std::endl;
    // if (i>100000) break;
    // Get every other jetCnt
    if(jetCnt%2==0) continue;
    // Calculate the sum and assign it to the new branch
    TVector3 cluster_vec;
    cluster_vec.SetPtEtaPhi(clusterPt, clusterEta, clusterPhi);

    TVector3 jet_vec;
    jet_vec.SetPtEtaPhi(jetRawPt, jetRawEta, jetRawPhi);


    out_zT = cluster_vec.Pt()/jet_vec.Pt();
    out_zL = cluster_vec.Dot(jet_vec)/jet_vec.Mag2();
    out_zRel = (cluster_vec.Cross(jet_vec)).Mag()/jet_vec.Mag2();

    out_eventNumber = eventNumber;
    out_clusterE = clusterE;
    out_clusterEta = clusterEta;
    out_cluster_CENTER_LAMBDA = cluster_CENTER_LAMBDA;
    out_cluster_CENTER_MAG = cluster_CENTER_MAG;
    out_cluster_ENG_FRAC_EM = cluster_ENG_FRAC_EM;
    out_cluster_FIRST_ENG_DENS = cluster_FIRST_ENG_DENS;
    out_cluster_LATERAL = cluster_LATERAL;
    out_cluster_LONGITUDINAL = cluster_LONGITUDINAL;
    out_cluster_PTD = cluster_PTD;
    out_cluster_time = cluster_time;
    out_cluster_ISOLATION = cluster_ISOLATION;
    out_cluster_SECOND_TIME = cluster_SECOND_TIME;
    out_cluster_SIGNIFICANCE = cluster_SIGNIFICANCE;
    out_nPrimVtx = nPrimVtx;
    out_avgMu = avgMu;
    out_jetCnt = jetCnt;
    out_clusterPhi = clusterPhi;
    out_jetCalE = jetCalE;
    out_jetCalEta = jetCalEta;
    out_jetRawE = jetRawE;
    out_jetRawPt = jetRawPt;
    out_truthJetE = truthJetE;
    out_truthJetPt = truthJetPt;
    out_clusterECalib = clusterECalib;
    out_cluster_ENG_CALIB_TOT = cluster_ENG_CALIB_TOT;
    // Fill the output tree
    outputTree->Fill();
  }
  // Write the output tree to the output file
  outputFile->Write();
  // Close both files
  outputFile->Close();
}
