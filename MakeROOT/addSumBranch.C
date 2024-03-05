#include <TFile.h>
#include <TTree.h>
#include "TVector3.h"

#include "TFile.h"
#include "TTree.h"




void addSumBranch() {
    TFile *inputFile = TFile::Open("/data/jmsardain/JetCalib/Akt4EMTopo.topo-cluster.root", "READ");
    TTree *inputTree = (TTree*) inputFile->Get("ClusterTree");

    TFile *outputFile = new TFile("output.root", "RECREATE");

    TTree* outputTree = inputTree->CloneTree(-1, "fast");


    Float_t clusterPt, clusterEta, clusterPhi;
    Float_t jetRawPt, jetRawEta, jetRawPhi;

    // Create new variable for the sum branch
    Double_t zT, zL, zRel;

    // Set branch addresses for existing branches
    inputTree->SetBranchAddress("clusterPt", &clusterPt);
    inputTree->SetBranchAddress("clusterEta", &clusterEta);
    inputTree->SetBranchAddress("clusterPhi", &clusterPhi);

    inputTree->SetBranchAddress("jetRawPt", &jetRawPt);
    inputTree->SetBranchAddress("jetRawEta", &jetRawEta);
    inputTree->SetBranchAddress("jetRawPhi", &jetRawPhi);
    // Create a new branch for the sum
    TBranch* b_zT   = outputTree->Branch("zT", &zT, "zT/D");
    TBranch* b_zL   = outputTree->Branch("zL", &zL, "zL/D");
    TBranch* b_zRel = outputTree->Branch("zRel", &zRel, "zRel/D");

    // Loop over entries in the input tree
    Long64_t nEntries = inputTree->GetEntries();
    std::cout << " " << nEntries << std::endl;
    Long64_t nEntries_wCut = 0;

    for (Long64_t i = 0; i < nEntries; ++i) {
        // Get entry from the input tree
        inputTree->GetEntry(i);

        if(i%100000==0) std::cout << "Entry #" << i << std::endl;
        // if (i>100000) break;
        if(TMath::Abs(jetRawEta) < 1) continue;
        nEntries_wCut++;
        // Calculate the sum and assign it to the new branch
        TVector3 cluster_vec;
        cluster_vec.SetPtEtaPhi(clusterPt, clusterEta, clusterPhi);

        TVector3 jet_vec;
        jet_vec.SetPtEtaPhi(jetRawPt, jetRawEta, jetRawPhi);

        zT = cluster_vec.Pt()/jet_vec.Pt();
        zL = cluster_vec.Dot(jet_vec)/jet_vec.Mag2();
        zRel = (cluster_vec.Cross(jet_vec)).Mag()/jet_vec.Mag2();

        // Fill the output tree
        b_zT->Fill();
        b_zL->Fill();
        b_zRel->Fill();
    }
    std::cout << " " << nEntries_wCut << std::endl;
    // Write the output tree to the output file
    outputFile->Write();
    // Close both files
    outputFile->Close();
}
