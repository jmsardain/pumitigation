//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Tue Jan 23 14:00:13 2024 by ROOT version 6.30/02
// from TTree aTree/
// found on file: /home/jmsardain/JetCalib/PUMitigation/latest/pumitigation/ckpts/EdgeConv/out_EdgeConv_.root
//////////////////////////////////////////////////////////

#ifndef Plot_h
#define Plot_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class Plot {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Float_t         jetCnt;
   Float_t         eventNumber;
   Float_t         jetRawE;
   Float_t         jetRawPt;
   Float_t         truthJetE;
   Float_t         truthJetPt;
   Float_t         jetCalE;
   Float_t         clusterE;
   Float_t         clusterEta;
   Float_t         clusterPhi;
   Float_t         clusterECalib;
   Float_t         cluster_ENG_CALIB_TOT;
   Float_t         labels;
   Float_t         score;
   Float_t         clusterEexp;
   Float_t         Scores;

   // List of branches
   TBranch        *b_jetCnt;   //!
   TBranch        *b_eventNumber;   //!
   TBranch        *b_jetRawE;   //!
   TBranch        *b_jetRawPt;   //!
   TBranch        *b_truthJetE;   //!
   TBranch        *b_truthJetPt;   //!
   TBranch        *b_jetCalE;   //!
   TBranch        *b_clusterE;   //!
   TBranch        *b_clusterEta;   //!
   TBranch        *b_clusterPhi;   //!
   TBranch        *b_clusterECalib;   //!
   TBranch        *b_cluster_ENG_CALIB_TOT;   //!
   TBranch        *b_labels;   //!
   TBranch        *b_score;   //!
   TBranch        *b_clusterEexp;   //!
   TBranch        *b_Scores;   //!

   Plot(TTree *tree=0);
   virtual ~Plot();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef Plot_cxx
Plot::Plot(TTree *tree) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/home/jmsardain/JetCalib/PUMitigation/latest/pumitigation/ckpts/EdgeConv/out_EdgeConv_.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/home/jmsardain/JetCalib/PUMitigation/latest/pumitigation/ckpts/EdgeConv/out_EdgeConv_.root");
      }
      f->GetObject("aTree",tree);

   }
   Init(tree);
}

Plot::~Plot()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t Plot::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t Plot::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void Plot::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("jetCnt", &jetCnt, &b_jetCnt);
   fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
   fChain->SetBranchAddress("jetRawE", &jetRawE, &b_jetRawE);
   fChain->SetBranchAddress("jetRawPt", &jetRawPt, &b_jetRawPt);
   fChain->SetBranchAddress("truthJetE", &truthJetE, &b_truthJetE);
   fChain->SetBranchAddress("truthJetPt", &truthJetPt, &b_truthJetPt);
   fChain->SetBranchAddress("jetCalE", &jetCalE, &b_jetCalE);
   fChain->SetBranchAddress("clusterE", &clusterE, &b_clusterE);
   fChain->SetBranchAddress("clusterEta", &clusterEta, &b_clusterEta);
   fChain->SetBranchAddress("clusterPhi", &clusterPhi, &b_clusterPhi);
   fChain->SetBranchAddress("clusterECalib", &clusterECalib, &b_clusterECalib);
   fChain->SetBranchAddress("cluster_ENG_CALIB_TOT", &cluster_ENG_CALIB_TOT, &b_cluster_ENG_CALIB_TOT);
   fChain->SetBranchAddress("labels", &labels, &b_labels);
   fChain->SetBranchAddress("score", &score, &b_score);
   fChain->SetBranchAddress("clusterEexp", &clusterEexp, &b_clusterEexp);
   fChain->SetBranchAddress("Scores", &Scores, &b_Scores);
   Notify();
}

Bool_t Plot::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void Plot::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t Plot::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef Plot_cxx
