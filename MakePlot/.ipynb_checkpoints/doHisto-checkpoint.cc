#include <iostream>
#include "TString.h"
#include "HistoMaker.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"
#include <string>
#include "AtlasStyle.h"
#include "AtlasLabels.h"
#include "AtlasUtils.h"

using namespace std ;

// int main(int argc, char* argv[]){
int main(){
	SetAtlasStyle();
	// std::string name(argv[1]);
	// data
	// TString theLink = "data/user.zhcui.34376381._000001.ANALYSIS.root";
	// TString dataOrMC = TString(name.c_str());
	TString theLink = "/data/jmsardain/JetCalib/Akt4EMTopo.topo-cluster.root";

	TChain * myChain = new TChain( "ClusterTree" ) ;

	cout << theLink << endl ;

	myChain->Add( theLink );
	cout << "my chain = " << myChain->GetEntries() << endl ;

	// gROOT->LoadMacro("/afs/cern.ch/work/j/jzahredd/EarlyRun3/HistoMaker.C+");

	HistoMaker * myAnalysis ;
	myAnalysis =  new HistoMaker( myChain ) ;
	myAnalysis->Loop();
	return 0;

}