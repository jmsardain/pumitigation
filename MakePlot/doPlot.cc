#include <iostream>
#include "TString.h"
#include "Plot.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"
#include <string>
#include "AtlasStyle.h"
#include "AtlasLabels.h"
#include "AtlasUtils.h"


using namespace std ;

int main(int argc, char* argv[]){
	SetAtlasStyle();
// int main(){
 	std::string name(argv[1]);
	// data
	// TString theLink = "data/user.zhcui.34376381._000001.ANALYSIS.root";
	TString dataOrMC = TString(name.c_str());
	// TString theLink = "/home/jmsardain/JetCalib/PUMitigation/latest/pumitigation/ckpts/EdgeConv/out_EdgeConv_.root";
	// TString theLink = "/home/jmsardain/JetCalib/PUMitigation/latest/pumitigation/ckpts/EdgeConv/out_EdgeConv_.root";

	TChain * myChain = new TChain( "aTree" ) ;

	cout << dataOrMC << endl ;

	myChain->Add( dataOrMC );
	cout << "my chain = " << myChain->GetEntries() << endl ;

	// gROOT->LoadMacro("/afs/cern.ch/work/j/jzahredd/EarlyRun3/HistoMaker.C+");

	Plot * myAnalysis ;
	myAnalysis =  new Plot( myChain ) ;
	myAnalysis->Loop();
	return 0;

}
