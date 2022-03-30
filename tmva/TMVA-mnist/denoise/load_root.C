#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TMVA/Event.h"
#include "TRandom2.h"
#include "TStopwatch.h"


std::vector<TMVA::Event *> loadEvents();
const std::vector<TMVA::Event *> loadEventsClass();

int load_root(){
    TStopwatch timer;
    timer.Start();
    const std::vector<TMVA::Event *> allData = loadEvents();
    timer.Stop();

    Double_t dt = timer.RealTime();
    std::cout << "time for loading Events  " << dt << "s\n";  
    
    return 0;
}


std::vector<TMVA::Event *> loadEvents(){ //target is denoisy input
    std::cout << "loading Events from file ... \n" ; 
    TFile *file = TFile::Open("mnist.root"); 
    TTree *tree = (TTree*)file->Get("train"); 
    Float_t x0[784];
    tree->SetBranchAddress("x", x0);
    std::vector<TMVA::Event*> allData;
    
    TRandom2 *rand = new TRandom2(123); //fixed seed
    
    Long64_t nofEntries = tree->GetEntries();
    for(Long64_t i=0; i < nofEntries; i++){
        tree->GetEntry(i);
        std::vector<Float_t> input; 
        std::vector<Float_t> target; 
        
        for(int j=0; j < 784; j++){
            target.push_back(x0[j]);
            //add noise
            Float_t tar = x0[j] + rand->Rndm() * 0.2;
            if(tar > 1.) tar = 1. ;
            input.push_back(tar);
        }
        

        TMVA::Event* ev = new TMVA::Event(input, target);
        allData.push_back(ev); 
    }
    std::cout << "loading finished ! \n" ; 
    return allData;
}

const std::vector<TMVA::Event *> loadEventsClass(){ //target is class
    TFile *file = TFile::Open("mnist.root");
    TTree *tree = (TTree*)file->Get("train");

    Float_t x0[784];
    Float_t y0;
    tree->SetBranchAddress("x", x0);
    tree->SetBranchAddress("y", &y0);
    
    std::vector<TMVA::Event*> allData;
    
    Long64_t nofEntries = tree->GetEntries();
    for(Long64_t i=0; i < nofEntries; i++){
        tree->GetEntry(i);
        std::vector<Float_t> input; //single event input value
        std::vector<Float_t> target; //single event target value
        target.push_back(y0);
        for(int j=0; j < 784; j++){
            input.push_back(x0[j]);
        }
        
        TMVA::Event* ev = new TMVA::Event(input, target);
        allData.push_back(ev);
    }

    return allData;
}