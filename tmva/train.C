#include <iostream>
#include <string>

#if !defined(__CLING__)
#include "TFile.h"
#include "TTree.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/MethodDL.h"
#endif

void train(){

    TMVA::Tools::Instance();
    ROOT::EnableImplicitMT(0);
    TFile* output = TFile::Open("mcbm_pred.root", "RECREATE");
    TMVA::Factory* factory = new TMVA::Factory(
        "ring_reco", output,
        "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Verbose"
    );

    // load data
    TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");
    TFile* data = TFile::Open("../data/10k.root");
    dataloader->AddRegressionTree((TTree*)data->Get("train"), 1.0, TMVA::Types::ETreeType::kTraining);     

    for(int i{} ; i < 72*32 ; i++){
        dataloader->AddVariable("x[" + std::to_string(i) + "]", "x_" + std::to_string(i), "" );
    }
    // or AddVariableArray
    for(int i{} ; i < 15 ; i++){
        dataloader->AddTarget("y[" + std::to_string(i) + "]", "y_" + std::to_string(i), "" );
    }

    dataloader->PrepareTrainingAndTestTree(TCut(""), "SplitMode=Random:NormMode=None:V:!Correlations:!CalcCorrelations");

    // define model & options
    TString batchLayoutString = "BatchLayout=256|1|2304:";
    TString inputLayoutString = "InputLayout=1|72|32:";

    // define model architecture
    TString l 	   = "Layout=";
    TString block1 = "CONV|64|3|3|1|1|1|1|RELU,BNORM,CONV|64|3|3|1|1|1|1|RELU,BNORM,MAXPOOL|2|2|2|2,";
    TString block2 = "CONV|128|3|3|1|1|1|1|RELU,BNORM,CONV|128|3|3|1|1|1|1|RELU,BNORM,MAXPOOL|2|2|2|2,";
    TString block3 = "CONV|256|3|3|1|1|1|1|RELU,BNORM,CONV|256|3|3|1|1|1|1|RELU,BNORM,MAXPOOL|2|2|2|2,";
    TString block4 = "CONV|512|3|3|1|1|1|1|RELU,BNORM,CONV|512|3|3|1|1|1|1|RELU,BNORM,MAXPOOL|3|2|2|2,";
    //TString block2 = "";
    //TString block3 = "";
    //TString block4 = "";
    TString out    = "RESHAPE|FLAT,BNORM,DENSE|64|RELU,BNORM,DENSE|15|RELU:";

    //TString dropout = "DropConfig=0.05";
    TString dropout = "";

    TString lr_schedule = "LearningRateSchedule=0,0.002501;1,0.005;2,0.0075;3,0.01;4,0.0075;5,0.005;6,0.002501;7,1e-06;8,0.001251;9,0.002501;10,0.003751;11,0.005;12,0.003751;13,0.002501;14,0.001251;15,1e-06;16,0.000626;17,0.001251;18,0.001876;19,0.002501;20,0.001876;21,0.001251;22,0.000626;23,1e-06,";

    TString layoutString = l + block1 + block2 + block3 + block4 + out;
    TString trainingString = "TrainingStrategy=MaxEpochs=24,BatchSize=256,TestRepetitions=1,Optimizer=SGD,LearningRate=0.0025,Momentum=0.9,";
    TString cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Verbose:Architecture=GPU";
    TString options = batchLayoutString + inputLayoutString + layoutString +
    trainingString + dropout + lr_schedule + cnnOptions;
    // Book method
    factory->BookMethod(dataloader, TMVA::Types::kDL, "tmvaDL", options);
    TMVA::MethodDL* method = dynamic_cast<TMVA::MethodDL*>(factory->GetMethod(dataloader->GetName(), "tmvaDL"));
    method->Train();

    method->WriteStateToFile();
}
