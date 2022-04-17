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
    TFile* data = TFile::Open("datasets/32k.root");
    dataloader->AddRegressionTree((TTree*)data->Get("train"), 1.0);     

    for(int i{} ; i < 72*32 ; i++){
        dataloader->AddVariable("x[" + std::to_string(i) + "]", "x_" + std::to_string(i), "" );
    }
    // or AddVariableArray
    for(int i{} ; i < 15 ; i++){
        dataloader->AddTarget("y[" + std::to_string(i) + "]", "y_" + std::to_string(i), "" );
    }

    dataloader->PrepareTrainingAndTestTree(TCut(""), "SplitMode=Random:NormMode=None:V:!Correlations:!CalcCorrelations");

    // define model & options
    TString batchLayoutString = "BatchLayout=32|1|2304:";
    TString inputLayoutString = "InputLayout=1|72|32:";

    // define model architecture
    TString l 	   = "Layout=";
    TString block1 = "CONV|16|3|3|1|1|1|1|RELU,CONV|16|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,";
    TString block2 = "CONV|32|3|3|1|1|1|1|RELU,CONV|32|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,";
    TString block3 = "CONV|64|3|3|1|1|1|1|RELU,CONV|64|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,";
    TString out    = "RESHAPE|FLAT,DENSE|1024|RELU,DENSE|15|RELU:";

    TString lr_schedule = ":";

    TString layoutString = l + block1 + block2 + block3 + out;
    TString trainingString = "TrainingStrategy=MaxEpochs=10,BatchSize=32,TestRepetitions=1,Optimizer=ADAM,LearningRate=1e-3:";
    TString cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Verbose:Architecture=GPU";
    TString options = batchLayoutString + inputLayoutString + layoutString + trainingString + cnnOptions; //+ lr_schedule;
    // Book method
    factory->BookMethod(dataloader, TMVA::Types::kDL, "tmvaDL", options);
    TMVA::MethodDL* method = dynamic_cast<TMVA::MethodDL*>(factory->GetMethod(dataloader->GetName(), "tmvaDL"));
    method->Train();

    method->WriteStateToFile();
}
