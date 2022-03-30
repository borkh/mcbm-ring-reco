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
    TFile* output = TFile::Open("mcbm_pred.root", "RECREATE");
    TMVA::Factory* factory = new TMVA::Factory(
        "ring_reco", output,
        "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug"
    );

    // load data
    TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");
    TFile* data = TFile::Open("mcbm_nonorm.root");
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
    TString batchLayoutString = "BatchLayout=64|1|2304:";
    TString inputLayoutString = "InputLayout=1|72|32:";
    TString layoutString =
    "Layout=CONV|8|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,CONV|16|3|3|1|1|1|1|RELU,CONV|32|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,MAXPOOL|2|2|1|1,RESHAPE|FLAT,DENSE|256|RELU,DENSE|15|RELU:";
    TString trainingString = "TrainingStrategy=MaxEpochs=20,BatchSize=64,Optimizer=ADAM,LearningRate=1e-3:";
    TString cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Debug:Architecture=GPU";
    TString options = batchLayoutString + inputLayoutString + layoutString + trainingString + cnnOptions;
    // Book method
    factory->BookMethod(dataloader, TMVA::Types::kDL, "tmvaDL", options);
    TMVA::MethodDL* method = dynamic_cast<TMVA::MethodDL*>(factory->GetMethod(dataloader->GetName(), "tmvaDL"));
    method->Train();
    method->WriteStateToFile();
}
