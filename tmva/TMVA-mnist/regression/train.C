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
    TFile* output = TFile::Open("TMVA.root", "RECREATE");
    TMVA::Factory* factory = new TMVA::Factory(
        "MNIST", output,
        "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug"
    );

    // load data
    TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");
    TFile* data = TFile::Open("mnist0.root");
    //dataloader->AddRegressionTarget(); //FIXME:
    dataloader->AddRegressionTree((TTree*)data->Get("train"), 1.0);     

    for(int i{} ; i < 28*28 ; i++){
        dataloader->AddVariable("x[" + std::to_string(i) + "]", "x_" + std::to_string(i), "" );
    }
    // or AddVariableArray
    for(int i{} ; i < 10 ; i++){
        dataloader->AddTarget("y[" + std::to_string(i) + "]", "y_" + std::to_string(i), "" );
    }

    dataloader->PrepareTrainingAndTestTree(TCut(""), "SplitMode=Random:NormMode=None:V:!Correlations:!CalcCorrelations");

    // define model & options
    TString batchLayoutString = "BatchLayout=100|1|784:";
    TString inputLayoutString = "InputLayout=1|28|28:";
    TString layoutString = "Layout=CONV|4|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,RESHAPE|FLAT,DENSE|16|RELU,DENSE|10|SIGMOID:";
    TString trainingString = "TrainingStrategy=MaxEpochs=10,BatchSize=100,Optimizer=ADAM,LearningRate=1e-3:";
    TString cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Debug:Architecture=GPU";
    TString options = batchLayoutString + inputLayoutString + layoutString + trainingString + cnnOptions;
    // Book method
    factory->BookMethod(dataloader, TMVA::Types::kDL, "tmvaDL", options);
    TMVA::MethodDL* method = dynamic_cast<TMVA::MethodDL*>(factory->GetMethod(dataloader->GetName(), "tmvaDL"));
    method->Train();
    method->WriteStateToFile();
    

}
