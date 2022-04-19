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
    TFile* data = TFile::Open("../data/200k.root");
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
    TString batchLayoutString = "BatchLayout=32|1|2304:";
    TString inputLayoutString = "InputLayout=1|72|32:";

    // define model architecture
    TString l 	   = "Layout=";
    TString block1 = "CONV|16|3|3|1|1|1|1|RELU,CONV|16|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,";
    TString block2 = "CONV|32|3|3|1|1|1|1|RELU,CONV|32|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,";
    TString block3 = "CONV|64|3|3|1|1|1|1|RELU,CONV|64|3|3|1|1|1|1|RELU,MAXPOOL|2|2|2|2,";
    TString out    = "RESHAPE|FLAT,DENSE|1024|RELU,DENSE|15|RELU:";

    TString dropout = "DropConfig=0.1+0.1+0.0+0.1+0.1+0.0+0.1+0.1+0.0+0.5+0.5+0.0,";
    //TString dropout = "DropConfig=0.05";

    TString lr_schedule = "0,0.0009994863;1,0.0009964071;2,0.0009902674;3,0.000981105;4,0.00096897647;5,0.0009539566;6,0.00093613786;7,0.00091563025;8,0.0008925602;9,0.0008670697;10,0.0008393163;11,0.0008094708;12,0.0007777173;13,0.00074425153;14,0.0007092799;15,0.00067301805;16,0.0006356895;17,0.00059752434;18,0.0005587579;19,0.0005196293;20,0.0004803796;21,0.0004412509;22,0.00040248438;23,0.00036431907;24,0.0003269903;25,0.00029072817;26,0.00025575634;27,0.00022229028;28,0.00019053643;29,0.00016069054;30,0.00013293662;31,0.000107445754;32,8.437512e-05;33,6.386695e-05;34,4.604769e-05;35,3.1027204e-05;36,1.8898094e-05;37,9.7351385e-06;38,3.594834e-06;39,5.150354e-07;40,0.0004997437;41,0.000498204;42,0.00049513427;43,0.000490553;44,0.00048448876;45,0.00047697884;46,0.00046806945;47,0.00045781565;48,0.00044628058;49,0.00043353537;50,0.0004196586;51,0.00040473585;52,0.00038885913;53,0.00037212626;54,0.00035464048;55,0.00033650955;56,0.00031784526;57,0.0002987627;58,0.00027937948;59,0.00025981513;60,0.00024019029;61,0.00022062592;62,0.00020124266;63,0.00018216002;64,0.00016349564;65,0.0001453646;66,0.00012787865;67,0.00011114564;68,9.526872e-05;69,8.034577e-05;70,6.646881e-05;71,5.3723375e-05;72,4.2188054e-05;73,3.1933974e-05;74,2.3024344e-05;75,1.5514102e-05;76,9.449547e-06;77,4.8680695e-06;78,1.7979171e-06;79,2.5801762e-07;80,0.00024987233;81,0.00024910254;82,0.0002475676;83,0.000245277;84,0.00024224489;85,0.00023848991;86,0.00023403524;87,0.0002289083;88,0.00022314079;89,0.00021676817;90,0.00020982981;91,0.0002023684;92,0.00019443006;93,0.00018606363;94,0.00017732073;95,0.00016825527;96,0.00015892313;97,0.00014938184;98,0.00013969024;99,0.00012990806;100,0.00012009563;101,0.000110313464;102,0.000100621844;103,9.108051e-05;104,8.174832e-05;105,7.26828e-05;106,6.393983e-05;107,5.557332e-05;108,4.7634858e-05;109,4.0173385e-05;110,3.3234905e-05;111,2.6862186e-05;112,2.109453e-05;113,1.596749e-05;114,1.1512673e-05;115,7.7575505e-06;116,4.725273e-06;117,2.434535e-06;118,8.994585e-07;119,1.2950883e-07;";

    //TString lr_schedule = "";

    TString layoutString = l + block1 + block2 + block3 + out;
    TString trainingString =
    "TrainingStrategy=MaxEpochs=120,BatchSize=32,TestRepetitions=1,Optimizer=ADAM,LearningRate=0.001,";
    TString cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Verbose:Architecture=CPU";
    TString options = batchLayoutString + inputLayoutString + layoutString +
    trainingString + dropout + lr_schedule + cnnOptions;
    // Book method
    factory->BookMethod(dataloader, TMVA::Types::kDL, "tmvaDL", options);
    TMVA::MethodDL* method = dynamic_cast<TMVA::MethodDL*>(factory->GetMethod(dataloader->GetName(), "tmvaDL"));
    method->Train();

    method->WriteStateToFile();
}
