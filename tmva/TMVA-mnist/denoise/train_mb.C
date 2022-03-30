#include <iostream>
#include <vector>

#include "TMVA/DNN/DeepNet.h"
#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "RConfigure.h"   // for definition of R__HAS_CUDNN
#include "TMVA/Tools.h"
#include "TMVA/Configurable.h"
#include "TMVA/IMethod.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DLMinimizers.h"
#include "TMVA/DNN/Optimizer.h"
#include "TMVA/DNN/SGD.h"
#include "TMVA/DNN/Adam.h"
#include "TMVA/DNN/Adagrad.h"
#include "TMVA/DNN/RMSProp.h"
#include "TMVA/DNN/Adadelta.h"
#include "TMVA/Timer.h"
#include "TMVA/Tools.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TRandom2.h"
#include "TStopwatch.h"
#ifdef R__HAS_TMVAGPU
//#include "TMVA/DNN/Architectures/Cuda.h" //cuda is ~4.5x slower than cudnn! and saving layer weights bugs ?!
#ifdef R__HAS_CUDNN
#include "TMVA/DNN/Architectures/TCudnn.h"
#endif
#endif

using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;
using TMVA::DNN::EOptimizer;

using Architecture_t = TMVA::DNN::TCudnn<Float_t>;
using Scalar_t = typename Architecture_t::Scalar_t;
using Layer_t = TMVA::DNN::VGeneralLayer<Architecture_t>;
using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t, Layer_t>;
using TMVAInput_t =  std::tuple<const std::vector<TMVA::Event *> &, const TMVA::DataSetInfo &>;
using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

std::vector<TMVA::Event *> loadEvents();
TMVA::DataSetInfo &getDataSetInfo();
void SaveModelToXML(void * , DeepNet_t &);

void train_mb(){
    
    TMVA::Tools::Instance();
    

    Architecture_t::SetRandomSeed(10);

    // load data
    size_t nTrainingSamples(9000);
    size_t nValidationSamples(1000);
    const std::vector<TMVA::Event *> allData = loadEvents(); //TODO: free allData memory
    const std::vector<TMVA::Event *> eventCollectionTraining{allData.begin(), allData.begin() + nTrainingSamples};
    const std::vector<TMVA::Event *> eventCollectionValidation{allData.begin() + nTrainingSamples, allData.end()};

    std::cout << "load complete" << std::endl;
    
    
    /* For building a CNN one needs to define
    -  Input Layout :  number of channels (in this case = 1)  | image height | image width
    -  Batch Layout :  batch size | number of channels | image size = (height*width) */

    size_t nThreads = 1; 
    size_t batchSize = 10;
    size_t inputDepth  = 1;
    size_t inputHeight = 28;
    size_t inputWidth  = 28;
    size_t batchDepth  = batchSize; 
    size_t batchHeight = 1;
    size_t batchWidth  = inputHeight*inputWidth; 
    ELossFunction J    = ELossFunction::kMeanSquaredError;
    EInitialization I  = EInitialization::kUniform;
    ERegularization R    = ERegularization::kNone;
    EOptimizer O         = EOptimizer::kAdam;
    Scalar_t weightDecay = 0.0; 
    
    DeepNet_t deepNet(batchSize, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R, weightDecay);
    //DeepNet_t fNet(1, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, J, I, R, weightDecay);
    TConvLayer<Architecture_t> *convLayer1 = deepNet.AddConvLayer(8, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    /* TConvLayer<Architecture_t> *convLayer2 = deepNet.AddConvLayer(16, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    TConvLayer<Architecture_t> *convLayer3 = deepNet.AddConvLayer(32, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    TConvLayer<Architecture_t> *convLayer4 = deepNet.AddConvLayer(32, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    TConvLayer<Architecture_t> *convLayer5 = deepNet.AddConvLayer(16, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu);
    TConvLayer<Architecture_t> *convLayer6 = deepNet.AddConvLayer(8, 3, 3, 1, 1, 1, 1, EActivationFunction::kRelu); */
    TConvLayer<Architecture_t> *convLayer7 = deepNet.AddConvLayer(1, 3, 3, 1, 1, 1, 1, EActivationFunction::kSigmoid);
    TReshapeLayer<Architecture_t> *reshape1 = deepNet.AddReshapeLayer(0, 0, 0, 1);

    /* TDenseLayer<Architecture_t> *denseLayer1 = deepNet.AddDenseLayer(20, EActivationFunction::kRelu, 0.);
    TDenseLayer<Architecture_t> *denseLayer2 = deepNet.AddDenseLayer(28*28, EActivationFunction::kRelu, 0.); */
    deepNet.Initialize();

    std::cout  << "*****   Deep Learning Network  *****" << std::endl;
    deepNet.Print();


    // Loading the training and validation datasets
    TMVA::DataSetInfo dsix;
    for(int i{0}; i < 28*28; i++){
        TString in1; in1.Form("in%d", i);
        TString in2; in2.Form("input pixel %d", i);
        dsix.AddVariable(in1, in2, "", 0., 1.);

        TString out1; out1.Form("out%d", i);
        TString out2; out2.Form("output pixel %d", i);
        dsix.AddTarget(out1, out2, "", 0., 1.);
    }
    TMVAInput_t trainingTuple = std::tie(eventCollectionTraining, dsix); //TODO: getDataSetInfo()
    TensorDataLoader_t trainingData(trainingTuple, nTrainingSamples, batchSize,
                                    {inputDepth, inputHeight, inputWidth},
                                    {deepNet.GetBatchDepth(), deepNet.GetBatchHeight(), deepNet.GetBatchWidth()} ,
                                    deepNet.GetOutputWidth(), nThreads);

    TMVAInput_t validationTuple = std::tie(eventCollectionValidation, dsix);
    TensorDataLoader_t validationData(validationTuple, nValidationSamples, batchSize,
                                    {inputDepth, inputHeight, inputWidth},
                                    { deepNet.GetBatchDepth(),deepNet.GetBatchHeight(), deepNet.GetBatchWidth()} ,
                                    deepNet.GetOutputWidth(), nThreads);

    std::cout << "training and validation dataset created !" << std::endl;
    
    Double_t minValError = 0.0;
    for (auto batch : validationData) {
        auto inputTensor = batch.GetInput();
        auto outputMatrix = batch.GetOutput();
        auto weights = batch.GetWeights();
/*         Architecture_t::PrintTensor(batch.GetInput(),"input tensor",true);
        typename Architecture_t::Tensor_t tOut(batch.GetOutput());
        typename Architecture_t::Tensor_t tW(batch.GetWeights());
        Architecture_t::PrintTensor(tOut,"label tensor",true)   ;
        Architecture_t::PrintTensor(tW,"weight tensor",true)  ; */
        deepNet.Forward(inputTensor, false);
        minValError += deepNet.Loss(inputTensor, outputMatrix, weights, false, false);
    }
    minValError /= (Double_t)(nValidationSamples / batchSize);
    std::cout << "initial valLoss: "<< minValError << std::endl;
        
    std::unique_ptr<TMVA::DNN::VOptimizer<Architecture_t, Layer_t, DeepNet_t>> optimizer;
    optimizer = std::unique_ptr<TMVA::DNN::TAdam<Architecture_t, Layer_t, DeepNet_t>>(
                new TMVA::DNN::TAdam<Architecture_t, Layer_t, DeepNet_t>(
                deepNet, 0.001, 0.9, 0.999, 1e-7));


    size_t batchesInEpoch = nTrainingSamples / deepNet.GetBatchSize();
    size_t epoch(0);
    size_t maxEpoch(2);
    TMVA::RandomGenerator<TRandom3> rng(123); 
    std::cout  << "*****  Starting training loop  *****" << std::endl;
    while(epoch < maxEpoch){
        TStopwatch timer;
        timer.Start();
        trainingData.Shuffle(rng);
        for (size_t i = 0; i < batchesInEpoch; ++i ){
            //std::cin.get();
            auto my_batch = trainingData.GetTensorBatch();
            deepNet.Forward(my_batch.GetInput(), false);
            deepNet.Backward(my_batch.GetInput(), my_batch.GetOutput(), my_batch.GetWeights());
            optimizer->IncrementGlobalStep();
            optimizer->Step();
        }
        
        // Compute training error.
        Double_t trainError = 0.;
        for (auto batch : trainingData){
            auto inputTensor = batch.GetInput();
            auto outputMatrix = batch.GetOutput();
            auto weights = batch.GetWeights();
            trainError += deepNet.Loss(inputTensor, outputMatrix, weights, false, false);
        }
        trainError /= (Double_t)(nTrainingSamples / batchSize);
        
        // Compute validation error.
        Double_t valError = 0.;
        for (auto batch : validationData){
            auto inputTensor = batch.GetInput();
            auto outputMatrix = batch.GetOutput();
            auto weights = batch.GetWeights();
            valError += deepNet.Loss(inputTensor, outputMatrix, weights, false, false);
        }
        valError /= (Double_t)(nValidationSamples / batchSize);
        timer.Stop();
        Double_t dt = timer.RealTime();
    
        std::cout   << "Epoch: " << epoch + 1 
                    << std::setw(12) << "  Time: " << dt <<"s" 
                    << std::setw(12) << "  Loss: " << trainError 
                    << std::setw(12) << "  valLoss: " << valError << "\n";
        
        epoch++;
    }

    // Save Model ......
    TString xmlName("model.xml");
    void* doc       = TMVA::gTools().xmlengine().NewDoc();
    void* rootnode  = TMVA::gTools().AddChild(0, "MethodSetup", "", true);
    TMVA::gTools().xmlengine().DocSetRootElement(doc, rootnode);
    TMVA::gTools().AddAttr(rootnode, "Method", "test_mb");
    SaveModelToXML(rootnode, deepNet); //FIXME:
    TMVA::gTools().xmlengine().SaveDoc(doc, xmlName);
    TMVA::gTools().xmlengine().FreeDoc(doc);
 

}


std::vector<TMVA::Event *> loadEvents(){ //target is denoisy input

    TStopwatch timer;
    timer.Start();
    std::cout << "loading Events from file ... \n" ; 
    TFile *file = TFile::Open("../mnist.root"); 
    TTree *tree = (TTree*)file->Get("train"); 
    Float_t x0[784];
    tree->SetBranchAddress("x", x0);
    std::vector<TMVA::Event*> allData;
    
    TRandom2 *rand = new TRandom2(123); //fixed seed
    
    Long64_t nofEntries = tree->GetEntries();
    for(Long64_t i=0; i < 10000; i++){
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
    timer.Stop();
    Double_t dt = timer.RealTime();
    std::cout << "loading finished ! \n" ; 
    std::cout << "time for loading Events  " << dt << "s\n";  
    return allData;
}

/* TMVA::DataSetInfo &getDataSetInfo(){
    TMVA::DataSetInfo* dsi;
    for(int i{0}; i < 28*28; i++){
        TString in1; in1.Form("in%d", i);
        TString in2; in2.Form("input pixel %d", i);
        dsi->AddVariable(in1, in2, "", 0., 1.);

        TString out1; out1.Form("out%d", i);
        TString out2; out2.Form("output pixel %d", i);
        dsi->AddTarget(out1, out2, "", 0., 1.);
    } 
    return *dsi;
} */

void SaveModelToXML(void * parent, DeepNet_t &net){ 
    //TMVA::Tools::Instance();
    auto &xmlEngine = TMVA::gTools().xmlengine();
    void *nn = xmlEngine.NewChild(parent, 0, "Weights");
    // Deep Net specific info
    Int_t depth = net.GetDepth();

    Int_t inputDepth = net.GetInputDepth();
    Int_t inputHeight = net.GetInputHeight();
    Int_t inputWidth = net.GetInputWidth();

    Int_t batchSize = net.GetBatchSize();

    Int_t batchDepth = net.GetBatchDepth();
    Int_t batchHeight = net.GetBatchHeight();
    Int_t batchWidth = net.GetBatchWidth();

    char lossFunction = static_cast<char>(net.GetLossFunction());
    char initialization = static_cast<char>(net.GetInitialization());
    char regularization = static_cast<char>(net.GetRegularization());

    Double_t weightDecay = net.GetWeightDecay();
    //char outputFunction = static_cast<char>(net.GetOutputFunction()); // ??? OutputFunction="S" 
    // Add attributes to the parent node
    xmlEngine.NewAttr(nn, 0, "NetDepth", TMVA::gTools().StringFromInt(depth));

    xmlEngine.NewAttr(nn, 0, "InputDepth", TMVA::gTools().StringFromInt(inputDepth));
    xmlEngine.NewAttr(nn, 0, "InputHeight", TMVA::gTools().StringFromInt(inputHeight));
    xmlEngine.NewAttr(nn, 0, "InputWidth", TMVA::gTools().StringFromInt(inputWidth));

    xmlEngine.NewAttr(nn, 0, "BatchSize", TMVA::gTools().StringFromInt(batchSize));
    xmlEngine.NewAttr(nn, 0, "BatchDepth", TMVA::gTools().StringFromInt(batchDepth));
    xmlEngine.NewAttr(nn, 0, "BatchHeight", TMVA::gTools().StringFromInt(batchHeight));
    xmlEngine.NewAttr(nn, 0, "BatchWidth", TMVA::gTools().StringFromInt(batchWidth));

    xmlEngine.NewAttr(nn, 0, "LossFunction", TString(lossFunction));
    xmlEngine.NewAttr(nn, 0, "Initialization", TString(initialization));
    xmlEngine.NewAttr(nn, 0, "Regularization", TString(regularization));
    //xmlEngine.NewAttr(nn, 0, "OutputFunction", TString(outputFunction));

    TMVA::gTools().AddAttr(nn, "WeightDecay", weightDecay);

    for (Int_t i = 0; i < depth; i++){
        net.GetLayerAt(i)->AddWeightsXMLTo(nn);
    }
}