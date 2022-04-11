#include <iostream>

#if !defined(__CLING__)
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
#include "TMVA/DNN/Architectures/Cpu/CpuTensor.h"
#ifdef R__HAS_TMVAGPU
//#include "TMVA/DNN/Architectures/Cuda.h" 
#ifdef R__HAS_CUDNN
//#include "TMVA/DNN/Architectures/TCudnn.h"
#endif
#endif
#endif


using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;
using TMVA::DNN::EOptimizer;

using Architecture_t = TMVA::DNN::TCpu<Float_t>;
using Scalar_t = typename Architecture_t::Scalar_t;
using Matrix_t = typename Architecture_t::Matrix_t;
using Tensor_t = typename Architecture_t::Tensor_t;
using Layer_t = TMVA::DNN::VGeneralLayer<Architecture_t>;
using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t, Layer_t>;
using TMVAInput_t =  std::tuple<const std::vector<TMVA::Event *> &, const TMVA::DataSetInfo &>;
using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

DeepNet_t* ReadModelFromXML(TString);

void predict_lowlevel(){

    // load data
    TString fileXML = "./dataset/weights/ring_reco_tmvaDL.weights.xml";
    DeepNet_t* fNet = ReadModelFromXML(fileXML);
    Tensor_t input = TCpuTensor<Float_t>(1, 1, 28, 28);
    input.Zero();
    Matrix_t pred = TCpuMatrix<Float_t>(1, 10);
    fNet->Prediction(pred, input, TMVA::DNN::EOutputFunction::kIdentity);
    for(int i{}; i < 10; i++){
        std::cout << pred(0,i) << std::endl;
    }
 
}


DeepNet_t* ReadModelFromXML(TString xmlFile){

    void* model = TMVA::gTools().xmlengine().ParseFile(xmlFile);
    void* rootnode = TMVA::gTools().xmlengine().DocGetRootElement(model);
    //ReadModelFromXML(rootnode, deepNet);
    auto netXML = TMVA::gTools().GetChild(rootnode, "Weights");
    if (!netXML){
        netXML = rootnode;
    }

    size_t netDepth;
    TMVA::gTools().ReadAttr(netXML, "NetDepth", netDepth);

    size_t inputDepth, inputHeight, inputWidth;
    TMVA::gTools().ReadAttr(netXML, "InputDepth", inputDepth);
    TMVA::gTools().ReadAttr(netXML, "InputHeight", inputHeight);
    TMVA::gTools().ReadAttr(netXML, "InputWidth", inputWidth);
    //std::cout << "inputDepth" << inputDepth << std::endl;
    size_t batchSize, batchDepth, batchHeight, batchWidth;
    TMVA::gTools().ReadAttr(netXML, "BatchSize", batchSize);
    // use always batchsize = 1
    //batchSize = 1;
    TMVA::gTools().ReadAttr(netXML, "BatchDepth", batchDepth);
    TMVA::gTools().ReadAttr(netXML, "BatchHeight", batchHeight);
    TMVA::gTools().ReadAttr(netXML, "BatchWidth",  batchWidth);

    char lossFunctionChar;
    TMVA::gTools().ReadAttr(netXML, "LossFunction", lossFunctionChar);
    char initializationChar;
    TMVA::gTools().ReadAttr(netXML, "Initialization", initializationChar);
    char regularizationChar;
    TMVA::gTools().ReadAttr(netXML, "Regularization", regularizationChar);
    //char outputFunctionChar;
    //TMVA::gTools().ReadAttr(netXML, "OutputFunction", outputFunctionChar);
    double weightDecay;
    TMVA::gTools().ReadAttr(netXML, "WeightDecay", weightDecay);

    // ---- create deepnet ----
    DeepNet_t* fNet = new DeepNet_t(1, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, 
                            static_cast<ELossFunction>(lossFunctionChar),
                            static_cast<EInitialization>(initializationChar),
                            static_cast<ERegularization>(regularizationChar),
                            weightDecay);
    
    auto layerXML = TMVA::gTools().xmlengine().GetChild(netXML);

    // loop on the layer and add them to the network
    for (size_t i = 0; i < netDepth; i++) {

        TString layerName = TMVA::gTools().xmlengine().GetNodeName(layerXML);

        // case of dense layer
        if (layerName == "DenseLayer") {

            // read width and activation function and then we can create the layer
            size_t width = 0;
            TMVA::gTools().ReadAttr(layerXML, "Width", width);

            // Read activation function.
            TString funcString;
            TMVA::gTools().ReadAttr(layerXML, "ActivationFunction", funcString);
            EActivationFunction func = static_cast<EActivationFunction>(funcString.Atoi());


            fNet->AddDenseLayer(width, func, 0.0); // no need to pass dropout probability

        }
        // Convolutional Layer
        else if (layerName == "ConvLayer") {

            // read width and activation function and then we can create the layer
            size_t depth = 0;
            TMVA::gTools().ReadAttr(layerXML, "Depth", depth);
            size_t fltHeight, fltWidth = 0;
            size_t strideRows, strideCols = 0;
            size_t padHeight, padWidth = 0;
            TMVA::gTools().ReadAttr(layerXML, "FilterHeight", fltHeight);
            TMVA::gTools().ReadAttr(layerXML, "FilterWidth", fltWidth);
            TMVA::gTools().ReadAttr(layerXML, "StrideRows", strideRows);
            TMVA::gTools().ReadAttr(layerXML, "StrideCols", strideCols);
            TMVA::gTools().ReadAttr(layerXML, "PaddingHeight", padHeight);
            TMVA::gTools().ReadAttr(layerXML, "PaddingWidth", padWidth);

            // Read activation function.
            TString funcString;
            TMVA::gTools().ReadAttr(layerXML, "ActivationFunction", funcString);
            EActivationFunction actFunction = static_cast<EActivationFunction>(funcString.Atoi());


            fNet->AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
                                padHeight, padWidth, actFunction);

        }

        // MaxPool Layer
        else if (layerName == "MaxPoolLayer") {

            // read maxpool layer info
            size_t filterHeight, filterWidth = 0;
            size_t strideRows, strideCols = 0;
            TMVA::gTools().ReadAttr(layerXML, "FilterHeight", filterHeight);
            TMVA::gTools().ReadAttr(layerXML, "FilterWidth", filterWidth);
            TMVA::gTools().ReadAttr(layerXML, "StrideRows", strideRows);
            TMVA::gTools().ReadAttr(layerXML, "StrideCols", strideCols);

            fNet->AddMaxPoolLayer(filterHeight, filterWidth, strideRows, strideCols);
        }
        // Reshape Layer
        else if (layerName == "ReshapeLayer") {

            // read reshape layer info
            size_t depth, height, width = 0;
            TMVA::gTools().ReadAttr(layerXML, "Depth", depth);
            TMVA::gTools().ReadAttr(layerXML, "Height", height);
            TMVA::gTools().ReadAttr(layerXML, "Width", width);
            int flattening = 0;
            TMVA::gTools().ReadAttr(layerXML, "Flattening",flattening );

            fNet->AddReshapeLayer(depth, height, width, flattening);

        }
        // RNN Layer
        else if (layerName == "RNNLayer") {

            // read RNN layer info
            size_t  stateSize,inputSize, timeSteps = 0;
            int rememberState= 0;
            int returnSequence = 0;
            TMVA::gTools().ReadAttr(layerXML, "StateSize", stateSize);
            TMVA::gTools().ReadAttr(layerXML, "InputSize", inputSize);
            TMVA::gTools().ReadAttr(layerXML, "TimeSteps", timeSteps);
            TMVA::gTools().ReadAttr(layerXML, "RememberState", rememberState );
            TMVA::gTools().ReadAttr(layerXML, "ReturnSequence", returnSequence);

            fNet->AddBasicRNNLayer(stateSize, inputSize, timeSteps, rememberState, returnSequence);

        }
        // BatchNorm Layer
        else if (layerName == "BatchNormLayer") {
            // use some dammy value which will be overwrittem in BatchNormLayer::ReadWeightsFromXML
            fNet->AddBatchNormLayer(0., 0.0);
        }
        // read weights and biases
        //fNet->GetLayers().back()->Print();
        fNet->GetLayers().back()->ReadWeightsFromXML(layerXML);

        // read next layer
        layerXML = TMVA::gTools().GetNextChild(layerXML);
    }
    std::cout  << "*****   Deep Learning Network  *****" << std::endl;
    fNet->Print();

    TMVA::gTools().xmlengine().FreeDoc(model);
    return fNet;
}
