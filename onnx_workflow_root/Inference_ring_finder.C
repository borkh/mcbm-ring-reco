#include <iostream>
#if !defined(__CLING__)
#include "onnxruntime_cxx_api.h"
#endif


void Inference_ring_finder(){

    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ring_finder"};
    Ort::Session session(env, "model.onnx", Ort::SessionOptions(nullptr));

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"reshape"};

    std::array<float, 72*32> input_image{};
    std::ifstream file("img.csv");
    std::string line;
    int i{};
    while(std::getline(file, line)){
        std::stringstream lineStream(line);
        std::string cell;
        while(std::getline(lineStream, cell, ',')){
            input_image[i] = std::stof(cell);
            i++;
        }
    }

    std::array<int64_t, 4> input_shape{1, 72, 32, 1};
    std::array<int64_t, 2> output_shape{5, 5};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
    
    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* intarr = output_tensor.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {intarr, intarr + 25};
    for(int i{}; i < output_tensor_values.size(); i++){
        std::cout << output_tensor_values[i] << std::endl;
    }

}