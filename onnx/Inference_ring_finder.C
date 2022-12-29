#if !defined(__CLING__)
#include "onnxruntime_cxx_api.h"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include <dirent.h>

using namespace std;

int number_of_png_files_in_dir(string dir_name) {
    DIR *dir;
    struct dirent *ent;
    int number_of_files = 0;
    if ((dir = opendir (dir_name.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            string file_name = ent->d_name;
            if (file_name.find(".png") != string::npos) {
                number_of_files++;
            }
        }
        closedir (dir);
    } else {
        perror ("");
        return EXIT_FAILURE;
    }
    return number_of_files;
}

void Inference_ring_finder() {

    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ring_finder"};
    Ort::Session session(env, "../models/model.onnx", Ort::SessionOptions(nullptr));

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"reshape"};

    string dir_name = "../data/test/X/";
    int number_of_files = number_of_png_files_in_dir(dir_name);


    for (int i = 0; i < number_of_files; i++) {
        array<float, 72*32> input_image{};

        int width = 32, height = 72, channels = 1;
        string filename = dir_name + to_string(i) + ".png";

        unsigned char* pixels = stbi_load(filename.c_str(), &width, &height, &channels, 0);


        for (int j{}; j < 72*32; j++) {
            input_image[j] = pixels[j] / 255.0;
        }

        ifstream file(filename);

        array<int64_t, 4> input_shape{1, 72, 32, 1};
        array<int64_t, 2> output_shape{5, 5};

        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());

        auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* intarr = output_tensor.front().GetTensorMutableData<float>();
        vector<float> output_tensor_values {intarr, intarr + 25};
        for(int i{}; i < output_tensor_values.size(); i++) {
            cout << output_tensor_values[i] << ",";
        }
        file.close();
    }
}