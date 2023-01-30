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
    int n_files = 0;
    if ((dir = opendir (dir_name.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            string file_name = ent->d_name;
            if (file_name.find(".png") != string::npos) {
                n_files++;
            }
        }
        closedir (dir);
    } else {
        perror ("");
        return EXIT_FAILURE;
    }
    return n_files;
}

void Inference_ring_finder() {

    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ring_finder"};
    Ort::Session session(env, "../models/model.onnx", Ort::SessionOptions(nullptr));

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"reshape"};

    string dir_name = "../data/test/X/";
    int n_files = number_of_png_files_in_dir(dir_name);
    int bs = 1000, w = 32, h = 72, c = 1;
    int n_rings = 5, n_params = 5;

    int n_batches = n_files / bs;

    int input_size = bs * w * h * c;

    for (int i = 0; i < n_batches; i++) {
        vector<float> input_batch(input_size);

        // load images and write them to input_batch
        for (int j{}; j < bs; j++) {
            string filename = dir_name + to_string(i * bs + j) + ".png";
            unsigned char* pixels = stbi_load(filename.c_str(), &w, &h, &c, 0);
            for (int k{}; k < w * h * c; k++) {
                input_batch[j * w * h * c + k] = pixels[k] / 255.0;
            }
        }

        array<int64_t, 4> input_shape{bs, h, w, c};
        array<int64_t, 2> output_shape{n_rings, n_params};

        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_batch.data(), input_batch.size(), input_shape.data(), input_shape.size());

        auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* intarr = output_tensor.front().GetTensorMutableData<float>();
        vector<float> output_tensor_values {intarr, intarr + bs * n_rings * n_params};
        for(int i{}; i < output_tensor_values.size(); i++) {
            cout << output_tensor_values[i] << ",";
        }
    }
}