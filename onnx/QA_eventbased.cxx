#ifndef CBM_QA_EVENTBASED
#include<fstream>
#include<iostream>
#include<tuple>
#define CBM_QA_EVENTBASED

Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ring_finder"};
Ort::Session session(env, "../models/model.onnx", Ort::SessionOptions(nullptr));

class QA : public FairTask{
	virtual InitStatus Init();
  	virtual void      Exec(Option_t * option);
  	virtual void      Finish();
  	virtual void      Reset();

	void InitHistograms();
	void DrawHist();


private:
	int eventcnt{};
	CbmDigiManager* fDigiMan = nullptr;
	CbmHistManager* fHM = nullptr;

	TClonesArray* fMCTracks = nullptr;
	TClonesArray* fRichPoints = nullptr;
	TClonesArray* fRichHits = nullptr;
	TClonesArray* fRichRings = nullptr;
	TClonesArray* fRichRingMatches = nullptr;

	ClassDef(QA,1);
};

InitStatus QA::Init(){
  std::cout << "QA Init called !" << std::endl;
  FairRootManager* manager = FairRootManager::Instance();

  fDigiMan = CbmDigiManager::Instance();
  fDigiMan->Init();

  if ( ! fDigiMan->IsPresent(ECbmModuleId::kRich) )
    Fatal("QA::Init", "No Rich Digis!");
  
  fMCTracks =(TClonesArray*) manager->GetObject("MCTrack");
  if (nullptr == fMCTracks) { Fatal("QA::Init", "No MC Track");}
  
  fRichPoints =(TClonesArray*) manager->GetObject("RichPoint");
  if (nullptr == fRichPoints) { Fatal("QA::Init", "No Rich Points!");}

  fRichHits =(TClonesArray*) manager->GetObject("RichHit");
  if (nullptr == fRichHits) { Fatal("QA::Init", "No Rich Hits!");}

  fRichRings =(TClonesArray*) manager->GetObject("RichRing");
  if (nullptr == fRichHits) { Fatal("QA::Init", "No Rich Rings!");}


  return kSUCCESS;
 }

Int_t getPixelNr(CbmRichHit* hit) { // only works for "mcbm_beam_2020_03" geometry!
  constexpr double px_width = 21.2 / 32.; // 13.1 - (- 8.1) / 32 pixel
  constexpr double px_height = 47.7/ 72.; // 23.85 - (-23.85) / 72 pixel
  for(int i{}; i < 32; i++){
    if (hit->GetX() >= (-8.1 + i*px_width) && hit->GetX() < (-8.1 + (i+1)*px_width)){
      for(int j{}; j < 72; j++){
	 if(hit->GetY() <= (23.85 - j*px_height) && hit->GetY() > (23.85 - (j+1)*px_height)){
	   return (32*j) + i; //return pixel number in [0,2303]
	 }
      }
    }
  }
  return -1; // if nothing found
}

void QA::Exec(Option_t* /*option*/) {
	eventcnt++;
  int n_pixels = 2304;
  std::vector<int> indices;
  for (int i{}; i < fRichHits->GetEntriesFast(); i++) {
          CbmRichHit* hit = static_cast<CbmRichHit*>(fRichHits->At(i));
          indices.push_back(getPixelNr(hit));
  }

  int bs = 1, w = 32, h = 72, c = 1;
  int n_rings = 5, n_params = 5;
  int input_size = bs * w * h * c;

  vector<float> input_batch(input_size);

  for (int i = 0; i < n_pixels; i++) {
      // write 0 unless i is in indices
      if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
              input_batch[i] = 1;
              cout << 1;
      }
      else {
              input_batch[i] = 0;
              cout << 0;
      }
      if (i < n_pixels - 1) {
              cout << ",";
      }
  }
  cout << std::endl;

  const char* input_names[] = {"input_1"};
  const char* output_names[] = {"reshape"};

  int n_files = 1;
  int n_batches = n_files / bs;

  array<int64_t, 4> input_shape{bs, h, w, c};
  array<int64_t, 2> output_shape{n_rings, n_params};

  auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_batch.data(), input_batch.size(), input_shape.data(), input_shape.size());

  auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
  float* intarr = output_tensor.front().GetTensorMutableData<float>();
  vector<float> output_tensor_values {intarr, intarr + bs * n_rings * n_params};
  for(int i{}; i < output_tensor_values.size(); i++) {
      cout << output_tensor_values[i];
      if (i < output_tensor_values.size() - 1) {
              cout << ",";
      }
  }
  cout << std::endl;
}

 
void QA::Finish() {
}

void QA::Reset() {}

#endif
