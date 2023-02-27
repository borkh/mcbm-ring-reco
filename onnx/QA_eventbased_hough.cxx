#ifndef CBM_QA_EVENTBASED_HOUGH
#include<fstream>
#include<iostream>
#include<tuple>
#define CBM_QA_EVENTBASED_HOUGH


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

Float_t getCenterXPixel(CbmRichRing* ring) { // only works for "mcbm_beam_2020_03" geometry!
  constexpr double px_width = 21.2 / 32.; // 13.1 - (- 8.1) / 32 pixel
  constexpr double px_height = 47.7/ 72.; // 23.85 - (-23.85) / 72 pixel
  for(int i{}; i < 320; i++){
    if (ring->GetCenterX() >= (-8.1 + i/10.*px_width) && ring->GetCenterX() < (-8.1 + (i+1)/10.*px_width)){
      for(int j{}; j < 720; j++){
	 if(ring->GetCenterY() <= (23.85 - j/10.*px_height) && ring->GetCenterY() > (23.85 - (j+1)/10.*px_height)){
	   return i/10.;
	 }
      }
    }
  }
  return -1; // if nothing found
}

Float_t getCenterYPixel(CbmRichRing* ring) { // only works for "mcbm_beam_2020_03" geometry!
  constexpr double px_width = 21.2 / 32.; // 13.1 - (- 8.1) / 32 pixel
  constexpr double px_height = 47.7/ 72.; // 23.85 - (-23.85) / 72 pixel
  for(int i{}; i < 320; i++){
    if (ring->GetCenterX() >= (-8.1 + i/10.*px_width) && ring->GetCenterX() < (-8.1 + (i+1)/10.*px_width)){
      for(int j{}; j < 720; j++){
	 if(ring->GetCenterY() <= (23.85 - j/10.*px_height) && ring->GetCenterY() > (23.85 - (j+1)/10.*px_height)){
	   return j/10.;
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

	for (int i = 0; i < n_pixels; i++) {
		// write 0 unless i is in indices
		if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
				cout << 1;
		}
		else {
				cout << 0;
		}
		if (i < n_pixels - 1) {
				cout << ",";
		}
	}
	cout << std::endl;

	int max_rings = 5;
	for (int i = 0; i < max_rings; i++) {
		if (i < fRichRings->GetEntriesFast()) {
			CbmRichRing* ring = static_cast<CbmRichRing*>(fRichRings->At(i));
			cout << getCenterXPixel(ring) - 0.5 << "," <<
				getCenterYPixel(ring) - 0.5 << "," <<
				ring->GetRadius()*32./21.2 << "," <<
				ring->GetRadius()*32./21.2 << ",0";
		}
		else {
			cout << "0,0,0,0,0";
		}

		if (i < max_rings - 1) {
			cout << ",";
		}
	}
	cout << std::endl;

	
}
 
void QA::Finish() {
}

void QA::Reset() {}

#endif
