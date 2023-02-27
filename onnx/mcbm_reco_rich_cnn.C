R__LOAD_LIBRARY(/home/robin/fair_install/FairSoftInst/lib/libonnxruntime.so)
#if !defined(__CLING__)
#include "TROOT.h"
#include "TStopwatch.h"
#include "FairRunAna.h"
#include "FairFileSource.h"
#include "FairMonitor.h"
#include "CbmSetup.h"
#include "CbmMCDataManager.h"
#include "CbmMcbm2018EventBuilder.h"
#include "CbmRichMCbmHitProducer.h"
#include "CbmRichReconstruction.h"
#include "CbmMatchRecoToMC.h"
#include "FairRuntimeDb.h"
#include "FairSystemInfo.h"
#include "FairParRootFileIo.h"
#include "FairParAsciiFileIo.h"
#endif

#include "/home/robin/fair_install/FairSoftInst/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "QA_eventbased.cxx"
#include<cstdlib>


void mcbm_reco_rich_cnn(Int_t nEvents = 1000,
			 TString dataset = "data/runx",
			 const char* setupName = "mcbm_beam_2020_03",
			 bool timebased = false)
{
  bool useIdealDenoise = true;

    // --- Logger settings ----------------------------------------------------
  TString logLevel     = "NOLOG";
  TString logVerbosity = "LOW";
  // ------------------------------------------------------------------------


	// -----   Environment   --------------------------------------------------
  TString myName = "mcbm_reco";                    // this macro's name for screen output
  TString srcDir = gSystem->Getenv("VMCWORKDIR");  // top source directory
  // ------------------------------------------------------------------------


  // -----   File names   ---------------------------------------------------
  TString rawFile = dataset + ".event.raw.root";
  if (timebased) rawFile = dataset + ".raw.root";
  TString traFile = dataset + ".tra.root";
  TString parFile = dataset + ".par.root";
  TString recFile = dataset + ".rec.root";
  // ------------------------------------------------------------------------


  // -----   Load the geometry setup   --------------------------------------
  std::cout << std::endl;
  TString setupFile  = srcDir + "/geometry/setup/setup_" + setupName + ".C";
  TString setupFunct = "setup_";
  setupFunct         = setupFunct + setupName + "()";
  std::cout << "-I- " << myName << ": Loading macro " << setupFile << std::endl;
  gROOT->LoadMacro(setupFile);
  gROOT->ProcessLine(setupFunct);
  CbmSetup* setup = CbmSetup::Instance();
  //setup->RemoveModule(ECbmModuleId::kPsd);
  // ------------------------------------------------------------------------


	// -----   Parameter files as input to the runtime database   -------------
  std::cout << std::endl;
  std::cout << "-I- " << myName << ": Defining parameter files " << std::endl;
  TList* parFileList = new TList();
  TString geoTag;
  // ------------------------------------------------------------------------
  // -----   Timer   --------------------------------------------------------
  TStopwatch timer;
  timer.Start();
  // ------------------------------------------------------------------------


  // ----    Debug option   -------------------------------------------------
  gDebug = 0;
  // ------------------------------------------------------------------------


  // -----   Input file   ---------------------------------------------------
  std::cout << std::endl;
  std::cout << "-I- " << myName << ": Using input file " << rawFile << std::endl;
  // ------------------------------------------------------------------------


  // -----   FairRunAna   ---------------------------------------------------
  FairRunAna* run = new FairRunAna();
	FairFileSource* inputSource = new FairFileSource(rawFile);
  inputSource->AddFriend(traFile);
  run->SetSource(inputSource);
	run->SetOutputFile(recFile);
  run->SetGenerateRunInfo(kTRUE);
  // ------------------------------------------------------------------------


  // -----   Logger settings   ----------------------------------------------
  FairLogger::GetLogger()->SetLogScreenLevel(logLevel.Data());
  FairLogger::GetLogger()->SetLogVerbosityLevel(logVerbosity.Data());
  // ------------------------------------------------------------------------


  // ----- MC Data Manager   ------------------------------------------------
  CbmMCDataManager* mcManager= new CbmMCDataManager("MCManager", 1);
  mcManager->AddFile(rawFile);
  if (!timebased) run->AddTask(mcManager);
  // ------------------------------------------------------------------------


	// ----- Eventbuilder for timebased reco ----------------------------------
  CbmMcbm2018EventBuilder* eventBuilder = new CbmMcbm2018EventBuilder();
  //  eventBuilder->SetEventBuilderAlgo(EventBuilderAlgo::MaximumTimeGap);
  //  eventBuilder->SetMaximumTimeGap(100.);
  eventBuilder->SetEventBuilderAlgo(EventBuilderAlgo::FixedTimeWindow);
  eventBuilder->SetFixedTimeWindow(100.);
  eventBuilder->SetTriggerMinNumberT0(1);
  eventBuilder->SetTriggerMinNumberSts(0);
  eventBuilder->SetTriggerMinNumberMuch(0);
  eventBuilder->SetTriggerMinNumberTof(0);
  eventBuilder->SetTriggerMinNumberRich(5);
	eventBuilder->SetFillHistos(kFALSE);  // to prevent memory leak???

  if (timebased) run->AddTask(eventBuilder);
  // ------------------------------------------------------------------------


	// -----   Local reconstruction in RICH   ----------------------------------
  if (setup->IsActive(ECbmModuleId::kRich)) {
    CbmRichMCbmHitProducer* richHitProd = new CbmRichMCbmHitProducer();
    //richHitProd->setToTLimits(23.7,30.0);
    //richHitProd->applyToTCut();  // Don't use in Simulation
    //richHitProd->DoRestrictToAcc(); //restrict to mRICH Geometry from MAR2019
    //richHitProd->DoRestrictToFullAcc();  // rstrict to Full mRICH Geometry
    run->AddTask(richHitProd);
    std::cout << "-I- hitProducer: Added task " << richHitProd->GetName() << std::endl;

    // if(useIdealDenoise){//set NNresponse here
    //    CbmRichDenoiseIdeal* richDenoiseIdeal = new CbmRichDenoiseIdeal();
    //    run->AddTask(richDenoiseIdeal);
    // }
    CbmRichReconstruction* richReco = new CbmRichReconstruction();
    richReco->UseMCbmSetup();
    // richReco->SetFinderName("hough");
    run->AddTask(richReco);
    std::cout << "-I- richReco: Added task " << richReco->GetName() << std::endl;
  }
  // ------------------------------------------------------------------------

  // ----- Match reconstuctions to MC simulation elements -------------------
  CbmMatchRecoToMC* match2 = new CbmMatchRecoToMC();
  if (!timebased) run->AddTask(match2);
  // ------------------------------------------------------------------------
  
  // =========================================================================
  // ===                               Your QA                             ===
  // =========================================================================
  
    QA* mRichQa = new QA();
    run->AddTask(mRichQa);
  
  // =========================================================================
  
  // -----  Parameter database   --------------------------------------------
  std::cout << std::endl << std::endl;
  std::cout << "-I- " << myName << ": Set runtime DB" << std::endl;
  FairRuntimeDb* rtdb        = run->GetRuntimeDb();
  FairParRootFileIo* parIo1  = new FairParRootFileIo();
  // FairParAsciiFileIo* parIo2 = new FairParAsciiFileIo();
  parIo1->open(parFile.Data(), "UPDATE");
  // parIo2->open(parFileList, "in");
  rtdb->setFirstInput(parIo1);
  // rtdb->setSecondInput(parIo2);
  // ------------------------------------------------------------------------


  // -----   Run initialisation   -------------------------------------------
  std::cout << std::endl;
  std::cout << "-I- " << myName << ": Initialise run" << std::endl;
  run->Init();
  // ------------------------------------------------------------------------


  // -----   Database update   ----------------------------------------------
  rtdb->setOutput(parIo1);
  rtdb->saveOutput();
  rtdb->print();
  // ------------------------------------------------------------------------


  // -----   Start run   ----------------------------------------------------
  std::cout << std::endl << std::endl;
  std::cout << "-I- " << myName << ": Starting run" << std::endl;
  run->Run(0, nEvents);
  // ------------------------------------------------------------------------


  // -----   Finish   -------------------------------------------------------
  timer.Stop();
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  std::cout << std::endl << std::endl;
  std::cout << "Macro finished successfully." << std::endl;
  std::cout << "Output file is " << recFile << std::endl;
  std::cout << "Parameter file is " << parFile << std::endl;
  std::cout << "Real time " << rtime << " s, CPU time " << ctime << " s" << std::endl;
  std::cout << std::endl;
  std::cout << " Test passed" << std::endl;
  std::cout << " All ok " << std::endl;
  // ------------------------------------------------------------------------

	
  // -----   Resource monitoring   ------------------------------------------
  // if (true /*hasFairMonitor*/ /*Has_Fair_Monitor()*/) {  // FairRoot Version >= 15.11
  //   // Extract the maximal used memory an add is as Dart measurement
  //   // This line is filtered by CTest and the value send to CDash
  //   FairSystemInfo sysInfo;
  //   Float_t maxMemory = sysInfo.GetMaxMemory();
  //   std::cout << "<DartMeasurement name=\"MaxMemory\" type=\"numeric/double\">";
  //   std::cout << maxMemory;
  //   std::cout << "</DartMeasurement>" << std::endl;

  //   Float_t cpuUsage = ctime / rtime;
  //   std::cout << "<DartMeasurement name=\"CpuLoad\" type=\"numeric/double\">";
  //   std::cout << cpuUsage;
  //   std::cout << "</DartMeasurement>" << std::endl;

  //   FairMonitor* tempMon = FairMonitor::GetMonitor();
  //   tempMon->Print();
  // }
}
