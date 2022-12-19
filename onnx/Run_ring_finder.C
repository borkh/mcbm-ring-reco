#if !defined(__CLING__)
#include "TROOT.h"
#endif

void Run_ring_finder(){
    
    gROOT->ProcessLine("gSystem->Load(\"/home/robin/fair_install/FairSoftInst/lib/libonnxruntime.so\");");
    gROOT->ProcessLine("gInterpreter->AddIncludePath(\"-I/home/robin/fair_install/FairSoftInst/include/onnxruntime/core/session\");");
    gROOT->ProcessLine("#include <onnxruntime_cxx_api.h>");
    gROOT->ProcessLine("#include \"Inference_ring_finder.C\"");
    gROOT->ProcessLine("Inference_ring_finder()");
    
}

