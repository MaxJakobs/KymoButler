#!/usr/bin/env wolframscript
(* ::Package:: *)

dir=DirectoryName@$InputFileName;
If[dir=="",dir=NotebookDirectory[]];


Get[FileNameJoin@{dir,"..","packages","KymoButler.wl"}]
Get[FileNameJoin@{dir,"..","packages","KymoButlerPProc.wl"}]


models=Quiet[loadDefaultNets[dir<>".."]];


Print@"testing unidirectional";
resultUNI=UniKymoButler[Import[dir<>"unitest.png"],(*detection threshold*).2,(*target Device (GPU or CPU), GPU only available with NVIDIA GPUs*)"CPU",models["uninet"],(*minimum particle size*)3,(*minimum frame number*)3];
resultUNI2=UniKymoButler[Import[dir<>"unitest2.png"],(*detection threshold*).2,(*target Device (GPU or CPU), GPU only available with NVIDIA GPUs*)"CPU",models["uninet"],(*minimum particle size*)3,(*minimum frame number*)3];


uniButtest=VerificationTest[Length@resultUNI[[-1]],7]
uniButtest2=VerificationTest[Length@resultUNI2[[-2]],9]


Print@"testing bidirectional";


resultBI=BiKymoButler[Import[dir<>"bitest.png"],(*detection threshold*).2,(*decision threshold do not change*).5,(*target Device (GPU or CPU)*)"CPU",models["binet"],models["decnet"],(*minimum particle size*)10,(*minimum frame number*)10];


biButtest=VerificationTest[Length@resultBI[[-1]],13]


Print@"testing postprocessing";


processed=pproc[resultBI[[-1]] (*tracks are the last entry in the results*),1(*pixelsize time*),1(*pixelsize space*)]


biButpproctest=VerificationTest[Length@processed[[-1]],15]

Print@"test results:"
report=TestReport@{uniButtest,uniButtest2,biButtest,biButpproctest,};
Print@Normal@Dataset[report[{"TimeElapsed","TestsFailedCount","TestsSucceededCount"}]]

If[report["TestsFailedCount"]==0,
Print@"Deploying code...";
DumpSave["dump.mx", {"KymoButler`","KymoButlerPProc`"}];
DeleteObject[CloudObject["packages/KymoButler/code64x"]];
obj = CopyFile["dump.mx", CloudObject["packages/KymoButler/code64x"]];
SetOptions[obj, Permissions -> "Public"];
DeleteFile["dump.mx"]
]
