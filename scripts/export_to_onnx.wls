#!/usr/bin/env wolframscript
(* Export KymoButler neural networks from Mathematica to ONNX format *)

(* Set working directory to the repo root *)
kbDir = DirectoryName[$InputFileName];
If[kbDir === "", kbDir = Directory[]];
(* Go up one level from scripts/ to repo root *)
kbDir = ParentDirectory[kbDir];
outDir = FileNameJoin[{kbDir, "models"}];
SetDirectory[kbDir];

(* Load packages *)
Print["Loading packages..."];
Get["packages/NeuralNetworkDefs.wl"];
Get["packages/KymoButler.wl"];

(* Load pre-trained models from Wolfram Cloud *)
Print["Loading models from Wolfram Cloud (this may take a few minutes)..."];
models = KymoButler`loadDefaultNets[kbDir];

Print["Models loaded: ", Keys[models]];

(* Create output directory *)
Quiet@CreateDirectory[outDir, CreateIntermediateDirectories -> True];

(* Fix unspecified shapes by specifying concrete input sizes *)
(* The UNET needs a concrete input shape for ONNX export *)
sz = 256;

(* Bidirectional net: needs grayscale image input *)
Print["Preparing bidirectional net for export..."];
binetFixed = NetReplacePart[models["binet"], "Input" -> NetEncoder[{"Image", {sz, sz}, "Grayscale"}]];
Print["Exporting bidirectional segmentation net..."];
Export[FileNameJoin[{outDir, "bidirectional_seg.onnx"}], binetFixed, "ONNX"]

(* Classification net *)
Print["Exporting classification net..."];
Export[FileNameJoin[{outDir, "classifier.onnx"}], models["classnet"], "ONNX"]

(* Unidirectional net: needs grayscale image input *)
Print["Preparing unidirectional net for export..."];
uninetFixed = NetReplacePart[models["uninet"], "Input" -> NetEncoder[{"Image", {sz, sz}, "Grayscale"}]];
Print["Exporting unidirectional segmentation net..."];
Export[FileNameJoin[{outDir, "unidirectional_seg.onnx"}], uninetFixed, "ONNX"]

(* Decision module - may need fixing too *)
Print["Exporting decision module..."];
Export[FileNameJoin[{outDir, "decision_module.onnx"}], models["decnet"], "ONNX"]

(* Export reference outputs as MX (Mathematica's native) and raw data *)
Print["Generating reference outputs..."];
testImg = Import[FileNameJoin[{kbDir, "TestAndDeploy", "bitest.png"}]];
testImgPrep = ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@testImg, "Grayscale"];
dim = ImageDimensions@testImgPrep;
pred = binetFixed[ImageResize[testImgPrep, {sz, sz}]];
(* Export as raw binary data *)
Export[FileNameJoin[{outDir, "bitest_reference_output.dat"}], Normal@pred, "Table"]

(* Also export individual layer weights as a fallback *)
Print["Exporting raw weights as fallback..."];
Quiet@CreateDirectory[FileNameJoin[{outDir, "raw_weights"}], CreateIntermediateDirectories -> True];

(* Extract and save all array-valued parameters from the binet *)
binetArrays = NetExtract[models["binet"], "Arrays"];
Export[FileNameJoin[{outDir, "raw_weights", "binet_arrays.mx"}], binetArrays, "MX"];

uninetArrays = NetExtract[models["uninet"], "Arrays"];
Export[FileNameJoin[{outDir, "raw_weights", "uninet_arrays.mx"}], uninetArrays, "MX"];

decnetArrays = NetExtract[models["decnet"], "Arrays"];
Export[FileNameJoin[{outDir, "raw_weights", "decnet_arrays.mx"}], decnetArrays, "MX"];

classnetArrays = NetExtract[models["classnet"], "Arrays"];
Export[FileNameJoin[{outDir, "raw_weights", "classnet_arrays.mx"}], classnetArrays, "MX"];

Print["Done! Check ", outDir, " for exported files."];
Print["Next step: python scripts/convert_weights.py"];
