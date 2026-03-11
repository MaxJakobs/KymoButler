#!/usr/bin/env wolframscript
(* Export KymoButler neural networks from Mathematica to ONNX format *)

(* Set working directory to the repo root *)
kbDir = DirectoryName[$InputFileName];
If[kbDir === "", kbDir = Directory[]];
(* Go up two levels from legacy/scripts/ to repo root *)
kbDir = ParentDirectory[ParentDirectory[kbDir]];
outDir = FileNameJoin[{kbDir, "models"}];
SetDirectory[kbDir];

(* Load packages *)
Print["Loading packages..."];
Get["legacy/packages/NeuralNetworkDefs.wl"];
Get["legacy/packages/KymoButler.wl"];

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

(* ===== Generate reference predictions for validation ===== *)
(* This lets us verify that the Python ONNX pipeline matches Mathematica natively *)

refDir = FileNameJoin[{outDir, "reference"}];
Quiet@CreateDirectory[refDir, CreateIntermediateDirectories -> True];

(* Helper: preprocess image the same way KymoButler does *)
prepImage[path_] := Module[{kym, rawkym, bool, negkym, dim},
	kym = Import[path];
	rawkym = ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym, "Grayscale"];
	bool = KymoButler`Private`isNegated@rawkym;
	negkym = If[bool, ColorNegate@rawkym, rawkym];
	negkym = KymoButler`Private`normlines@negkym;
	dim = ImageDimensions@negkym;
	{negkym, dim, bool}
];

(* Bidirectional reference: bitest.png *)
Print["Generating reference predictions for bitest.png..."];
{biImg, biDim, biBool} = prepImage[FileNameJoin[{kbDir, "tests", "data", "bitest.png"}]];
biResized = ImageResize[biImg, 16*Round@N[biDim/16]];
biPred = models["binet"][{ImageData@biResized}];
Print["  bitest.png prediction shape: ", Dimensions@Normal@biPred];
Export[FileNameJoin[{refDir, "bitest_binet_pred.mat"}],
	{"pred" -> Normal@biPred}, "MAT"];

(* Unidirectional reference: unitest.png *)
Print["Generating reference predictions for unitest.png..."];
{uniImg1, uniDim1, uniBool1} = prepImage[FileNameJoin[{kbDir, "tests", "data", "unitest.png"}]];
uniResized1 = ImageResize[uniImg1, 16*Round@N[uniDim1/16]];
uniPred1 = models["uninet"][{ImageData@uniResized1}];
Print["  unitest.png ant prediction shape: ", Dimensions@Normal@uniPred1["ant"]];
Export[FileNameJoin[{refDir, "unitest_ant_pred.mat"}],
	{"pred" -> Normal@uniPred1["ant"]}, "MAT"];
Export[FileNameJoin[{refDir, "unitest_ret_pred.mat"}],
	{"pred" -> Normal@uniPred1["ret"]}, "MAT"];

(* Unidirectional reference: unitest2.png *)
Print["Generating reference predictions for unitest2.png..."];
{uniImg2, uniDim2, uniBool2} = prepImage[FileNameJoin[{kbDir, "tests", "data", "unitest2.png"}]];
uniResized2 = ImageResize[uniImg2, 16*Round@N[uniDim2/16]];
uniPred2 = models["uninet"][{ImageData@uniResized2}];
Print["  unitest2.png ant prediction shape: ", Dimensions@Normal@uniPred2["ant"]];
Export[FileNameJoin[{refDir, "unitest2_ant_pred.mat"}],
	{"pred" -> Normal@uniPred2["ant"]}, "MAT"];
Export[FileNameJoin[{refDir, "unitest2_ret_pred.mat"}],
	{"pred" -> Normal@uniPred2["ret"]}, "MAT"];

(* Also run the full Mathematica pipeline to get ground truth track counts *)
Print["Running full Mathematica pipeline for track count reference..."];
biResult = KymoButler`BiKymoButler[
	Import[FileNameJoin[{kbDir, "tests", "data", "bitest.png"}]], 0.2, 0.5, "CPU",
	models["binet"], models["decnet"], 10, 10];
biTracks = biResult[[5]];
{_,_,_,_,uniAnts1, uniRets1} = KymoButler`UniKymoButlerTrack[
	uniBool1, uniImg1, Association["ant" -> Image@Normal@uniPred1["ant"], "ret" -> Image@Normal@uniPred1["ret"]],
	0.2, 3, 3];
{_,_,_,_,uniAnts2, uniRets2} = KymoButler`UniKymoButlerTrack[
	uniBool2, uniImg2, Association["ant" -> Image@Normal@uniPred2["ant"], "ret" -> Image@Normal@uniPred2["ret"]],
	0.2, 3, 3];

Print["Reference track counts:"];
Print["  bitest.png: ", Length@biTracks, " tracks"];
Print["  unitest.png: ", Length@uniAnts1, " ant + ", Length@uniRets1, " ret tracks"];
Print["  unitest2.png: ", Length@uniAnts2, " ant + ", Length@uniRets2, " ret tracks"];

(* Save track counts as JSON *)
Export[FileNameJoin[{refDir, "reference_track_counts.json"}],
	<|"bitest" -> Length@biTracks,
	  "unitest_ant" -> Length@uniAnts1, "unitest_ret" -> Length@uniRets1,
	  "unitest2_ant" -> Length@uniAnts2, "unitest2_ret" -> Length@uniRets2|>,
	"JSON"];

(* Also export individual layer weights as a fallback *)
Print["Exporting raw weights as fallback..."];
Quiet@CreateDirectory[FileNameJoin[{outDir, "raw_weights"}], CreateIntermediateDirectories -> True];

(* Extract and save all learnable parameters from the nets *)
binetArrays = NetExtract[models["binet"], "LearnableParameters"];
Export[FileNameJoin[{outDir, "raw_weights", "binet_params.mx"}], binetArrays, "MX"];

uninetArrays = NetExtract[models["uninet"], "LearnableParameters"];
Export[FileNameJoin[{outDir, "raw_weights", "uninet_params.mx"}], uninetArrays, "MX"];

decnetArrays = NetExtract[models["decnet"], "LearnableParameters"];
Export[FileNameJoin[{outDir, "raw_weights", "decnet_params.mx"}], decnetArrays, "MX"];

classnetArrays = NetExtract[models["classnet"], "LearnableParameters"];
Export[FileNameJoin[{outDir, "raw_weights", "classnet_params.mx"}], classnetArrays, "MX"];

Print["Done! Check ", outDir, " for exported files."];
Print["Next step: python scripts/convert_weights.py"];
