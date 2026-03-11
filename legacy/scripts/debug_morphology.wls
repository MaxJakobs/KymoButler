#!/usr/bin/env wolframscript
(* Debug morphological pipeline for unidirectional analysis *)

kbDir = DirectoryName[$InputFileName];
If[kbDir === "", kbDir = Directory[]];
kbDir = ParentDirectory[ParentDirectory[kbDir]];
SetDirectory[kbDir];

Get["legacy/packages/NeuralNetworkDefs.wl"];
Get["legacy/packages/KymoButler.wl"];

Print["Loading models..."];
models = KymoButler`loadDefaultNets[kbDir];

refDir = FileNameJoin[{kbDir, "models", "reference"}];

(* Process unitest.png *)
Print["Processing unitest.png..."];
kym = Import[FileNameJoin[{kbDir, "tests", "data", "unitest.png"}]];
rawkym = ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym, "Grayscale"];
bool = KymoButler`Private`isNegated@rawkym;
negkym = If[bool, ColorNegate@rawkym, rawkym];
negkym = KymoButler`Private`normlines@negkym;
dim = ImageDimensions@negkym;
Print["  dim: ", dim];

resized = ImageResize[negkym, 16*Round@N[dim/16]];
Print["  resized dim: ", ImageDimensions@resized];

out = models["uninet"][{ImageData@resized}];
Print["  ant prediction shape: ", Dimensions@Normal@out["ant"]];

(* Step through morphology *)
binthresh = 0.2;
minSz = 3;
minFr = 3;

pred = Image@Normal@out["ant"];
step0 = Binarize[pred, binthresh];
Print["  after binarize: ", Total[ImageData@step0, 2], " fg pixels"];

step1 = ImageResize[step0, dim];
Print["  after resize: ", Total[ImageData@step1, 2], " fg pixels"];

step2 = Thinning@step1;
Print["  after thin1: ", Total[ImageData@step2, 2], " fg pixels"];

step3 = KymoButler`Private`SmoothBinUni@KymoButler`Private`SmoothBinUni@KymoButler`Private`SmoothBinUni@step2;
Print["  after smooth x3: ", Total[ImageData@step3, 2], " fg pixels"];

step4 = Thinning@step3;
Print["  after thin2: ", Total[ImageData@step4, 2], " fg pixels"];

step5 = Pruning[step4, 2];
Print["  after prune(2): ", Total[ImageData@step5, 2], " fg pixels"];

(* Count components before filter *)
comps = MorphologicalComponents@step5;
nComps = Max@comps;
Print["  components before filter: ", nComps];

step6 = SelectComponents[step5, #Length >= minSz && #BoundingBox[[2,2]] - #BoundingBox[[1,2]] >= minFr &];
comps2 = MorphologicalComponents@step6;
nComps2 = Max@comps2;
Print["  components after filter: ", nComps2];

(* Show filtered component details *)
Print["  Filtered components details:"];
measures = ComponentMeasurements[comps, {"Length", "BoundingBox"}];
Do[
	len = measures[[i, 2, 1]];
	bb = measures[[i, 2, 2]];
	span = bb[[2,2]] - bb[[1,2]];
	kept = len >= minSz && span >= minFr;
	If[!kept, Print["    REMOVED comp ", i, ": Length=", len, ", span=", span]],
	{i, Length@measures}
];

(* Also export intermediate results for visual comparison *)
Export[FileNameJoin[{refDir, "unitest_ant_binarized.mat"}], {"data" -> ImageData[step0]}, "MAT"];
Export[FileNameJoin[{refDir, "unitest_ant_thinned.mat"}], {"data" -> ImageData[step4]}, "MAT"];
Export[FileNameJoin[{refDir, "unitest_ant_pruned.mat"}], {"data" -> ImageData[step5]}, "MAT"];
Export[FileNameJoin[{refDir, "unitest_ant_filtered.mat"}], {"data" -> ImageData[step6]}, "MAT"];

Print["\n=== unitest2.png ==="];
kym2 = Import[FileNameJoin[{kbDir, "tests", "data", "unitest2.png"}]];
rawkym2 = ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym2, "Grayscale"];
bool2 = KymoButler`Private`isNegated@rawkym2;
negkym2 = If[bool2, ColorNegate@rawkym2, rawkym2];
negkym2 = KymoButler`Private`normlines@negkym2;
dim2 = ImageDimensions@negkym2;
resized2 = ImageResize[negkym2, 16*Round@N[dim2/16]];
out2 = models["uninet"][{ImageData@resized2}];

pred2 = Image@Normal@out2["ant"];
s0 = Binarize[pred2, binthresh];
s1 = ImageResize[s0, dim2];
s2 = Thinning@s1;
s3 = KymoButler`Private`SmoothBinUni@KymoButler`Private`SmoothBinUni@KymoButler`Private`SmoothBinUni@s2;
s4 = Thinning@s3;
s5 = Pruning[s4, 2];
c0 = Max@MorphologicalComponents@s5;
s6 = SelectComponents[s5, #Length >= minSz && #BoundingBox[[2,2]] - #BoundingBox[[1,2]] >= minFr &];
c1 = Max@MorphologicalComponents@s6;
Print["  components before filter: ", c0, " after filter: ", c1];

Print["\nDone!"];
