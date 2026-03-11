#!/usr/bin/env wolframscript
(* Export reference track coordinates from Mathematica for comparison with Python *)

kbDir = DirectoryName[$InputFileName];
If[kbDir === "", kbDir = Directory[]];
kbDir = ParentDirectory[ParentDirectory[kbDir]];
SetDirectory[kbDir];

Get["legacy/packages/NeuralNetworkDefs.wl"];
Get["legacy/packages/KymoButler.wl"];

Print["Loading models..."];
models = KymoButler`loadDefaultNets[kbDir];

refDir = FileNameJoin[{kbDir, "models", "reference"}];
If[!DirectoryQ[refDir], CreateDirectory[refDir]];

(* === Unidirectional: unitest.png === *)
Print["\n=== unitest.png (unidirectional) ==="];
kym = Import[FileNameJoin[{kbDir, "tests", "data", "unitest.png"}]];
result = KymoButler`UniKymoButler[kym, 0.2, "CPU", models["uninet"], 3, 3];
(* result = {tmpkym, coloredlines, overlay, overlaylabeled, antrks, retrks} *)
antTrks = result[[5]];
retTrks = result[[6]];
Print["  ant tracks: ", Length@antTrks];
Print["  ret tracks: ", Length@retTrks];

(* Export overlay image *)
Export[FileNameJoin[{refDir, "unitest_overlay.png"}], result[[3]]];
Print["  Saved overlay"];

(* Export track coordinates as JSON *)
antJSON = Map[Map[{"time" -> #[[1]], "space" -> #[[2]]} &, #] &, antTrks];
retJSON = Map[Map[{"time" -> #[[1]], "space" -> #[[2]]} &, #] &, retTrks];
Export[FileNameJoin[{refDir, "unitest_tracks.json"}],
  <|"ant" -> antJSON, "ret" -> retJSON|>, "JSON"];
Print["  Saved track coordinates"];

(* === Unidirectional: unitest2.png === *)
Print["\n=== unitest2.png (unidirectional) ==="];
kym2 = Import[FileNameJoin[{kbDir, "tests", "data", "unitest2.png"}]];
result2 = KymoButler`UniKymoButler[kym2, 0.2, "CPU", models["uninet"], 3, 3];
antTrks2 = result2[[5]];
retTrks2 = result2[[6]];
Print["  ant tracks: ", Length@antTrks2];
Print["  ret tracks: ", Length@retTrks2];

Export[FileNameJoin[{refDir, "unitest2_overlay.png"}], result2[[3]]];
antJSON2 = Map[Map[{"time" -> #[[1]], "space" -> #[[2]]} &, #] &, antTrks2];
retJSON2 = Map[Map[{"time" -> #[[1]], "space" -> #[[2]]} &, #] &, retTrks2];
Export[FileNameJoin[{refDir, "unitest2_tracks.json"}],
  <|"ant" -> antJSON2, "ret" -> retJSON2|>, "JSON"];
Print["  Saved overlay and tracks"];

(* === Bidirectional: bitest.png === *)
Print["\n=== bitest.png (bidirectional) ==="];
kym3 = Import[FileNameJoin[{kbDir, "tests", "data", "bitest.png"}]];
result3 = KymoButler`BiKymoButler[kym3, 0.2, 0.5, "CPU", models["binet"], models["decnet"], 10, 10];
(* BiKymoButler returns: {kym, coloredlines, overlay, overlaylabeled, trks} *)
biTrks = result3[[5]];
Print["  tracks: ", Length@biTrks];

Export[FileNameJoin[{refDir, "bitest_overlay.png"}], result3[[3]]];
biJSON = Map[Map[{"time" -> #[[1]], "space" -> #[[2]]} &, #] &, biTrks];
Export[FileNameJoin[{refDir, "bitest_tracks.json"}],
  <|"tracks" -> biJSON|>, "JSON"];
Print["  Saved overlay and tracks"];

Print["\nDone! Reference files saved to: ", refDir];
