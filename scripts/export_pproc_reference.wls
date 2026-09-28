#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Mathematica ground truth for KymoButler's output files (tests/data/reference/pproc_mma.json).

   Usage: wolframscript -file scripts/export_pproc_reference.wls <input.json> <output.json> <workdir>
   (run by scripts/make_pproc_reference.py, which writes the input and assembles the fixture).

   For every track set and pixel size pair of the input it exports, with ExportString[..., "CSV"]:
     summary: pproc[trks, tsz, xsz, minT, minSz, thr, 2, version][[2]]   (the web form's summary.csv)
     local:   pprocLocal[trks, tsz, xsz][[2]]                            (the Mathematica package's)
     coords:  ConvertToCSV[trks, tsz, xsz]                               (the web form's trackCoordinates.csv)
   with pproc/pprocLocal from legacy/packages/KymoButlerPProc.wl and ConvertToCSV as defined in the
   KymoButler web form's deployment package (KymoButlerDeployment.wl, reproduced verbatim below).
   It also records ToString of header numbers (typed ones, and machine reals given as JSON numbers),
   CSV cells of assorted numbers, the overlay's "coloredlines" image of
   BiKymoButlerTrack/UniKymoButlerTrack (KymoButler.wl) for some track sets, with RandomColor[]
   replaced by a colour that encodes the track number, the overlay
   ImageCompose[ImageAdjust@kym, RemoveBackground[coloredlines, {Black, .01}]] for some images, and
   the bytes ImageCompose gives a 16-bit and a Real32 kymograph (written as <name>_kymtype.png). *)

dir = DirectoryName[$InputFileName];
Get[FileNameJoin[{dir, "..", "legacy", "packages", "KymoButlerPProc.wl"}]];

ConvertToCSV[trks_,tsz_,xsz_]:=Module[
	{maxL=Max@Map[Length,trks],tmp,rescaledtrks},
	If[Length@trks>0,
		rescaledtrks=Map[{#[[1]]*tsz,#[[2]]*xsz}&,trks,{2}];
		tmp=MapThread[Prepend[PadRight[#1,maxL,{{Null,Null}}],{"t [pixel] #"<>ToString@#2,"x [pixel] #"<>ToString@#2}]&,{rescaledtrks,Range@Length@rescaledtrks}];
		Map[Flatten@tmp[[;;,#]]&,Range@(maxL+1)],{"None"}]
		];

(* KymoButler.wl's coloredlines with RandomColor[] replaced by the colour of track number k *)
col[k_] := RGBColor[(50 + k)/255., (50 + Mod[k*37, 150])/255., 200/255.];
coloredlines[trks_, dim_] := ImageRotate[Image[Show[Image@Table[0, dim[[1]], dim[[2]]],
    Graphics@MapIndexed[{col[First@#2], Style[Line@#1, Antialiasing -> False]} &, Map[# - {1, 0} &, trks, {2}]]],
    ImageSize -> Reverse@dim], -Pi/2];
rawkym[file_] := ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@Import[file], "Grayscale"];

{inFile, outFile, work} = Rest[$ScriptCommandLine];
in = Import[inFile, "RawJSON"];
num[s_String] := ToExpression[s];  (* "1" -> 1, "0.5" -> 0.5: integers stay exact, as the form gives them *)
hdr = in["header"];
out = <|"generator" -> "Mathematica " <> $Version <> ", scripts/export_pproc_reference.wls",
  "header" -> hdr, "csv" -> <||>|>;
Do[
  With[{trks = in["sets"][name], tsz = num[p[[1]]], xsz = num[p[[2]]]},
    out["csv"][name <> "|" <> p[[1]] <> "|" <> p[[2]]] = <|
      "summary" -> Quiet@ExportString[pproc[trks, tsz, xsz, hdr["minT"], hdr["minSz"], num[hdr["thr"]], 2, hdr["version"]][[2]], "CSV"],
      "local" -> Quiet@ExportString[pprocLocal[trks, tsz, xsz][[2]], "CSV"],
      "coords" -> Quiet@ExportString[ConvertToCSV[trks, tsz, xsz], "CSV"]
    |>],
  {name, Keys[in["sets"]]}, {p, in["pixel_sizes"]}];
out["tostring"] = Map[{#, ToString[num[#]]} &, in["tostring"]];
(* machine reals (JSON numbers; a typed 17-digit string would be an arbitrary-precision number) *)
out["tostring_reals"] = Map[{ToString[#, InputForm], ToString[#]} &, in["tostring_reals"]];
(* one number as a CSV cell: in a ragged table (InputForm), a typed column, a column with a Null *)
out["cells"] = Map[{#, ExportString[{{"a", "b"}, {num[#]}}, "CSV"], ExportString[{{"a"}, {num[#]}}, "CSV"],
    ExportString[{{"a"}, {num[#]}, {Null}}, "CSV"]} &, in["cells"]];
(* coloredlines: written as PNG files into work/, named <case>.png *)
out["lines"] = Association@Map[
    (Export[FileNameJoin[{work, #["name"] <> "_lines.png"}], coloredlines[in["sets"][#["set"]], #["dims"]]];
     #["name"] -> ImageDimensions@coloredlines[in["sets"][#["set"]], #["dims"]]) &, in["lines"]];
(* the whole overlay on a kymograph file: bidirectional composes onto ImageAdjust@kym *)
out["overlays"] = Association@Map[
    Module[{kym = rawkym[#["image"]], ov},
      ov = ImageCompose[ImageAdjust@kym, RemoveBackground[coloredlines[in["sets"][#["set"]], ImageDimensions@kym], {Black, .01}]];
      Export[FileNameJoin[{work, #["name"] <> "_overlay.png"}], ov];
      #["name"] -> {ImageType@kym, ImageType@ov, ImageChannels@ov}] &, in["overlays"]];
(* the kymograph bytes under the overlay for non-Byte kymographs: a Bit16 ramp of all 65536 values
   and seeded Real32 noise, composed with a fully transparent line image *)
blank[dims_] := RemoveBackground[Image[ConstantArray[{0, 0, 0}, Reverse@dims], "Byte"], {Black, .01}];
SeedRandom[1];
kymtypes = <|"bit16" -> Image[Partition[Range[0, 65535], 256], "Bit16"],
  "real32" -> Image[RandomReal[1, {64, 64}], "Real32"]|>;
kymtype[name_, kym_] := Module[{ov = ImageCompose[kym, blank@ImageDimensions@kym]},
   Export[FileNameJoin[{work, name <> "_kymtype.png"}], ov];
   name -> <|"type" -> ImageType@kym, "overlay" -> {ImageType@ov, ImageChannels@ov},
     "values" -> If[ImageType@kym === "Real32", Map[ToString[#, InputForm] &, Flatten@ImageData@kym], {}]|>];
out["kymtypes"] = Association@KeyValueMap[kymtype, kymtypes];
Export[outFile, out, "RawJSON"];
Print["wrote ", outFile, ": ", Length[out["csv"]], " track set / pixel size pairs"];
