(* ::Package:: *)

BeginPackage["KymoButlerPProcV1v0v0`"]

pproc::usage="pproc[trks_,tsz_,xsz_] postprocessing butler output"

Begin["`Private`"]



(* ::Section:: *)
(*Basic*)


getDerivedQuantities[trk_]:=Module[{direct,v,diff=Differences@trk,dist,pauseT,T,reversals,segments},
If[Length@trk>1,
v=Mean[Flatten[Abs/@Ratios/@diff]];(*Mean Frame to frame velocity*)
direct=Sign[trk[[-1,2]]-trk[[1,2]]]; (*direction*)
dist=Total@Abs@Differences@trk[[;;,2]]; 
T=trk[[-1,1]]-trk[[1,1]]+1;
<|"direct"->direct,"v"->v,"dist"->dist,"pauseT"->0,"T"->Abs@T,"reversals"->0|>,
Missing[]]
];




pproc[trks_,tsz_,xsz_]:=Module[{quant},
	quant=Map[Round[N@getDerivedQuantities@#,0.0001]&,trks];
	{{Histogram[Map[xsz/tsz*#["v"]&,quant],AxesLabel->{"v [um/sec]","Counts"}, PlotLabel->"average track velocities",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}],
	Histogram[Map[tsz*#["T"]&,quant],AxesLabel->{"T [sec]","Counts"}, PlotLabel->"Track durations",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}],
	Histogram[Map[xsz*#["dist"]&,quant],AxesLabel->{"Distance [um]","Counts"}, PlotLabel->"Track distances",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}]},
	Flatten[{{{"pixelsize time= "<>ToString@tsz<>" sec","pixelsize space= "<>ToString@xsz<>" um"},
	{"Direction","Av frame2frame velocity [um/sec]","track duration [sec]","track total displacement [um]","Start2end velocity [um/sec]"(*,"pause time [sec]","reversals #"*)}},
	Transpose@{Map[#["direct"]&,quant],Map[Round[xsz/tsz*#["v"],.0001]&,quant],Map[Round[tsz*#["T"],.0001]&,quant],Map[Round[xsz*#["dist"],.0001]&,quant],Map[Round[xsz*#["dist"]/#["T"]/tsz,.0001]&,quant](*,Map[tsz*#["pauseT"]&,quant],Map[#["reversals"]&,quant]*)}},1]}
];


(*Basic Cloud form setup*)
ConvertToCSV[trks_,tsz_,xsz_]:=Module[
	{maxL=Max@Map[Length,trks],tmp,rescaledtrks},
	If[Length@trks>0,
		rescaledtrks=Map[{#[[1]]*tsz,#[[2]]*xsz}&,trks,{2}];
		tmp=MapThread[Prepend[PadRight[#1,maxL,{{Null,Null}}],{"t [pixel] #"<>ToString@#2,"x [pixel] #"<>ToString@#2}]&,{rescaledtrks,Range@Length@rescaledtrks}];
		Map[Flatten@tmp[[;;,#]]&,Range@(maxL+1)],{"None"}]
		];
		
ConvertToCSVquant[q_,name_]:=Module[
	{maxL=Max@Map[Length,q],tmp},
	If[Length@q>0,
		(*rescaledv=Map[#*xsz/tsz&,f2fv,{2}];*)
		tmp=MapThread[Prepend[PadRight[#1,maxL,{{Null}}],{name<>" #"<>ToString@#2}]&,{q,Range@Length@q}];
		Map[Flatten@tmp[[;;,#]]&,Range@(maxL+1)],{"None"}]
		];



End[]
EndPackage[]

