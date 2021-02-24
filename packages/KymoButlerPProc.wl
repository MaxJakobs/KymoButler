(* ::Package:: *)

BeginPackage["KymoButlerPProc`"]

pprocLocal::usage="pproc[trks_,tsz_,xsz_] postprocessing butler output"
pproc::usage="pproc[trks_,tsz_,xsz_] postprocessing butler output"
getDerivedQuantities::usage="get derived quantities from track"
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




pprocLocal[trks_,tsz_,xsz_]:=Module[{quant},
	quant=Map[Round[N@getDerivedQuantities@#,0.0001]&,trks];
	{{Histogram[Map[xsz/tsz*#["v"]&,quant],AxesLabel->{"v [um/sec]","Counts"}, PlotLabel->"average track velocities",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}],
	Histogram[Map[tsz*#["T"]&,quant],AxesLabel->{"T [sec]","Counts"}, PlotLabel->"Track durations",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}],
	Histogram[Map[xsz*#["dist"]&,quant],AxesLabel->{"Distance [um]","Counts"}, PlotLabel->"Track distances",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}]},
	Flatten[{{{"pixelsize time= "<>ToString@tsz<>" sec","pixelsize space= "<>ToString@xsz<>" um"},
	{"Direction","Av frame2frame velocity [um/sec]","track duration [sec]","track total displacement [um]","Start2end velocity [um/sec]"(*,"pause time [sec]","reversals #"*)}},
	Transpose@{Map[#["direct"]&,quant],Map[Round[xsz/tsz*#["v"],.0001]&,quant],Map[Round[tsz*#["T"],.0001]&,quant],Map[Round[xsz*#["dist"],.0001]&,quant],Map[Round[xsz*#["dist"]/#["T"]/tsz,.0001]&,quant](*,Map[tsz*#["pauseT"]&,quant],Map[#["reversals"]&,quant]*)}},1]}
];


pproc[trks_,tsz_,xsz_,minT_,minSz_,thr_,class_,version_]:=Module[{quant},
	quant=Map[Round[N@getDerivedQuantities@#,0.0001]&,trks];
	{{Histogram[Map[xsz/tsz*#["v"]&,quant],AxesLabel->{"v [um/sec]","Counts"}, PlotLabel->"average track velocities",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}],
	Histogram[Map[tsz*#["T"]&,quant],AxesLabel->{"T [sec]","Counts"}, PlotLabel->"Track durations",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}],
	Histogram[Map[xsz*#["dist"]&,quant],AxesLabel->{"Distance [um]","Counts"}, PlotLabel->"Track distances",LabelStyle->{"Font"->"Arial",Black},TicksStyle->{Black,Thick},AxesStyle->{Thick,Black}]},
	Flatten[{{{"KymoButler Version "<>version<>" Summary","pixelsize time= "<>ToString@tsz<>" sec","pixelsize space= "<>ToString@xsz<>" um","minimum frames= "<>ToString@minT,"minimum obj size= "<>ToString@minSz,"threshold= "<>ToString@thr},
	{"Direction","Av frame2frame velocity [um/sec]","track duration [sec]","track total displacement [um]","Start2end velocity [um/sec]"(*,"pause time [sec]","reversals #"*)}},
	Transpose@{Map[#["direct"]&,quant],Map[Round[xsz/tsz*#["v"],.0001]&,quant],Map[Round[tsz*#["T"],.0001]&,quant],Map[Round[xsz*#["dist"],.0001]&,quant],Map[Round[xsz*#["dist"]/#["T"]/tsz,.0001]&,quant](*,Map[tsz*#["pauseT"]&,quant],Map[#["reversals"]&,quant]*)}},1]}
];


End[]
EndPackage[]

