(* ::Package:: *)

BeginPackage["KymoButler`"]
UniKymoButler::usage="Function analyses a unidirectional kymograph"
BiKymoButler::usage="Function analyses a bidirectional kymograph"
BiKymoButlerSegment::usage="Function segments a bidirectional kymograph"
UniKymoButlerSegment::usage="Function segments a unidirectional kymograph"
BiKymoButlerTrack::usage="BiKymoButlerTrack[pred_,kym_,kympreproc_,bool_,binthresh_,vthr_,vismod_,minSz_,minFr_,debug_:False] Function tracks a previously segmented bidirectional kymograph"
UniKymoButlerTrack::usage="Function tracks a previously segmented unidirectional kymograph"
benchmarkPrediction::usage="calculates track recal precision for a prediction and a ground truth"
resolveOvlps::usage="Deletes overlapping tracks for benchmarking"
loadDefaultNets::usage="loads default networks"
deployPackage::usage="Deploys Package to  cloud"


Begin["`Private`"]


loadDefaultNets[dir_]:=Module[{tmp,netnames,cpath},
	netnames=If[ToExpression@StringTake[$Version,4]>12,
		{"Bidirectional_Segmentation_Module_V1_1","Classification_Module_V1_0","Unidrectional_Segmentation_Module_V1_0","Decision_Module_V1_0"},
		{"Bidirectional_Segmentation_Module_V1_1_Math12_1","Classification_Module_V1_0_Math12_1","Unidrectional_Segmentation_Module_V1_0","Decision_Module_V1_0"}];
	cpath="https://www.wolframcloud.com/objects/deepmirror/Projects/KymoButler/networks/";
	CreateDirectory@FileNameJoin@{dir,"models"};
	AssociationThread[{"binet","classnet","uninet","decnet"}->Map[If[
		FileExistsQ@FileNameJoin@{dir,"models",#},
			Import@FileNameJoin@{dir,"models",#},
			(tmp=CloudImport@CloudObject[cpath<>#];
			Export[FileNameJoin@{dir,"models",#},tmp,"MX"];
			tmp)
			]&,netnames]]
]


(* ::Section:: *)
(* Unidirectional Functions *)


isNegated[kym_]:=Module[{n1=Total@Binarize[kym, .5],n2=Total@Binarize[ColorNegate@kym,.5]},
If[n1>=n2,True,False]];

SmoothBinUni[out_]:=out+HitMissTransform[out,{{{0,1,0},{0,-1,1},{0,1,0}},{{0,1,0},{1,-1,1},{0,0,0}},{{0,1,0},{1,-1,0},{0,1,0}},{{0,0,0},{1,-1,1},{0,1,0}}},Padding->0];

normlines[kym_]:=ImageAdjust@Image@Map[If[Mean[#]>0.,#/Mean@#,#]&,ImageData@kym];

UniKymoButlerSegment[kym_,net_,tD_]:=Module[{rawkym,bool,negkym,out,dim=ImageDimensions@kym},
	(*Adjust Image*)
	rawkym=ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym,"Grayscale"];
	(*ColorNegate if backgroudn white*)
	bool=isNegated@rawkym;
	negkym=If[bool,ColorNegate@rawkym,rawkym];
	(*normalize kymolines*)
	negkym=normlines@negkym;
	(*Run net*)
	out=net[{ImageData@ImageResize[negkym,16*Round@N[dim/16]]},TargetDevice->tD];
	If[FailureQ@out,$Failed,
	{bool,rawkym,negkym,Image/@out}]
];

UniKymoButlerTrack[bool_,tmpkym_,out_,binthresh_,minSz_(*default 3*),minFr_(*default 3*)]:=Module[{tmpa,tmpr,antPaths,retPaths,antrks,retrks,dim=ImageDimensions@tmpkym,coloredlines,overlay,ca,cr,labels,overlaylabeled},
	tmpa=SelectComponents[Pruning[Thinning@SmoothBinUni@SmoothBinUni@SmoothBinUni@Thinning@ImageResize[Binarize[out["ant"],binthresh],dim],2],#Length>=minSz&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>=minFr&];
	tmpr=SelectComponents[Pruning[Thinning@SmoothBinUni@SmoothBinUni@SmoothBinUni@Thinning@ImageResize[Binarize[out["ret"],binthresh],dim],2],#Length>=minSz&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>=minFr&];
	
	tmpa=MorphologicalComponents@tmpa;
	tmpr=MorphologicalComponents@tmpr;

	(*get pixel values*)
	antPaths=Image@tmpa;
	retPaths=Image@tmpr;
	antrks=Transpose[{dim[[2]]+1-PixelValuePositions[antPaths,#][[;;,2]],PixelValuePositions[antPaths,#][[;;,1]]}]&/@Range[Length@Union@Flatten@tmpa-1];
	retrks=Transpose[{dim[[2]]+1-PixelValuePositions[retPaths,#][[;;,2]],PixelValuePositions[retPaths,#][[;;,1]]}]&/@Range[Length@Union@Flatten@tmpr-1];

	If[Length@antrks>0 || Length@retrks>0,	
	(*Extract average values at timepoints*)
	antrks=Map[Round/@Mean/@GatherBy[#,First]&,antrks];
	retrks=Map[Round/@Mean/@GatherBy[#,First]&,retrks];
	
	antrks=Select[antrks,First@Last@#-First@First@#>=minFr&];
	retrks=Select[retrks,First@Last@#-First@First@#>=minFr&];
	(*colored lines and overlays*)
	coloredlines=ImageRotate[Image[Show[Image@Table[0,dim[[1]],dim[[2]]],Graphics@Map[{RandomColor[],Style[Line@#,Antialiasing->False]}&,Map[#-{1,0}&,Flatten[{antrks,retrks},1],{2}]]],ImageSize->Reverse@dim],-Pi/2];
	overlay=ImageCompose[tmpkym,RemoveBackground[coloredlines,{Black,.01}]];
	(*get labels and label overlay*)
	ca=SparseArray[Flatten[Map[Thread[antrks[[#]]->#]&,Range@Length@antrks],1],{dim[[2]],dim[[1]]}];
	cr=SparseArray[Flatten[Map[Thread[retrks[[#]]->#]&,Range@Length@retrks],1],{dim[[2]],dim[[1]]}];
	ca=ComponentMeasurements[ca,"Centroid"];
	cr=ComponentMeasurements[cr,"Centroid"];
	(*relabel retrograde ones*)
	cr=ReplacePart[#,1->(#[[1]]+Length@ca)]&/@cr;
	labels=Join[ca,cr];
	Sow@labels;
	overlaylabeled=HighlightImage[overlay,Map[ImageMarker[labels[[#,2]]+{0,5},Graphics[{If[bool,Black,White],Text[Style[ToString@#,FontSize->Scaled@.03]]}]]&,Range[Length@labels]]];
	{tmpkym,coloredlines,overlay,overlaylabeled,antrks,retrks},
	{tmpkym,Image@Table[0,dim[[2]],dim[[1]]],tmpkym,tmpkym,{},{}}
        ]

];

UniKymoButler[kym_,binthresh_,tD_,net_,minSz_,minFr_]:=Module[
	{a,r,rawkym,bool,negkym,out},
	{bool,rawkym,negkym,out}=UniKymoButlerSegment[kym,net,tD];
	UniKymoButlerTrack[bool,rawkym,out,binthresh,minSz,minFr]
];


(* ::Section:: *)
(* Bidirectional Functions *)


MaxIndx[a_]:=First@First@Position[a,Last@Union@a]
MinIndx[a_]:=First@First@Position[a,First@Union@a]

FindShortPathImage[bin_,s_,f_]:=Module[{bindat=Round@ImageData@bin,vertices,centroids,renumber,neighbors,edges,g,path},
	bindat=ReplacePart[bindat,{s,f}->0];
	renumber=Module[{i=3},ReplaceAll[1:>i++]@#]&;
	bindat=renumber@bindat;
	bindat=ReplacePart[bindat,{s->1,f->2}];
	vertices=ComponentMeasurements[bindat,"Label"][[All,1]];
	centroids=ComponentMeasurements[bindat,"Centroid"];
	neighbors=ComponentMeasurements[bindat,"Neighbors"];
	edges=UndirectedEdge@@@DeleteDuplicates[Sort/@Flatten[Thread/@neighbors]];
	g=Graph[vertices,edges,VertexCoordinates->centroids];
	path=FindShortestPath[g,1,2];
	If[Length@path>0,
	First[Position[bindat,#]]&/@path,{}]
];

SortCoords[x_]:=Module[
	{outL={x[[1]]},outR={x[[1]]},xtmp=x[[2;;]],foo},
	While[Length@xtmp>0,
		foo=Nearest[xtmp,Last@outL,{All,1.5}];
		If[Length@foo>0,
			outL=Flatten[{outL,{First@SortBy[foo,Last]}},1];
			xtmp=DeleteCases[xtmp,First@SortBy[foo,Last]];,
			xtmp={}]
	];
	xtmp=x[[2;;]];
	While[Length@xtmp>0,
		foo=Nearest[xtmp,Last@outR,{All,1.5}];
		If[Length@foo>0,
			outR=Flatten[{outR,{Last@SortBy[foo,Last]}},1];
			xtmp=DeleteCases[xtmp,Last@SortBy[foo,Last]];,
			xtmp={}]
	];
	Last@SortBy[{outR,outL},Length]
];

GetTile[kym_,trk_,allyx_,vismoddim_]:=Module[{dim=ImageDimensions@kym,win,t},
	win=Transpose@{Round[Last@trk-(vismoddim/2)],Round[Last@trk+(vismoddim/2-1)]};
	(*if boundary rescale*)
	{t,win}=AbsoluteTiming@{Which[First@Union@win[[1]]<=0,win[[1]]-First@Union@win[[1]]+1,
	Last@Union@win[[1]]>dim[[2]],win[[1]]-(Last@Union@win[[1]]-dim[[2]]),
	True,win[[1]]],
	Which[First@Union@win[[2]]<=0, win[[2]]-First@Union@win[[2]]+1,
	Last@Union@win[[2]]>dim[[1]], win[[2]]-(Last@Union@win[[2]]-dim[[1]]),
	True,win[[2]]]};
	Sow[t,"boundrescale"];
	(*return tile binary rescaled candidate*)
	{ImageAdjust@ImageTake[kym,win[[1]],win[[2]]],
	Image@Take[SparseArray[Round/@trk->1,{dim[[2]],dim[[1]]}],win[[1]],win[[2]]],
	Image@Take[SparseArray[Round/@allyx->1,{dim[[2]],dim[[1]]}],win[[1]],win[[2]]],win}
];

GetCandFromPmap[pmap_,thr_]:=Module[{
	comp=ComponentMeasurements[Binarize[pmap,thr],{"Mask","Count"}],maxA},
	If[Length@comp>0(*&Length@comp<4*),
	maxA=MaxIndx@Map[#[[2,2]]&,comp];
	(*return candidates with maximum area*)
	Sow[Total@Flatten[comp[[maxA,2,1]]*ImageData@pmap]/Total[Flatten@comp[[maxA,2,1]]],"DecisionProb"];
	Position[Round@ImageData@Thinning@Image@comp[[maxA,2,1]],1],
	{}]];

(*GetCandLinpred[bin_,fullbin_]:=Module[{
comp=ComponentMeasurements[Binarize[ImageReflect[Dilation[ImagePad[ImageTake[bin,{1,25}],{{0,0},{23,0}},"Reflected"]-bin,1],Left]*fullbin-bin,.5],{"Mask","Count"}],
maxA},
If[Length@comp>0(*&Length@comp<4*),
maxA=MaxIndx@Map[#[[2,2]]&,comp];
(*return candidates with maximum area*)
Sow[RandomReal[],"DecisionProb"];
Position[Round@ImageData@Thinning@Image@comp[[maxA,2,1]],1],
{}]];*)

GetCand[kym_,trk_,allyx_,thr_,vismod_,debug_:False]:=Module[{
	(*find all candidates 8 pixels away*)
	t,padkym,tmp,tile,allyxtmp,allyxR,bin,trkR,rc,pmap,win,cands,select,cost,fullbin,lastTrk,result,trktmp,dim,shortestpath},
	(*rescale everything to a padded kymograph*)
	(*dim=Last@First@NetInformation[vismod,"InputPorts"];*)
	Sow[First@AbsoluteTiming[dim=48;
	padkym=Image@ArrayPad[ImageData@kym,Round[1+dim/2],.1];
	allyxtmp=allyx+Round[1+dim/2];
	trktmp=trk+Round[1+dim/2];
	allyxtmp=SortBy[Nearest[allyxtmp,Last@trktmp,{All,dim*1.5}],First];],"prep"];
	
	If[Length@allyxtmp>0,
		trktmp=Drop[trktmp,-1];

		{t,{tile,bin,fullbin,win}}=AbsoluteTiming@GetTile[padkym,trktmp,allyxtmp,dim];
		Sow[t,"gettiles"];
		(*Load net and execute*)
		{t,pmap}=AbsoluteTiming@Image@Map[Last,vismod@Map[ImageData[#,Interleaving->False]&,{ImageAdjust@tile,bin,fullbin}],{2}];
		Sow[t,"applyvismod"];
		(*rescale allyxtmp and track*)
		Sow[First@AbsoluteTiming[
		allyxR=Map[#-First@Transpose@win+{1,1}&,allyxtmp];
		trkR=Map[#-First@Transpose@win+{1,1}&,trktmp];
		lastTrk=Last@trktmp-First@Transpose@win+{1,1};],"rescale"];
		(*get coordinates of largest connected object*)
		
		{t,cands}=AbsoluteTiming@GetCandFromPmap[pmap,thr];
		Sow[t,"getcandfrompmap"];
		(*get coordinates of largest connected object with simple linear prediction*)
		(*cands=GetCandLinpred[bin,fullbin];*)
		(*Delete coords that we know allready*)
		cands=Complement[cands,trktmp];
		
		If[debug,Sow[{tile,bin,(*allyxR,lastTrk,cands,*)Image@SparseArray[Select[allyxR,First@Union@#>0&&Last@Union@#< dim+1&]->1,{dim,dim}],pmap,Image@ReplacePart[Table[0,dim,dim],cands->1](*,Image@ReplacePart[Table[0,dim,dim],select\[Rule]1]*)}];];
		(*Select from all*)
		(*Roughly Sort candidates by distance to last entry in trk, additionally sort so that they are one connected line from start to end*)
	
		{t,result}=AbsoluteTiming[cands=SortBy[cands,N@EuclideanDistance[#,lastTrk]&];
		If[Length@cands>2 &&(*EuclideanDistance[lastTrk,First@cands]<15&&*)Mean[cands[[;;,1]]]-First@lastTrk>=-1,(*Dont use dots as predictions, has to be more than 2 also remove crazy prediction*)
			cands=SortCoords@cands;
			(*do pathfinding to fill any gaps in the prediction*)
			select=If[EuclideanDistance[lastTrk,First@cands]>1.5,
				shortestpath=FindShortPathImage[Image@SparseArray[Select[allyxR,First@Union@#>0&&Last@Union@#<dim+1&]->1,{dim,dim}],lastTrk,First@cands];
				Join[shortestpath,cands]
				,cands];
			(*Only select at most 24 coordinates*)
			select=Take[select,UpTo@24];
			(*Delete selection of completely stupid pathfinding results*)
			If[Mean[select[[;;,1]]]-First@lastTrk>=-1,
				If[debug,Sow[{"sel"->Image@ReplacePart[Table[0,dim,dim],select->1],Graphics@Line@select
				(*,"sel"->select,"win"->win,"dim"\[Rule]dim*)}];];
				(*return rescaled selection*)
				Map[#+First@Transpose@win-{1,1}-{Round[1+dim/2],Round[1+dim/2]}&,select],
				If[Mean[cands[[;;,1]]]-First@lastTrk>=-1,
					Map[#+First@Transpose@win-{1,1}-{Round[1+dim/2],Round[1+dim/2]}&,cands],{}]
			]
			,
			{}]];
		Sow[t,"pathfinding"];
		result,
	{}]];

GoBack[x_]:=Module[{i=-1,ret,trk},
While[-i<Length@x&& x[[i,1]]-x[[i-1,1]]<=0,i--];
Drop[x,i]]

GetNextCoord[trkCount_,allyx_,kym_,thr_,vismod_,debug_:False]:=Module[{foo,t,cand,trk=First@trkCount,backwrdscount=Last@trkCount,trknew,tmp},
	(*find first candidate by moving one step in any direction and delete candidates that are already part of track*)
	cand=If[Length@allyx>0,DeleteCases[Nearest[allyx,Last@trk,{All,1.5}],Alternatives@@trk],{}];
	trknew=trk;
	(*If more or less than one candidate use vision module to make decision*)
	tmp=If[Length@cand!=1,
		If[Length@trknew>2,
		{t,foo}=AbsoluteTiming@GetCand[kym,trknew,allyx,thr,vismod,debug];
		Sow[t,"getcand"];
		foo,
		{}],cand];
	(*Test if found candidate is step back in time*)
	(*reset counter if step forwards in time, Return*)
	If[Length@tmp>0 &&First@Last@tmp-First@Last@trknew>0,backwrdscount=0];
	If[Length@tmp>0 && First@Last@tmp-First@Last@trknew<0,If[ backwrdscount<1,backwrdscount++,
		trknew=GoBack@trk;(*Needed to return coordinates that are backwards in time back into Stmp*)
		tmp={}]];
	(*return new addition to track if it does not end on a previously occupied track*)
	If[Length@tmp>0 ,
		{Flatten[{trknew,tmp},1],backwrdscount},
		{Flatten[{trknew,{{0,0}}},1],backwrdscount}]]

MakeTrack[kym_,allyx_,thr_,seed_,vismod_,debug_:False]:=Module[{trk,t,return,tmp,backwrdscount,dim=ImageDimensions@kym},
	backwrdscount=0;
	{t,return}=AbsoluteTiming[
	(*find first candidate by moving one step in any direction that has not been previously occupied, then proceed as normal with getnextcoord*)
	tmp=DeleteCases[Nearest[allyx,seed,{All,1.5}],seed];
	If[Length@tmp>1,
		tmp={Last@SortBy[tmp,First]}];
	Which[Length@tmp==0,
		{seed},
		Length@tmp==1,
		trk={seed,First@tmp};
		trk=Most@First@NestWhile[GetNextCoord[#,allyx,kym,thr,vismod,debug]&,{trk,backwrdscount},Last@First@#!={0,0}&];
		Sow[1,"DecisionProb"];
		Sow["TrkDone","DecisionProb"];
		DeleteCases[trk,{_,0}|{0,_}],
		True,
		Print@tmp;
		Print@"Undef behavior";Abort[]]];
		Sow[t,"maketrack"];
		(*Delete out of bound coordinates*)
		Select[return,(0<First@#<=dim[[2]] &&0<Last@#<=dim[[1]])&]];

SmoothBin[out_]:=out+HitMissTransform[out,{\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"0", "1", "1"},
{"0", 
RowBox[{"-", "1"}], "1"},
{"0", "1", "1"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\),\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"1", "1", "1"},
{"1", 
RowBox[{"-", "1"}], "1"},
{"0", "0", "0"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\),\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"1", "1", "0"},
{"1", 
RowBox[{"-", "1"}], "0"},
{"1", "1", "0"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\),\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"0", "0", "0"},
{"1", 
RowBox[{"-", "1"}], "1"},
{"1", "1", "1"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\)},Padding->0]-HitMissTransform[out,{\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"0", 
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}]},
{"0", "1", 
RowBox[{"-", "1"}]},
{"0", 
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}]}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\),\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}]},
{
RowBox[{"-", "1"}], "1", 
RowBox[{"-", "1"}]},
{"0", "0", "0"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\),\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}], "0"},
{
RowBox[{"-", "1"}], "1", "0"},
{
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}], "0"}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\),\!\(\*
TagBox[
RowBox[{"(", GridBox[{
{"0", "0", "0"},
{
RowBox[{"-", "1"}], "1", 
RowBox[{"-", "1"}]},
{
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}], 
RowBox[{"-", "1"}]}
},
GridBoxAlignment->{"Columns" -> {{Center}}, "ColumnsIndexed" -> {}, "Rows" -> {{Baseline}}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}},
GridBoxSpacings->{"Columns" -> {Offset[0.27999999999999997`], {Offset[0.7]}, Offset[0.27999999999999997`]}, "ColumnsIndexed" -> {}, "Rows" -> {Offset[0.2], {Offset[0.4]}, Offset[0.2]}, "RowsIndexed" -> {}, "Items" -> {}, "ItemsIndexed" -> {}}], ")"}],
Function[BoxForm`e$, MatrixForm[BoxForm`e$]]]\)},Padding->0];

selectMask[x_,masks_]:=Module[{},
{x,First@First@Select[masks,MemberQ[#,x]&]}]
chewEnds[bin_]:=bin-HitMissTransform[bin,{{{-1,-1,-1},{-1,1,1},{-1,-1,-1}},{{-1,-1,-1},{1,1,-1},{-1,-1,-1}}},Padding->0]
chewAllEnds[bin_]:=Module[{old=bin,new=chewEnds@bin},While[old!=new,old=new; new=chewEnds@new;];new]
CatchStraddlers[trks_,paths_,vthr_,kym_,pathstmp_,dim_,allyx_,vismod_,debug_:False]:=Module[{seedbin,seeds,trkstmp,mask,tmp},
	(*Do the whole thing again to catch straddlers, i.e. segments that were omitted in the first round*)
	mask=Map[Keys,ArrayRules/@Values[ComponentMeasurements[{pathstmp,Map[If[#[[1]]==0,0,#[[2]]]&,Transpose/@Transpose[{MorphologicalComponents@pathstmp,MorphologicalComponents@Dilation[pathstmp,3,Padding->0]}],{2}]},"Mask"]],{2}];
	(*get seeds and coordinates*)
	seedbin=HitMissTransform[chewAllEnds@pathstmp,{{-1,-1,-1},{-1,1,-1},{0,0,0}},Padding->0];
	seeds=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[seedbin,1]],First];
	(*get a mask of all structures in the image and only select one seed (the highest) per structure*)
	tmp=Normal[Map[First,GroupBy[Map[selectMask[#,mask]&,seeds],Last@#&],{2}]];
	seeds=Flatten[Map[If[First@First@#<First@Union@#[[2,;;,1]],{First@#},Extract[#[[2]],Position[#[[2,;;,1]],First@Union@#[[2,;;,1]]]]]&,tmp],1];
	(**add coordinates from masks that have no seed in them*)
	seeds=Flatten[{seeds,Select[First/@mask,Not@MemberQ[tmp[[;;,1]],#]&]},1];
	seedbin=Image@SparseArray[seeds->1,{dim[[2]],dim[[1]]}];
	Sow@HighlightImage[pathstmp,seedbin];
	(*get tracks by calling MakeTrack onto each left over seed*)
	{t,trkstmp}=AbsoluteTiming@If[Length@seeds>0,
	Map[MakeTrack[kym,allyx,vthr,#,vismod,debug]&,seeds],{}];
	Flatten[{trks,trkstmp},1]
];

BiKymoButlerSegment[kym_,net_,tD_]:=Module[{bool,kympreproc,pred,dim=ImageDimensions@kym},
	kympreproc=ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym,"Grayscale"];
	(*ColorNegate if backgroudn white*)
	bool=isNegated@kympreproc;
	kympreproc=If[bool,ColorNegate@kympreproc,kympreproc];
	(*normalize kymolines*)
	kympreproc=normlines@kympreproc;
	pred=net[{ImageData@ImageResize[kympreproc,16*Round@N[dim/16]]},TargetDevice->tD];
	If[FailureQ@pred,$Failed,
	{bool,kym,kympreproc,Image@pred}]
];

BiKymoButlerTrack[pred_,kym_,kympreproc_,bool_,binthresh_,vthr_,vismod_,minSz_,minFr_,debug_:False]:=Module[
{dim=ImageDimensions@kym,coloredlines,overlay,overlaylabeled,labels,c,ptmp,sel,ovlpIDs,out,paths,pathstmp,inflp,t,seedbin,seeds,allyx,ptrk,trks,checkifAnyTrkisSubset,ovlpSegID},
	Sow[First@AbsoluteTiming[
	out=ImageResize[Binarize[pred,binthresh],dim];
	out=SmoothBin@SmoothBin@out;
	paths=SelectComponents[Pruning[Thinning@out,3],#Count>=minSz &&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>=minFr&];

	(*get seeds and coordinates*)
	seedbin=HitMissTransform[chewAllEnds@paths,{{-1,-1,-1},{-1,1,-1},{0,0,0}},Padding->0];
	seeds=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[seedbin,1]],First];
	allyx=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[paths(*-seedbin*),1]],First];
	Sow@HighlightImage[paths,seedbin];],"skeletonization"];

	(*get tracks by calling MakeTrack onto each seed, also reap all decision probabilities*)
	
	ptrk=Last@Reap[
		Sow[First@AbsoluteTiming[
		trks=Map[MakeTrack[kympreproc,allyx,vthr,#,vismod,debug]&,seeds];
		pathstmp=SelectComponents[paths-Image@SparseArray[Flatten[trks,1]->1,{dim[[2]],dim[[1]]}],#Count>5 &&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>=3&];],"firstmaketracks"];
		inflp=1;
		t=First@AbsoluteTiming@While[Total@pathstmp>5,
			inflp++;
			If[inflp>500,Print@"Warning! Inf Loop Abort!!";Print@trks;Print@pathstmp;Break[]];
			trks=CatchStraddlers[trks,paths,vthr,kympreproc,pathstmp,dim,allyx,vismod,debug];
			pathstmp=SelectComponents[paths-Image@SparseArray[Flatten[trks,1]->1,{dim[[2]],dim[[1]]}],#Count>5 &&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>=3&];
		];
	,"DecisionProb"];
	Sow[t,"catchstraddlers"];
	
	If[Length@trks>0,

	ptrk=Mean/@Select[SplitBy[Most@Last@ptrk,NumberQ],AllTrue[#,NumberQ]&];
	trks=Map[Which[#[[1]]>dim[[2]],{dim[[2]],#[[2]]},#[[1]]<1,{1,#[[2]]},True,#]&,trks,{2}];
	trks=Map[Which[#[[2]]>dim[[1]],{#[[1]],dim[[1]]},#[[2]]<1,{#[[1]],1},True,#]&,trks,{2}];

	
	(*Round tracks for each timepoint*)
	trks=Map[Round/@Mean/@GatherBy[#,First]&,trks];
	
	If[debug,Sow[trks,"trkspreSubset"]];
	
	If[debug,Sow@Rotate[Graphics@Transpose@{RandomColor[Length@trks],Line/@trks},-90 Degree]];
	
	Sow[First@AbsoluteTiming[
	(*delete tracks that are subsets of other tracks*)
	checkifAnyTrkisSubset[trks_,i_]:=MapThread[Abs[#1-#2]<=minSz && #2<Length@trks[[i]]&,{ReplacePart[Map[Length@Intersection[trks[[i]],#]&,trks],i->0],Length/@trks}];
	sel=Nor@@@Transpose[checkifAnyTrkisSubset[trks,#]&/@Range[Length@trks]];
	trks=Pick[trks,sel];
	Sow@{ptrk,sel};
	ptrk=Pick[ptrk,sel];
	
	If[debug,Sow@Rotate[Graphics@Transpose@{RandomColor[Length@trks],Line/@trks},-90 Degree]];
	
	(*resolve overlaps*)
	ovlpSegID[id_,trks_]:=Position[Map[Length@Intersection[trks[[id]],#]>10&,Drop[trks,id]],True]+id;
	ovlpIDs=Select[Flatten/@Transpose[{Range[Length@trks],ovlpSegID[#,trks]&/@Range[Length@trks]}],Length@#>1&];
	While[Length@ovlpIDs>0,
		Do[
			ptmp=Extract[ptrk,Partition[ovlpIDs[[i]],1]];
			trks=ReplacePart[trks,Map[#->DeleteCases[trks[[#]],Alternatives@@trks[[ovlpIDs[[i,MaxIndx@ptmp]]]]]&,Delete[ovlpIDs[[i]],MaxIndx@ptmp]]],
			{i,Length@ovlpIDs}
		];
		ovlpIDs=Select[Flatten/@Transpose[{Range[Length@trks],ovlpSegID[#,trks]&/@Range[Length@trks]}],Length@#>1&];
	];],"resolveovlps"];
	
	If[debug,Sow@Rotate[Graphics@Transpose@{RandomColor[Length@trks],Line/@trks},-90 Degree]];
	
	(*delete tracks that are subsets of other tracks again*)
	checkifAnyTrkisSubset[trks_,i_]:=MapThread[Abs[#1-#2]<=minSz && #2<Length@trks[[i]]&,{ReplacePart[Map[Length@Intersection[trks[[i]],#]&,trks],i->0],Length/@trks}];
	sel=Nor@@@Transpose[checkifAnyTrkisSubset[trks,#]&/@Range[Length@trks]];
	trks=Pick[trks,sel];
	Sow@{ptrk,sel};
	ptrk=Pick[ptrk,sel];

	If[debug,Sow@Rotate[Graphics@Transpose@{RandomColor[Length@trks],Line/@trks},-90 Degree]];
	
	(*Split tracks with ridiculous gaps*)
	trks=Flatten[Split[#,#2[[1]]-#1[[1]]<=2*minSz&]&/@trks,1];

	(*clear tracks that got too short*)
	(*trks=Select[trks,Length@#>=minSz&];*)
	trks=Select[trks,First@Last@#-First@First@#>=minFr&];
	
	Sow[First@AbsoluteTiming[
		(*colored lines and overlays*)
		coloredlines=ImageRotate[Image[Show[Image@Table[0,dim[[1]],dim[[2]]],Graphics@Map[{RandomColor[],Style[Line@#,Antialiasing->False]}&,Map[#-{1,0}&,trks,{2}]]],ImageSize->Reverse@dim],-Pi/2];
		overlay=ImageCompose[ImageAdjust@kym,RemoveBackground[coloredlines,{Black,.01}]];
		(*get labels and label overlay*)
		c=SparseArray[Flatten[Map[Thread[trks[[#]]->#]&,Range@Length@trks],1],{dim[[2]],dim[[1]]}];
		
		labels=ComponentMeasurements[c,"Centroid"];
		(*overlaylabeled=Image[HighlightImage[overlay,Map[ImageMarker[labels[[#,2]]+{0,5},Graphics[{If[bool,Black,White],Text[Style[ToString@#,FontSize->Scaled@.03]]}]]&,Range[Length@labels]]],ImageSize\[Rule]dim];*)
		overlaylabeled=HighlightImage[overlay,Map[ImageMarker[labels[[#,2]]+{0,5},Graphics[{If[bool,Black,White],Text[Style[ToString@#,FontSize->Scaled@.03]]}]]&,Range[Length@labels]]];
		
		],"colorize"];
	{kym,coloredlines,overlay,overlaylabeled,trks},
	 {kym,Image@Table[0,dim[[2]],dim[[1]]],ImageAdjust@kym,ImageAdjust@kym,{}}
	 ]];
	
	BiKymoButler[kym_,binthresh_,vthr_,tD_,cnet_,vismod_,minSz_,minFr_,debug_:False]:=Module[
	{rawkym,kympreproc,pred,bool,t,out},
	{t,{bool,rawkym,kympreproc,pred}}=AbsoluteTiming@BiKymoButlerSegment[kym,cnet,tD];
	If[debug,Sow[pred,"pmap"]];
	Sow[t,"seg"];
	{t,out}=AbsoluteTiming@BiKymoButlerTrack[pred,rawkym,kympreproc,bool,binthresh,vthr,vismod,minSz,minFr,debug];
	Sow[t,"tracking"];
	out
];


(* ::Section:: *)
(*Benchmarking Functions*)


findclosetrk[nf_,trk_]:=DeleteDuplicates@Flatten[Map[nf[#,{All,3.2}]&,trk],1]

rec[nfP_,syntrk_]:=Module[{tmp},
tmp=SortBy[findclosetrk[#,syntrk]&/@nfP,Length];
If[Length@Flatten@tmp>0,
1-Abs[Length[Last@tmp]-Length[syntrk]]/Length[syntrk],0]
]

prec[nfGT_,predtrk_]:=Module[{tmp},
tmp=SortBy[findclosetrk[#,predtrk]&/@nfGT,Length];
If[Length@Flatten@tmp>0,
1-Abs[Length[Last@tmp]-Length[predtrk]]/Length[predtrk],0]
]

getrecall[nfP_,trks_]:=N@Mean[rec[nfP,#]&/@trks];

benchmarkPrediction[kym_,ptrks_,trks_]:=Module[{pred,nfP,nfGT,precision,recall},
(*formulate  NN function for pred*)
nfP=Map[Nearest[#]&,Select[ptrks,Length@#>0&]];
(*formulate  function for gt*)
nfGT=Map[Nearest[#]&,Select[trks,Length@#>0&]];
precision=N@Mean[prec[nfGT,#]&/@ptrks];
recall=N@Mean[rec[nfP,#]&/@trks];
{kym,ptrks,"recall"->recall,"precision"->precision,"F1"->2*(precision*recall)/(precision+recall)}
]




End[]
EndPackage[]

