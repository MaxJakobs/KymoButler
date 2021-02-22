(* ::Package:: *)

BeginPackage["WaveletFilterKymoAnal`"]
AnalyseKymographBIwavelet::usage="analyse kymograph with wavelets"
Begin["`Private`"]
MaxIndx[a_]:=First@First@Position[a,Max@a]
MinIndx[a_]:=First@First@Position[a,Min@a]

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
First[Position[bindat,#]]&/@path,{}]];

SortCoords[x_]:=Module[{outL={x[[1]]},outR={x[[1]]},xtmp=x[[2;;]],foo},
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
(*{outR,outL}*)
Last@SortBy[{outR,outL},Length]];

GetTile[kym_,trk_,allyx_,vismoddim_]:=Module[{dim=ImageDimensions@kym,win},
win=Transpose@{Round[Last@trk-(vismoddim/2)],Round[Last@trk+(vismoddim/2-1)]};
(*if boundary rescale*)
win={Which[Min@win[[1]]<=0,win[[1]]-Min@win[[1]]+1,
Max@win[[1]]>dim[[2]],win[[1]]-(Max@win[[1]]-dim[[2]]),
True,win[[1]]],
Which[Min@win[[2]]<=0, win[[2]]-Min@win[[2]]+1,
Max@win[[2]]>dim[[1]], win[[2]]-(Max@win[[2]]-dim[[1]]),
True,win[[2]]]};
(*return tile binary rescaled candidate*)
{ImageAdjust@ImageTake[kym,win[[1]],win[[2]]],Image@Take[ReplacePart[Table[0,dim[[2]],dim[[1]]],Round/@trk->1],win[[1]],win[[2]]],
Image@Take[ReplacePart[Table[0,dim[[2]],dim[[1]]],Round/@allyx->1],win[[1]],win[[2]]],win}
];

GetCandFromPmap[pmap_,thr_]:=Module[{
comp=ComponentMeasurements[Binarize[pmap,thr],{"Mask","Count"}],
maxA},
If[Length@comp>0(*&Length@comp<4*),
maxA=MaxIndx@Map[#[[2,2]]&,comp];
(*return candidates with maximum area*)
Sow[Total[Image[comp[[maxA,2,1]]]*pmap]/Total[Image[comp[[maxA,2,1]]]],"DecisionProb"];
Position[Round@ImageData@Thinning@Image@comp[[maxA,2,1]],1],
{}]];

GetCandLinpred[bin_,fullbin_]:=Module[{
comp=ComponentMeasurements[Binarize[ImageReflect[Dilation[ImagePad[ImageTake[bin,{1,25}],{{0,0},{23,0}},"Reflected"]-bin,1],Left]*fullbin-bin,.5],{"Mask","Count"}],
maxA},
If[Length@comp>0(*&Length@comp<4*),
maxA=MaxIndx@Map[#[[2,2]]&,comp];
(*return candidates with maximum area*)
Sow[RandomReal[],"DecisionProb"];
Position[Round@ImageData@Thinning@Image@comp[[maxA,2,1]],1],
{}]];

GetCand[kym_,vismod_,trk_,allyx_,thr_]:=Module[{
(*find all candidates 8 pixels away*)
padkym,tmp,tile,allyxtmp,allyxR,bin,trkR,rc,pmap,win,cands,select,cost,fullbin,lastTrk,trktmp,dim,shortestpath},
(*rescale everything to a padded kymograph*)
dim=Last@First@NetInformation[vismod,"InputPorts"];
padkym=ImagePad[kym,Round[1+dim/2],.1];
allyxtmp=allyx+Round[1+dim/2];
trktmp=trk+Round[1+dim/2];
allyxtmp=SortBy[Nearest[allyxtmp,Last@trktmp,{All,dim*1.5}],First];
If[Length@allyxtmp>0,
trktmp=Drop[trktmp,-1];

{t,{tile,bin,fullbin,win}}=AbsoluteTiming@GetTile[padkym,trktmp,allyxtmp,dim];
{t,pmap}=AbsoluteTiming@Image@Map[Last,vismod@Map[ImageData[#,Interleaving->False]&,{ImageAdjust@tile,bin,fullbin}],{2}];
(*rescale allyxtmp and track*)
allyxR=Map[#-First@Transpose@win+{1,1}&,allyxtmp];
trkR=Map[#-First@Transpose@win+{1,1}&,trktmp];
lastTrk=Last@trktmp-First@Transpose@win+{1,1};
(*get coordinates of largest connected object*)
(*cands=GetCandFromPmap[pmap,thr];*)
(*get coordinates of largest connected object with simple linear prediction*)
cands=GetCandLinpred[bin,fullbin];
(*Delete coords that we know allready*)
cands=Complement[cands,trktmp];

Sow[{tile,bin,(*allyxR,lastTrk,cands,*)Image@ReplacePart[Table[0,dim,dim],Select[allyxR,Min@#>0&&Max@#< dim+1&]->1],pmap,Image@ReplacePart[Table[0,dim,dim],cands->1](*,Image@ReplacePart[Table[0,dim,dim],select\[Rule]1]*)}];
(*Select from all*)
(*Roughly Sort candidates by distance to last entry in trk, additionally sort so that they are one connected line from start to end*)
cands=SortBy[cands,N@EuclideanDistance[#,lastTrk]&];
If[Length@cands>2 &&EuclideanDistance[lastTrk,First@cands]<15&&Mean[cands[[;;,1]]]-First@lastTrk>=-1,(*Dont use dots as predictions, has to be more than 2 also remove crazy prediction*)
cands=SortCoords@cands;
(*do pathfinding to fill any gaps in the prediction*)
select=If[EuclideanDistance[lastTrk,First@cands]>1.5,
shortestpath=FindShortPathImage[Image@ReplacePart[Table[0,dim,dim],Select[allyxR,Min@#>0&&Max@#<dim+1&]->1],lastTrk,First@cands];
Join[shortestpath,cands]
,cands];
(*Only select at most 24 coordinates*)
select=Take[select,UpTo@24];
(*Delete selection of completely stupid pathfinding results*)
If[Mean[select[[;;,1]]]-First@lastTrk>=-1,
Sow[{"sel"->Image@ReplacePart[Table[0,dim,dim],select->1],Graphics@Line@select(*,"sel"->select,"win"->win,"dim"\[Rule]dim*)}];
(*return rescaled selection*)
Map[#+First@Transpose@win-{1,1}-{Round[1+dim/2],Round[1+dim/2]}&,select],
If[Mean[cands[[;;,1]]]-First@lastTrk>=-1,
Map[#+First@Transpose@win-{1,1}-{Round[1+dim/2],Round[1+dim/2]}&,cands],{}]
]
,
{}],
{}]]

GoBack[x_]:=Module[{i=-1,ret,trk},
While[-i<Length@x&& x[[i,1]]-x[[i-1,1]]<=0,i--];
Drop[x,i]]

GetNextCoord[trkCount_,allyx_,kym_,vismod_,thr_]:=Module[{cand,trk=First@trkCount,backwrdscount=Last@trkCount,trknew,tmp},
(*find first candidate by moving one step in any direction and delete candidates that are already part of track*)
cand=If[Length@allyx>0,DeleteCases[Nearest[allyx,Last@trk,{All,1.5}],Alternatives@@trk],{}];
trknew=trk;
(*If more than one candidate use vision module to make decision*)
tmp=If[Length@cand>1,
If[Length@trknew>2,
GetCand[kym,vismod,trknew,allyx,thr],
{}],cand];
(*Test if found candidate is step back in time*)
(*reset counter if step forwards in time, Return *)
If[Length@tmp>0 &&First@Last@tmp-First@Last@trknew>0,backwrdscount=0];
If[Length@tmp>0 && First@Last@tmp-First@Last@trknew<0,If[ backwrdscount<1,backwrdscount++,
trknew=GoBack@trk;(*Needed to return coordinates that are backwards in time back into Stmp*)
tmp={}]];
(*return new addition to track if it does not end on a previously occupied track*)
If[Length@tmp>0 ,
{Flatten[{trknew,tmp},1],backwrdscount},
{Flatten[{trknew,{{0,0}}},1],backwrdscount}]]

MakeTrack[kym_,allyx_,vismod_,thr_,seed_]:=Module[{trk,tmp,backwrdscount},
backwrdscount=0;
(*find first candidate by moving one step in any direction that has not been previously occupied, then proceed as normal with getnextcoord*)
tmp=DeleteCases[Nearest[allyx,seed,{All,1.5}],seed];
If[Length@tmp>1,
tmp={Last@SortBy[tmp,First]}];
Which[Length@tmp==0,
{seed},
Length@tmp==1,
trk={seed,First@tmp};
trk=Most@First@NestWhile[GetNextCoord[#,allyx,kym,vismod,thr]&,{trk,backwrdscount},Last@First@#!={0,0}&];
Sow[1,"DecisionProb"];
Sow["TrkDone","DecisionProb"];
DeleteCases[trk,{_,0}|{0,_}],
True,
Print@tmp;
Print@"Undef behavior";Abort[]]]

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
CatchStraddlers[trks_,paths_,vismod_,vthr_,kym_,pathstmp_,dim_,allyx_]:=Module[{seedbin,seeds,xtraseeds,trkstmp,mask,tmp},
(*Do the whole thing again to catch straddlers, i.e. segments that were omitted in the first round*)
mask=Map[Keys,ArrayRules/@Values[ComponentMeasurements[{pathstmp,Map[If[#[[1]]==0,0,#[[2]]]&,Transpose/@Transpose[{MorphologicalComponents@pathstmp,MorphologicalComponents@Dilation[pathstmp,3,Padding->0]}],{2}]},"Mask"]],{2}];
(*get seeds and coordinates*)
seedbin=HitMissTransform[chewAllEnds@pathstmp,{{-1,-1,-1},{-1,1,-1},{0,0,0}},Padding->0];
seeds=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[seedbin,1]],First];
(*get a mask of all structures in the image and only select one seed (the highest) per structure*)
tmp=Normal[Map[First,GroupBy[Map[selectMask[#,mask]&,seeds],Last@#&],{2}]];
seeds=Flatten[Map[If[First@First@#<Min@#[[2,;;,1]],{First@#},Extract[#[[2]],Position[#[[2,;;,1]],Min@#[[2,;;,1]]]]]&,tmp],1];
(**add coordinates from masks that have no seed in them*)
seeds=Flatten[{seeds,Select[First/@mask,Not@MemberQ[tmp[[;;,1]],#]&]},1];
seedbin=Image@ReplacePart[Table[0,dim[[2]],dim[[1]]],seeds->1];
Sow@HighlightImage[pathstmp,seedbin];
(*get tracks by calling MakeTrack onto each left over seed*)
{t,trkstmp}=AbsoluteTiming@If[Length@seeds>0,
Map[MakeTrack[kym,allyx,vismod,vthr,#]&,seeds],{}];
Flatten[{trks,trkstmp},1]];

(*AnalyseKymographBI[kym_,dim_,binthresh_,cnet_,vismod_,vthr_,td_]:=Module[
{seeds,seedbin,out,tmp,tmpkym,pred,trks,c,allyx,labels,paths,overlay,overlaylabeled,cf,bool,pathstmp,ptrk,ovlpIDs,ptmp,sel,inflp,coloredlines},
tmpkym=ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym,"Grayscale"];
(*ColorNegate if backgroudn white*)
bool=isNegated@tmpkym;
tmp=If[bool,ColorNegate@tmpkym,tmpkym];
(*normalize kymolines*)
tmp=ImageAdjust@Image@Map[#/Mean@#&,ImageData@tmp];
Sow@tmp;
(*Run Kymo Butler*)
{t,pred}=AbsoluteTiming@Image@cnet[{ImageData@ImageResize[tmp,16*Round@N[dim/16]]},TargetDevice->td];
Sow@pred;
out=ImageResize[Binarize[pred,binthresh],dim];
pred=ImageResize[pred,dim];
(*out=ImageResize[Binarize[pred,binthresh],dim];*)
out=SmoothBin@SmoothBin@out;
paths=SelectComponents[Pruning[Thinning@out,3],#Count>3&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>2&];

(*get seeds and coordinates*)
seedbin=HitMissTransform[chewAllEnds@paths,{{-1,-1,-1},{-1,1,-1},{0,0,0}},Padding->0];
seeds=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[seedbin,1]],First];
allyx=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[paths(*-seedbin*),1]],First];
Sow@HighlightImage[paths,seedbin];
(*get tracks by calling MakeTrack onto each seed, also reap all decision probabilities*)
ptrk=Last@Reap[
trks=Map[MakeTrack[tmp,allyx,vismod,vthr,#]&,seeds];
pathstmp=SelectComponents[paths-Image@ReplacePart[Table[0,dim[[2]],dim[[1]]],Flatten[trks,1]->1],#Count>5&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>2&];
inflp=1;
{t,null}=AbsoluteTiming@While[Total@pathstmp>5,
inflp++;
If[inflp>100,Print@"Warning! Inf Loop Abort!!";Break[]];
trks=CatchStraddlers[trks,paths,vismod,vthr,tmp,pathstmp,dim,allyx];
pathstmp=SelectComponents[paths-Image@ReplacePart[Table[0,dim[[2]],dim[[1]]],Flatten[trks,1]->1],#Count>5&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>2&];];
,"DecisionProb"];
ptrk=Mean/@Select[SplitBy[Most@Last@ptrk,NumberQ],AllTrue[#,NumberQ]&];
trks=Map[Which[#[[1]]>dim[[2]],{dim[[2]],#[[2]]},#[[1]]<1,{1,#[[2]]},True,#]&,trks,{2}];
trks=Map[Which[#[[2]]>dim[[1]],{#[[1]],dim[[1]]},#[[2]]<1,{#[[1]],1},True,#]&,trks,{2}];

(*Round tracks for each timepoint*)
trks=Map[Round/@Mean/@GatherBy[#,First]&,trks];

(*delete tracks that are subsets of other tracks*)
checkifAnyTrkisSubset[trks_,i_]:=MapThread[#1==#2&,{ReplacePart[Map[Length@Intersection[trks[[i]],#]&,trks],i->0],Length/@trks}];
sel=Nor@@@Transpose[checkifAnyTrkisSubset[trks,#]&/@Range[Length@trks]];
trks=Pick[trks,sel];
ptrk=Pick[ptrk,sel];
(*resolve overlaps*)
ovlpSegID[id_,trks_]:=Position[Map[Length@Intersection[trks[[id]],#]>10&,Drop[trks,id]],True]+id;
ovlpIDs=Select[Flatten/@Transpose[{Range[Length@trks],ovlpSegID[#,trks]&/@Range[Length@trks]}],Length@#>1&];
While[Length@ovlpIDs>0,
Do[
ptmp=Extract[ptrk,Partition[ovlpIDs[[i]],1]];
trks=ReplacePart[trks,Map[#->DeleteCases[trks[[#]],Alternatives@@trks[[ovlpIDs[[i,MaxIndx@ptmp]]]]]&,Delete[ovlpIDs[[i]],MaxIndx@ptmp]]],
{i,Length@ovlpIDs}
];
ovlpIDs=Select[Flatten/@Transpose[{Range[Length@trks],ovlpSegID[#,trks]&/@Range[Length@trks]}],Length@#>1&];];

(*clear tracks that got too short*)
trks=Select[trks,Length@#>3&];
trks=Select[trks,First@Last@#-First@First@#>2&];

(*colored lines and overlays*)
coloredlines=Dilation[ImageRotate[Rasterize[Show[Image@Table[0,dim[[1]],dim[[2]]],Graphics@Map[{RandomColor[],Style[Line@#,Antialiasing->False]}&,Map[#-{1,0}&,trks,{2}]]]],-Pi/2],1];
overlay=ImageCompose[tmpkym,RemoveBackground[coloredlines,{Black,.01}]];
(*get labels and label overlay*)
c=ReplacePart[Table[0,dim[[2]],dim[[1]]],Map[trks[[#]]->#&,Range@Length@trks]];
labels=ComponentMeasurements[c,"Centroid"];
overlaylabeled=HighlightImage[overlay,Map[ImageMarker[labels[[#,2]]+{0,5},Graphics[{If[bool,Black,White],Text[Style[ToString@#,FontSize->Scaled@.04]]}]]&,Range[Length@labels]]];

{tmpkym,coloredlines,overlay,overlaylabeled,trks}]
*)


AnalyseKymographBIwavelet[kym_,dim_,binthresh_,cnet_,vismod_,vthr_,minSz_,minT_]:=Module[
{seeds,seedbin,out,tmp,pred,trks,c,allyx,labels,paths,overlay,overlaylabeled,cf,bool,pathstmp,ptrk,ovlpIDs,ptmp,sel,inflp,dwd},
If[dim[[1]]<=5000&&dim[[2]]<=5000,
tmp=ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym,"Grayscale"];
(*ColorNegate if backgroudn white*)
bool=Mean@Flatten@ImageData@tmp>.75;
tmp=If[bool,ColorNegate@tmp,tmp];
(*normalize kymolines*)
tmp=ImageAdjust@Image@Map[#/Mean@#&,ImageData@tmp];
(*Run Kymo Butler*)

dwd=StationaryWaveletTransform[tmp,Automatic,2];
paths=SelectComponents[Thinning@DeleteSmallComponents[Pruning[Thinning@Dilation[Binarize[ImageAdjust@Total@Values@dwd[{{0},{1},{2},{0,0},{0,2},{0,1}},"Image"],binthresh],1],5],5],#Count>=minSz&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>=minT&];
Sow@paths;



(*get seeds and coordinates*)
seedbin=HitMissTransform[chewAllEnds@paths,{{-1,-1,-1},{-1,1,-1},{0,0,0}},Padding->0];
seeds=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[seedbin,1]],First];
allyx=SortBy[Map[Abs[{dim[[2]]+1,0}-#]&,Reverse/@PixelValuePositions[paths(*-seedbin*),1]],First];
Sow@HighlightImage[paths,seedbin];
(*get tracks by calling MakeTrack onto each seed, also reap all decision probabilities*)
ptrk=Last@Reap[
trks=Map[MakeTrack[tmp,allyx,vismod,vthr,#]&,seeds];
pathstmp=SelectComponents[paths-Image@ReplacePart[Table[0,dim[[2]],dim[[1]]],Flatten[trks,1]->1],#Count>5&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>2&];
inflp=1;
{t,null}=AbsoluteTiming@While[Total@pathstmp>5,
inflp++;
If[inflp>100,Print@"Warning! Inf Loop Abort!!";Break[]];
trks=CatchStraddlers[trks,paths,vismod,vthr,tmp,pathstmp,dim,allyx];
pathstmp=SelectComponents[paths-Image@ReplacePart[Table[0,dim[[2]],dim[[1]]],Flatten[trks,1]->1],#Count>5&&#BoundingBox[[2,2]]-#BoundingBox[[1,2]]>2&];];
,"DecisionProb"];
ptrk=Mean/@Select[SplitBy[Most@Last@ptrk,NumberQ],AllTrue[#,NumberQ]&];
trks=Map[Which[#[[1]]>dim[[2]],{dim[[2]],#[[2]]},#[[1]]<1,{1,#[[2]]},True,#]&,trks,{2}];
trks=Map[Which[#[[2]]>dim[[1]],{#[[1]],dim[[1]]},#[[2]]<1,{#[[1]],1},True,#]&,trks,{2}];
(*colorize*)
c=ReplacePart[Table[0,dim[[2]],dim[[1]]],Map[trks[[#]]->#&,Range@Length@trks]];
overlay=ImageCompose[tmp,RemoveBackground[Colorize@c,{Black,.0001}]];
(*label overlay*)
labels=ComponentMeasurements[c,"Centroid"];
overlaylabeled=HighlightImage[overlay,Map[ImageMarker[labels[[#,2]]+{0,5},Graphics[{White,Text[Style[ToString@#,FontSize->Scaled@.03]]}]]&,Range[Length@labels]]];

(*Round tracks for each timepoint*)
trks=Map[Round/@Mean/@GatherBy[#,First]&,trks];

(*delete tracks that are subsets of other tracks*)
checkifAnyTrkisSubset[trks_,i_]:=MapThread[#1==#2&,{ReplacePart[Map[Length@Intersection[trks[[i]],#]&,trks],i->0],Length/@trks}];
sel=Nor@@@Transpose[checkifAnyTrkisSubset[trks,#]&/@Range[Length@trks]];
trks=Pick[trks,sel];
ptrk=Pick[ptrk,sel];
(*resolve overlaps*)
ovlpSegID[id_,trks_]:=Position[Map[Length@Intersection[trks[[id]],#]>10&,Drop[trks,id]],True]+id;
ovlpIDs=Select[Flatten/@Transpose[{Range[Length@trks],ovlpSegID[#,trks]&/@Range[Length@trks]}],Length@#>1&];
While[Length@ovlpIDs>0,
Do[
ptmp=Extract[ptrk,Partition[ovlpIDs[[i]],1]];
trks=ReplacePart[trks,Map[#->DeleteCases[trks[[#]],Alternatives@@trks[[ovlpIDs[[i,MaxIndx@ptmp]]]]]&,Delete[ovlpIDs[[i]],MaxIndx@ptmp]]],
{i,Length@ovlpIDs}
];
ovlpIDs=Select[Flatten/@Transpose[{Range[Length@trks],ovlpSegID[#,trks]&/@Range[Length@trks]}],Length@#>1&];];

(*clear tracks that got too short*)
(*trks=Select[trks,Length@#>5&];*)
trks=Select[trks,First@Last@#-First@First@#>=minT&];
{tmp,overlay,overlaylabeled,trks},
"Image too large! Try a smaller one or upgrade to paid version"]];

End[]
EndPackage[]



