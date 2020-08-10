#!/usr/bin/env wolframscript
(* ::Package:: *)

Get["https://raw.githubusercontent.com/mark-caprio/CustomTicks/master/CustomTicks.m"]
Needs["CustomTicks`"]
Needs["ErrorBarPlots`"]
BootstrapMean[data_]:=Module[
{ResampledList,ResampledMeanDist,m,err},
(*Non-resampled bootstrap*)
n=10^4;
ResampledList=ArrayReshape[RandomChoice[data,Length[data] n],{Length[data],n}];
ResampledMeanDist=Total[ResampledList]/Length[data];(*Table[Mean[ResampledList[[i]]],{i,Length[ResampledList]}];*)
m=Mean[ResampledMeanDist];
err=Quantile[ResampledMeanDist,{0.025,0.975}];
(*err={m+StandardDeviation[ResampledMeanDist],m-StandardDeviation[ResampledMeanDist]};*)
Return[N[{m,err}]];
];
Shape[shape_,in_,out_:Black,size_:15,opacity_:Opacity[1],thickness_:Thin]:=
Which[shape=="Square",Graphics[{in,opacity,EdgeForm[{thickness,out}],Rectangle[]},PlotRangePadding->0,ImageSize->size],
shape=="Circle",
Graphics[{in,opacity,EdgeForm[{thickness,out}],Disk[]},PlotRangePadding->0,ImageSize->size],
shape=="Triangle",
Graphics[{in,opacity,EdgeForm[{thickness,out}],Polygon[{{1,0},{0,Sqrt[3]},{-1,0}}]},PlotRangePadding->0,ImageSize->size],
shape=="UDTriangle",
Graphics[{in,Opacity[1],EdgeForm[{AbsoluteThickness[0],out}],Polygon[{{1,Sqrt[3]/2},{0,-Sqrt[3]/2},{-1,Sqrt[3]/2}}]},PlotRangePadding->0,ImageSize->size],
shape=="Diamond",
Graphics[{in,opacity,EdgeForm[{thickness,out}],Polygon[{{1,0},{0,1},{-1,0},{0,-1}}]},PlotRangePadding->0,ImageSize->size]
];
Do[
Print[dataset];
dir="~/Dropbox/owl_in_one/Test-Train_MGT/Features+MGT/"<>dataset<>"/";
SetDirectory[dir];
Do[
If[dataset!="TILES"&&eventtype!="event_mgt",Continue[]];
effects=Import["MGT_day_dm.csv"];
(*heartrate, breathing rate:
- Mean.HeartRatePPG.FitBit-Heart-Rate
-Mean.HeartRate.OMSignal
-Mean.BreathingDepth.OMSignal
-Mean.BreathingRate.OMSignal
*)
If[dataset=="TILES",
events=Import["MGT_day_gt_appended.csv"];
removeInd={239, 333, 415, 469, 695, 705, 716, 758, 849, 853, 871, 874, 875, 885, 887, 1049, 1217, 1249, 1384, 1386, 1539, 1636, 1637, 1639, 1729, 1751, 1759, 1766, 1773, 1775, 1776, 1778, 1780, 1785, 1786, 1802, 1813, 1814, 1863, 1867, 2075, 2082, 2119, 2132, 2229, 2239, 2720, 2773, 2860, 2870, 2874, 2879, 2903, 3190, 3485, 3962, 3964, 3966, 3968, 3969, 3985, 4145, 4147, 4204, 4210, 4238, 4239, 4366, 4712, 4751, 4752, 4781, 4890, 4891, 4893, 4898, 4903, 4905, 4907, 4909, 4912, 4924, 4929, 5117, 5253, 5424, 5610, 6114, 6623, 6810, 6820, 6824, 6828, 7066, 7072, 7103, 7518, 7525, 7526, 7533, 7543, 7587, 7657, 7741, 9016, 9035, 9045, 9147, 9432, 9598, 9600, 9627, 10126, 10675, 11124, 11155, 11161, 11184, 11192, 11194, 11195, 11197, 11198, 11263, 11268, 11293, 11295, 11299};
,
(*if not hospital data*)
events=Import["MGT_day_gt.csv"];
removeInd={121, 175, 192, 301, 675, 962, 974, 976, 1096, 1098, 1114, 1116, 1173, 1269, 1297, 1318, 1359, 1566, 1571, 1604, 1607, 1611, 1619, 1626, 1723, 1726, 1729, 1866, 2068, 2202, 2211, 2260, 2270, 2303, 2316, 2409, 2412, 2442, 2512, 2609, 2610, 2696, 2889, 2892, 3615, 3768, 3912, 3969, 3972, 4016, 4017, 4105, 4107, 4134, 4138, 4154, 4157, 4265, 4310, 4416, 4435, 4437, 4465, 4466, 4679, 4695, 4715, 4720, 4745, 4765, 4824, 4825, 4834, 4837, 4852, 4872, 4896, 4898, 4910, 4938, 4972, 4974, 4975, 4981, 4986, 4994, 4996, 4999, 5178, 5201, 5244, 5246, 5247, 5283, 5325, 5333, 5379, 5396, 5403, 5406, 5410, 5418, 5424, 5463, 5464, 5466, 5467, 5600, 5842, 5946, 5954, 5959, 5968, 5970, 5971, 5973, 6010, 6182, 6187, 6374, 6565, 6620, 6624, 6626, 6816, 6858, 6959, 6975, 6976, 7077, 7082, 7087, 7093, 7118, 7334, 7338, 7435, 7456, 7460, 7706, 7710, 7900, 8131, 8140, 8212, 8214, 8216, 8237, 8539, 8540, 8545, 8550, 8732, 8954, 8994, 9073, 9196, 9198, 9264, 9337, 9354, 9442, 9470, 9491, 9493, 9495, 9497, 9540, 9548, 9554, 9572, 9578, 9603, 9625, 9669, 9677, 9679, 9681, 9683, 9690, 9693, 9706, 9714, 9716, 9718, 9719, 9771, 9879, 9957, 9971, 9977, 10011, 10058, 10149, 10151, 10183, 10236, 10419, 10428, 10436, 10447, 10559, 10578, 10613, 10647, 10660, 10701, 10867, 10960, 10962, 10965, 11002, 11020, 11113, 11206, 11215, 11226, 11306, 11927, 11938, 11940, 11958, 11960, 11975, 11979, 11989, 12091, 12218, 12220, 12309, 12354, 12516, 12541, 12698, 12705, 12722, 12723, 12736, 12755, 12882, 12884, 12910, 13104, 13243, 13257, 13280, 13281};
];

events=Delete[events,Partition[removeInd,1]];
constructs=Select[events[[1]],StringContainsQ[#,"_mgt"]&][[;;-3]];
(*event_mgt,isBusy?,isSick?,Patient-related?,death-related?*)


eventpos=Position[events[[1]],eventtype][[1,1]];
Print[eventpos];
timepos=Position[events[[1]],"Timestamp"][[1,1]];(*Date*)
Print[timepos];
uidpos=Position[events[[1]],"uid"][[1,1]];
Print[uidpos];

Do[
If[
StringContainsQ[effect,"OMSignal"]&&dataset!="TILES",Continue[];];
Print[effect];
If[ContainsAny[{"pos_af_mgt","neg_af_mgt","anxiety_mgt","stress_mgt"},{effect}],
effects=events;
];
effectpos=Position[effects[[1]],effect][[1,1]];
Print[effectpos];
(*constpos=Position[events[[1]],constructs[[i]]][[1,1]];
dateconstevent=events[[;;,{}]];*)
const=StringReplace[effect,"."->" "];
Print[const];
eventconstdata=DeleteCases[Table[
uid=events[[i,uidpos]];
date=ToExpression@StringReplace[ToString[DateObject[events[[i,timepos]]]-DateObject[{2018,1,1}]]," days"->""];
(*c=events[[i,constpos]];*)
c=effects[[i,effectpos]];
event=events[[i,eventpos]];
If[!c===""(*&&!event===""*),
{uid,date,c,event}
]
,{i,2,Length[events]}],Null];


(*Goal: find {user,day} atypical event occurs, then find subset of events where we know: {user, day-1}*)
(*{user, day, value, event\[Equal]1}*)
categories=Which[eventtype=="AtypicalEventCategory",{1,2,3,4,5,6,7},eventtype=="CoarsenedAtypicalEventCategory",{1,2,3},eventtype=="event_mgt",{1}];
Do[
constdayof=Select[eventconstdata,#[[4]]==cat&];
If[Length[constdayof]>10,
nullconstdayof=Select[eventconstdata,#[[4]]==0||#[[4]]==""&];
(*{user, day}*)
dateuidevent=constdayof[[;;,;;2]];
nulldateuidevent=nullconstdayof[[;;,;;2]];
(*{user, day-1}*)
dateuidbeforeevent=Transpose[{dateuidevent[[;;,1]],dateuidevent[[;;,2]]-1}];
nulldateuidbeforeevent=Transpose[{nulldateuidevent[[;;,1]],nulldateuidevent[[;;,2]]-1}];
(*{user, day+1}*)
dateuidafterevent=Transpose[{dateuidevent[[;;,1]],dateuidevent[[;;,2]]+1}];
nulldateuidafterevent=Transpose[{nulldateuidevent[[;;,1]],nulldateuidevent[[;;,2]]+1}];

constdaybefore=Flatten[Table[Select[eventconstdata,#[[;;2]]==dateuidbeforeevent[[i]]&],{i,Length[dateuidbeforeevent]}],1];
nullconstdaybefore=Flatten[Table[Select[eventconstdata,#[[;;2]]==nulldateuidbeforeevent[[i]]&],{i,Length[nulldateuidbeforeevent]}],1];
(*subset of events where we know day before*)
dateuidbeforeevent=constdaybefore[[;;,;;2]];
nulldateuidbeforeevent=nullconstdaybefore[[;;,;;2]];
dateuidevent=Transpose[{dateuidbeforeevent[[;;,1]],dateuidbeforeevent[[;;,2]]+1}];
dateuidafterevent=Transpose[{dateuidbeforeevent[[;;,1]],dateuidbeforeevent[[;;,2]]+2}];

nulldateuidevent=Transpose[{nulldateuidbeforeevent[[;;,1]],nulldateuidbeforeevent[[;;,2]]+1}];
nulldateuidafterevent=Transpose[{nulldateuidbeforeevent[[;;,1]],nulldateuidbeforeevent[[;;,2]]+2}];

constdayof=Flatten[Table[Select[eventconstdata,#[[;;2]]==dateuidevent[[i]]&],{i,Length[dateuidbeforeevent]}],1];
constdayafter=Table[Select[eventconstdata,#[[;;2]]==dateuidafterevent[[i]]&],{i,Length[dateuidbeforeevent]}];
constdiffof=constdayof[[;;,3]]-constdaybefore[[;;,3]];
constdiffafter=DeleteCases[Table[If[Length[constdayafter[[i]]]>0,constdayafter[[i,1,3]]-constdaybefore[[i,3]]],{i,Length[constdaybefore]}],Null];

nullconstdayof=Table[Select[eventconstdata,#[[;;2]]==nulldateuidevent[[i]]&],{i,Length[nulldateuidbeforeevent]}];
nullconstdayafter=Table[Select[eventconstdata,#[[;;2]]==nulldateuidafterevent[[i]]&],{i,Length[nulldateuidbeforeevent]}];
nullconstdiffof=DeleteCases[Table[If[Length[nullconstdayof[[i]]]>0,nullconstdayof[[i,1,3]]-nullconstdaybefore[[i,3]]],{i,Length[nullconstdaybefore]}],Null];
nullconstdiffafter=DeleteCases[Table[If[Length[nullconstdayafter[[i]]]>0,nullconstdayafter[[i,1,3]]-nullconstdaybefore[[i,3]]],{i,Length[nullconstdaybefore]}],Null];


(*
eventboot=Prepend[Table[
dat={constdiffof,constdiffafter}[[i]];
If[Length[dat]>0,
b=BootstrapMean[dat];
m=b[[1]];
err=b[[2]]-m;
{{i-1,m},ErrorBar[err]}
,
{{i-1,Null},ErrorBar[0]}
]
,{i,2}],{{-1,0},ErrorBar[0]}];

nulleventboot=Prepend[Table[
dat={nullconstdiffof,nullconstdiffafter}[[i]];
If[Length[dat]>0,
b=BootstrapMean[dat];
m=b[[1]];
err=b[[2]]-m;
{{i-1,m},ErrorBar[err]}
,
{{i-1,Null},ErrorBar[0]}
]
,{i,2}],{{-1,0},ErrorBar[0]}];

(*
null: find subjects with no event the day someone else has event.
look at day before, day of, day after
*)

plot=
Show[
ListPlot[{eventboot[[;;,1]],nulleventboot[[;;,1]],{{-5,0},{5,0}}},
PlotRange\[Rule]{{-1.2,1.2},All(*{{-0.1,0.500001},{-0.4,1.5},{-1,0.500001},{-0.2,0.70001}}[[i]]*)},
PlotMarkers\[Rule]{Shape["Diamond",Red,Black,15,Opacity[1],Thick],Shape["Circle",CMYKColor[0.5,0.1,0.3,0.1],Black],""},
Joined\[Rule]True,
PlotStyle\[Rule]{{Red},{CMYKColor[0.5,0.1,0.3,0.1],Dashed},{Black}},
Frame\[Rule]True,
Axes\[Rule]False,
FrameLabel\[Rule]Table[Style[ax,FontFamily\[Rule]"Arial",FontSize\[Rule]15,FontColor\[Rule]Black],{ax,{"Days Between Event",const}}],
FrameTicksStyle\[Rule]Directive[15,FontFamily\[Rule]"Arial",Grid\[Rule]True,FontColor\[Rule]Black],
FrameTicks\[Rule]{
{LinTicks[-4,10,Which[constructs[[i]]\[Equal]"pos_af_mgt",0.5,constructs[[i]]\[Equal]"neg_af_mgt",1,constructs[[i]]\[Equal]"anxiety_mgt",0.5,constructs[[i]]\[Equal]"stress_mgt",0.5],2,MajorTickLength\[Rule]{0.02,0},MinorTickLength\[Rule]{0.01,0}],
LinTicks[-4,10,Which[constructs[[i]]\[Equal]"pos_af_mgt",0.5,constructs[[i]]\[Equal]"neg_af_mgt",1,constructs[[i]]\[Equal]"anxiety_mgt",0.5,constructs[[i]]\[Equal]"stress_mgt",0.5],2,MajorTickLength\[Rule]{0.02,0},MinorTickLength\[Rule]{0.01,0},ShowTickLabels\[Rule]False]},
{LinTicks[-1,10,1,1,MajorTickLength\[Rule]{0.02,0},MinorTickLength\[Rule]{0.01,0}],
LinTicks[-1,10,1,1,MajorTickLength\[Rule]{0.02,0},MinorTickLength\[Rule]{0.01,0},ShowTickLabels\[Rule]False]}
},
AspectRatio\[Rule]1,
ImageSize\[Rule]300
],
ErrorListPlot[{eventboot,nulleventboot},PlotStyle\[Rule]{Red,CMYKColor[0.5,0.1,0.3,0.1]},PlotMarkers\[Rule]"",PlotRange\[Rule]{{0.8,2.2},All}],
ListPlot[{eventboot[[;;,1]],nulleventboot[[;;,1]]},
PlotRange\[Rule]{{-1.2,1.2},All},
PlotMarkers\[Rule]{Shape["Diamond",Red,Black,15,Opacity[1],Thick],Shape["Circle",CMYKColor[0.5,0.1,0.3,0.1],Black]}]
];
Print[plot];
Export["~/Dropbox/owl_in_one/Test-Train_MGT/AtypicalEvents/AtypicalEventFig_Dataset="<>dataset<>"_EventType="<>eventtype<>"_Construct="<>const<>".pdf",plot];
*)
eventboot=Prepend[Table[
dat={constdiffof,constdiffafter}[[i]];
If[Length[dat]>0,
b=BootstrapMean[dat];
m=b[[1]];
err=b[[2]]-m;
{i-1,m,err[[1]],err[[2]]}
,
{i-1,Null,0,0}
]
,{i,2}],{-1,0,0,0}];

nulleventboot=Prepend[Table[
dat={nullconstdiffof,nullconstdiffafter}[[i]];
If[Length[dat]>0,
b=BootstrapMean[dat];
m=b[[1]];
err=b[[2]]-m;
{i-1,m,err[[1]],err[[2]]}
,
{i-1,Null,0,0}
]
,{i,2}],{-1,0,0,0}];
If[Length[categories]==1,
Export["~/Dropbox/owl_in_one/Test-Train_MGT/AtypicalEvents/AtypicalEventFig_Dataset="<>dataset<>"_EventType="<>eventtype<>"_Construct="<>effect<>".csv",Prepend[eventboot,{"day","effect","error low","error high"}]];
Export["~/Dropbox/owl_in_one/Test-Train_MGT/AtypicalEvents/AtypicalEventFig_Dataset="<>dataset<>"_EventType="<>eventtype<>"_Construct="<>effect<>"_null.csv",Prepend[nulleventboot,{"day","effect","error low","error high"}]];
,
Export["~/Dropbox/owl_in_one/Test-Train_MGT/AtypicalEvents/AtypicalEventFig_Dataset="<>dataset<>"_EventType="<>eventtype<>"_Construct="<>effect<>"_"<>ToString[cat]<>".csv",Prepend[eventboot,{"day","effect","error low","error high"}]];
Export["~/Dropbox/owl_in_one/Test-Train_MGT/AtypicalEvents/AtypicalEventFig_Dataset="<>dataset<>"_EventType="<>eventtype<>"_Construct="<>effect<>"_"<>ToString[cat]<>"_null.csv",Prepend[nulleventboot,{"day","effect","error low","error high"}]];
];
];
,{cat,categories}];

,{effect,{"pos_af_mgt","neg_af_mgt","anxiety_mgt","stress_mgt"}(*{"Variance.HeartRatePPG.FitBit-Heart-Rate","Variance.HeartRate.OMSignal","Variance.BreathingDepth.OMSignal","Variance.BreathingRate.OMSignal"(*"Mean.HeartRatePPG.FitBit-Heart-Rate","Mean.HeartRate.OMSignal","Mean.BreathingDepth.OMSignal","Mean.BreathingRate.OMSignal"*)}*)}];

,{eventtype,{(*"CoarsenedAtypicalEventCategory","AtypicalEventCategory",*)"event_mgt"}}];

,{dataset,{"Hospital","Aerospace"}}]



