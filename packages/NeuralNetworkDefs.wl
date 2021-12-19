(* ::Package:: *)

BeginPackage["NeuralNetworkDefs`"]


UNET::usage="build arbitrary UNET"
UNETdsw::usage="UNETdsw[n_] using depth separable convolutions"
UNETdswUnidirectional::usage="..."
UNETunidirectional::usage="UNETdsw[n_] using depth separable convolutions"


Begin["`Private`"]


dir=$InputFileName


leayReLU[alpha_]:=ElementwiseLayer[Ramp[#]-alpha*Ramp[-#]&]
basicBlock[channels_,kernelSize_,opts___]:=NetChain[{ConvolutionLayer[channels,kernelSize,"PaddingSize"->(kernelSize-1)/2(*,"Biases"\[Rule]None*),opts],BatchNormalizationLayer[],leayReLU[0.1]}];
convBlock[channels_,kernelSize_,opts___]:=NetChain[{BatchNormalizationLayer[],leayReLU[0.1],ConvolutionLayer[channels,kernelSize,"PaddingSize"->(kernelSize-1)/2(*,"Biases"\[Rule]None*),opts]}];

(*Depth separated convolutional layer*)
DSConvolutionLayer[nin_,nout_,kernelSz_,stride_:1]:=NetChain@{ConvolutionLayer[nin,kernelSz,"Stride"->stride,"ChannelGroups"->nin,"PaddingSize"->1],ConvolutionLayer[nout,1]}
DSDeconvolutionLayer[n_,k_,stride_:1]:=NetChain@{DeconvolutionLayer[n,k,"Stride"->stride,"GroupNumber"->1],ConvolutionLayer[n,1]}
(*Depth separated convolutional block*)
dsconvBlock[channelsin_,channelsout_,kernelSize_,stride_:1]:=NetChain[{DSConvolutionLayer[channelsin,channelsout,kernelSize,stride],BatchNormalizationLayer[],leayReLU[0.1]}];



(* ::Section:: *)
(*UNETs*)


UNET:=Module[{n=64},
NetGraph[<|
"conv1"->{basicBlock[n,3,"PaddingSize"->1],basicBlock[n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool1"->{PoolingLayer[2,2]},
"conv2"->{basicBlock[2n,3,"PaddingSize"->1],basicBlock[2n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool2"->{PoolingLayer[2,2]},
"conv3"->{basicBlock[4n,3,"PaddingSize"->1],basicBlock[4n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool3"->{PoolingLayer[2,2]},
"conv4"->{basicBlock[8n,3,"PaddingSize"->1],basicBlock[8n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool4"->{PoolingLayer[2,2]},
"conv5"->{basicBlock[16n,3,"PaddingSize"->1],basicBlock[16n,3,"PaddingSize"->1],DropoutLayer[.2]},
"up1"->DeconvolutionLayer[8n,2,"Stride"->2],
"cat1"->CatenateLayer[],
"uconv1"->{basicBlock[8n,3,"PaddingSize"->1],basicBlock[8n,3,"PaddingSize"->1]},
"up2"->DeconvolutionLayer[4n,2,"Stride"->2],
"cat2"->CatenateLayer[],
"uconv2"->{basicBlock[4n,3,"PaddingSize"->1],basicBlock[4n,3,"PaddingSize"->1]},
"up3"->DeconvolutionLayer[2n,2,"Stride"->2],
"cat3"->CatenateLayer[],
"uconv3"->{basicBlock[2n,3,"PaddingSize"->1],basicBlock[2n,3,"PaddingSize"->1]},
"up4"->DeconvolutionLayer[n,2,"Stride"->2],
"cat4"->CatenateLayer[],
"uconv4"->{basicBlock[n,3,"PaddingSize"->1],basicBlock[n,3,"PaddingSize"->1]},
"class"->{ConvolutionLayer[2,1],TransposeLayer[{1<->2,2<->3}],SoftmaxLayer[-1]}
|>,{"conv1"->"maxpool1"->"conv2"->"maxpool2"->"conv3"->"maxpool3"->"conv4"->"maxpool4"->"conv5"->"up1",
{"up1","conv4"}->"cat1"->"uconv1"->"up2",
{"up2","conv3"}->"cat2"->"uconv2"->"up3",
{"up3","conv2"}->"cat3"->"uconv3"->"up4",
{"up4","conv1"}->"cat4"->"uconv4"->"class"
}]]

UNETunidirectional[sz_]:=Module[{n=64},
NetGraph[<|
"conv1"->{basicBlock[n,3,"PaddingSize"->1],basicBlock[n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool1"->{PoolingLayer[2,2]},
"conv2"->{basicBlock[2n,3,"PaddingSize"->1],basicBlock[2n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool2"->{PoolingLayer[2,2]},
"conv3"->{basicBlock[4n,3,"PaddingSize"->1],basicBlock[4n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool3"->{PoolingLayer[2,2]},
"conv4"->{basicBlock[8n,3,"PaddingSize"->1],basicBlock[8n,3,"PaddingSize"->1],DropoutLayer[.1]},
"maxpool4"->{PoolingLayer[2,2]},
"conv5"->{basicBlock[16n,3,"PaddingSize"->1],basicBlock[16n,3,"PaddingSize"->1],DropoutLayer[.2]},
"up1"->DeconvolutionLayer[8n,2,"Stride"->2],
"cat1"->CatenateLayer[],
"uconv1"->{basicBlock[8n,3,"PaddingSize"->1],basicBlock[8n,3,"PaddingSize"->1]},
"up2"->DeconvolutionLayer[4n,2,"Stride"->2],
"cat2"->CatenateLayer[],
"uconv2"->{basicBlock[4n,3,"PaddingSize"->1],basicBlock[4n,3,"PaddingSize"->1]},
"up3"->DeconvolutionLayer[2n,2,"Stride"->2],
"cat3"->CatenateLayer[],
"uconv3"->{basicBlock[2n,3,"PaddingSize"->1],basicBlock[2n,3,"PaddingSize"->1]},
"up4"->DeconvolutionLayer[n,2,"Stride"->2],
"cat4"->CatenateLayer[],
"uconv4"->{basicBlock[n,3,"PaddingSize"->1],basicBlock[n,3,"PaddingSize"->1]},
"ant"->{ConvolutionLayer[2,1],TransposeLayer[{1<->3,1<->2}],SoftmaxLayer[]},
"ret"->{ConvolutionLayer[2,1],TransposeLayer[{1<->3,1<->2}],SoftmaxLayer[]}
|>,{"conv1"->"maxpool1"->"conv2"->"maxpool2"->"conv3"->"maxpool3"->"conv4"->"maxpool4"->"conv5"->"up1",
{"up1","conv4"}->"cat1"->"uconv1"->"up2",
{"up2","conv3"}->"cat2"->"uconv2"->"up3",
{"up3","conv2"}->"cat3"->"uconv3"->"up4",
{"up4","conv1"}->"cat4"->"uconv4"->{"ant","ret"},
"ant"->NetPort["ant"],"ret"->NetPort["ret"]
}]]


UNETdswUnidirectional[n_]:=Module[{},
NetGraph[<|
"conv1"->{basicBlock[n,3,"PaddingSize"->1],dsconvBlock[n,n,3],DropoutLayer[.1]},
"maxpool1"->{PoolingLayer[2,2]},
"conv2"->{dsconvBlock[n,2n,3],dsconvBlock[2n,2n,3],DropoutLayer[.1]},
"maxpool2"->{PoolingLayer[2,2]},
"conv3"->{dsconvBlock[2n,4n,3],dsconvBlock[4n,4n,3],DropoutLayer[.1]},
"maxpool3"->{PoolingLayer[2,2]},
"conv4"->{dsconvBlock[4n,8n,3],dsconvBlock[8n,8n,3],DropoutLayer[.1]},
"maxpool4"->{PoolingLayer[2,2]},
"conv5"->{dsconvBlock[8n,16n,3],dsconvBlock[16n,16n,3],DropoutLayer[.2]},
"up1"->DeconvolutionLayer[8n,2,"Stride"->2],
"cat1"->CatenateLayer[],
"uconv1"->{dsconvBlock[8n,8n,3],dsconvBlock[8n,8n,3]},
"up2"->DeconvolutionLayer[4n,2,"Stride"->2],
"cat2"->CatenateLayer[],
"uconv2"->{dsconvBlock[4n,4n,3],dsconvBlock[4n,4n,3]},
"up3"->DeconvolutionLayer[2n,2,"Stride"->2],
"cat3"->CatenateLayer[],
"uconv3"->{dsconvBlock[2n,2n,3],dsconvBlock[2n,2n,3]},
"up4"->DeconvolutionLayer[n,2,"Stride"->2],
"cat4"->CatenateLayer[],
"uconv4"->{dsconvBlock[n,n,3],dsconvBlock[n,n,3]},
"ant"->{ConvolutionLayer[2,1],TransposeLayer[{1<->3,1<->2}],SoftmaxLayer[]},
"ret"->{ConvolutionLayer[2,1],TransposeLayer[{1<->3,1<->2}],SoftmaxLayer[]}
|>,{"conv1"->"maxpool1"->"conv2"->"maxpool2"->"conv3"->"maxpool3"->"conv4"->"maxpool4"->"conv5"->"up1",
{"up1","conv4"}->"cat1"->"uconv1"->"up2",
{"up2","conv3"}->"cat2"->"uconv2"->"up3",
{"up3","conv2"}->"cat3"->"uconv3"->"up4",
{"up4","conv1"}->"cat4"->"uconv4"->{"ant","ret"},
"ant"->NetPort["Antero"],"ret"->NetPort["Retro"]
}]]



UNETdsw[n_]:=Module[{},
NetGraph[<|
"conv1"->{basicBlock[n,3,"PaddingSize"->1],dsconvBlock[n,n,3],DropoutLayer[.1]},
"maxpool1"->{PoolingLayer[2,2]},
"conv2"->{dsconvBlock[n,2n,3],dsconvBlock[2n,2n,3],DropoutLayer[.1]},
"maxpool2"->{PoolingLayer[2,2]},
"conv3"->{dsconvBlock[2n,4n,3],dsconvBlock[4n,4n,3],DropoutLayer[.1]},
"maxpool3"->{PoolingLayer[2,2]},
"conv4"->{dsconvBlock[4n,8n,3],dsconvBlock[8n,8n,3],DropoutLayer[.1]},
"maxpool4"->{PoolingLayer[2,2]},
"conv5"->{dsconvBlock[8n,16n,3],dsconvBlock[16n,16n,3],DropoutLayer[.2]},
"up1"->DeconvolutionLayer[8n,2,"Stride"->2],
"cat1"->CatenateLayer[],
"uconv1"->{dsconvBlock[8n,8n,3],dsconvBlock[8n,8n,3]},
"up2"->DeconvolutionLayer[4n,2,"Stride"->2],
"cat2"->CatenateLayer[],
"uconv2"->{dsconvBlock[4n,4n,3],dsconvBlock[4n,4n,3]},
"up3"->DeconvolutionLayer[2n,2,"Stride"->2],
"cat3"->CatenateLayer[],
"uconv3"->{dsconvBlock[2n,2n,3],dsconvBlock[2n,2n,3]},
"up4"->DeconvolutionLayer[n,2,"Stride"->2],
"cat4"->CatenateLayer[],
"uconv4"->{dsconvBlock[n,n,3],dsconvBlock[n,n,3]},
"class"->{ConvolutionLayer[2,1],TransposeLayer[{1<->2,2<->3}],SoftmaxLayer[-1]}
|>,{"conv1"->"maxpool1"->"conv2"->"maxpool2"->"conv3"->"maxpool3"->"conv4"->"maxpool4"->"conv5"->"up1",
{"up1","conv4"}->"cat1"->"uconv1"->"up2",
{"up2","conv3"}->"cat2"->"uconv2"->"up3",
{"up3","conv2"}->"cat3"->"uconv3"->"up4",
{"up4","conv1"}->"cat4"->"uconv4"->"class"
}]]



End[]
EndPackage[]

