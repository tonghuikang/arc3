// AUTO-GENERATED from variants.jsonl — do not edit by hand.
// Lets index.html load variant design docs when opened via file:// (browsers block fetch() of local files).
window.__VARIANTS__ = [
  {
    "variant": "tiny",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 13574,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 32, 1 layer. Smallest conv variant; classifies each pixel independently."
  },
  {
    "variant": "baseline",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 27110,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 64, 1 layer. Identical to the runtime model.Model \u2014 the reference architecture."
  },
  {
    "variant": "wide",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 54182,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 128, 1 layer. Wider trunk than baseline."
  },
  {
    "variant": "xwide",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 108326,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 256, 1 layer. Widest single-layer trunk."
  },
  {
    "variant": "deep",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 31270,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 64, 2 layers. Adds depth to baseline at the same width."
  },
  {
    "variant": "deeper",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 35430,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 64, 3 layers. Even deeper at baseline width."
  },
  {
    "variant": "wide_deep",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 70694,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 128, 2 layers. Wide and deep."
  },
  {
    "variant": "wide_deeper",
    "family": "conv-mlp",
    "activation": "silu",
    "params": 87206,
    "spatial": "none (1x1 conv, per-pixel)",
    "summary": "Per-coordinate 1x1-conv MLP, hidden 128, 3 layers. Widest+deepest conv-mlp."
  },
  {
    "variant": "vit",
    "family": "transformer",
    "activation": "gelu",
    "params": 1837926,
    "spatial": "global self-attention over 256 patch tokens (patch 4)",
    "summary": "ViT: 4x4 patch-embed -> 256 tokens, dim 128, 4 layers, 4 heads. Pooled action/color heads; click map decoded+upsampled from tokens. First architecture that lets cells attend to each other."
  },
  {
    "variant": "vit_small",
    "family": "transformer",
    "activation": "gelu",
    "params": 2828726,
    "spatial": "global self-attention over 64 patch tokens (patch 8)",
    "summary": "ViT with coarse 8x8 patches -> 64 tokens, dim 96, 3 layers, 4 heads. ~16x lighter attention than vit; the large patch embed makes param count higher despite fewer tokens."
  },
  {
    "variant": "vit_deep",
    "family": "transformer",
    "activation": "gelu",
    "params": 2631014,
    "spatial": "global self-attention over 256 patch tokens (patch 4)",
    "summary": "Deeper ViT: 4x4 patches, dim 128, 8 layers, 8 heads. Tests whether more attention depth buys better cross-cell reasoning."
  },
  {
    "variant": "conv_vit",
    "family": "transformer",
    "activation": "gelu",
    "params": 1903718,
    "spatial": "2 full-res 3x3 convs (local) then global self-attention (patch 4)",
    "summary": "Hybrid: 2 conv-stem layers mix neighbours locally before patch-embed + transformer mix globally. Local-then-global recipe (early-conv ViT)."
  },
  {
    "variant": "dilated",
    "family": "spatial-cnn",
    "activation": "gelu",
    "params": 667718,
    "spatial": "dilated convs, receptive field grows 1->2->4->8->16 at full 64x64 res",
    "summary": "Stack of 5 dilated 3x3 convs (dilation 1,2,4,8,16), hidden 96. Wide receptive field at full resolution without downsampling or attention."
  },
  {
    "variant": "resnet",
    "family": "spatial-cnn",
    "activation": "gelu",
    "params": 1167878,
    "spatial": "5 residual 3x3 conv blocks at full 64x64 res",
    "summary": "Pre-activation residual conv stack (GroupNorm+GELU), hidden 96, 5 blocks. Residuals let it go deep with local 3x3 spatial context the 1x1 variants lack."
  },
  {
    "variant": "unet",
    "family": "spatial-cnn",
    "activation": "gelu",
    "params": 1214870,
    "spatial": "multi-scale: down 64->32->16, up 16->32->64 with skip connections",
    "summary": "U-Net encoder/decoder, base 48. Coarse bottleneck mixes long-range info; skip connections preserve fine click precision. Non-attention route to global context."
  },
  {
    "variant": "mixer",
    "family": "mlp-mixer",
    "activation": "gelu",
    "params": 2519846,
    "spatial": "global mixing via token-mixing MLP over 256 patch tokens (patch 4)",
    "summary": "MLP-Mixer: alternating token-mixing and channel-mixing MLPs, dim 128, 4 blocks. Global receptive field like a transformer but with MLPs instead of attention."
  },
  {
    "variant": "axial",
    "family": "axial-attn",
    "activation": "gelu",
    "params": 408998,
    "spatial": "axial attention: attend along rows then columns over 32x32 tokens (patch 2)",
    "summary": "Axial transformer: factorises 2-D attention into a row pass + a column pass, dim 64, 4 layers, 4 heads. Reaches every cell in two steps at a fraction of full-attention cost; keeps a finer token grid than vit."
  },
  {
    "variant": "vit_silu",
    "family": "transformer",
    "activation": "silu",
    "params": 1837926,
    "spatial": "global self-attention over 256 patch tokens (patch 4)",
    "summary": "SiLU activation ablation of vit (same patch 4 / dim 128 / 4 layers / 4 heads). New file; the GELU vit is left untouched."
  },
  {
    "variant": "dilated_silu",
    "family": "spatial-cnn",
    "activation": "silu",
    "params": 667718,
    "spatial": "dilated convs, receptive field grows 1->2->4->8->16 at full 64x64 res",
    "summary": "SiLU activation ablation of dilated (same dilation schedule, hidden 96)."
  },
  {
    "variant": "resnet_silu",
    "family": "spatial-cnn",
    "activation": "silu",
    "params": 1167878,
    "spatial": "5 residual 3x3 conv blocks at full 64x64 res",
    "summary": "SiLU activation ablation of resnet (same depth/width, GroupNorm)."
  },
  {
    "variant": "unet_silu",
    "family": "spatial-cnn",
    "activation": "silu",
    "params": 1214870,
    "spatial": "multi-scale: down 64->32->16, up 16->32->64 with skip connections",
    "summary": "SiLU activation ablation of unet (same encoder/decoder + skips, base 48)."
  },
  {
    "variant": "mixer_silu",
    "family": "mlp-mixer",
    "activation": "silu",
    "params": 2519846,
    "spatial": "global mixing via token-mixing MLP over 256 patch tokens (patch 4)",
    "summary": "SiLU activation ablation of mixer (same token/channel MLPs, dim 128, 4 blocks)."
  },
  {
    "variant": "axial_silu",
    "family": "axial-attn",
    "activation": "silu",
    "params": 408998,
    "spatial": "axial attention: attend along rows then columns over 32x32 tokens (patch 2)",
    "summary": "SiLU activation ablation of axial (same row/col attention, dim 64, 4 layers, 4 heads)."
  },
  {
    "variant": "coordconv",
    "family": "conv (coordconv)",
    "activation": "silu",
    "params": 298726,
    "spatial": "3x3 convs + absolute x/y coordinate planes",
    "summary": "CoordConv: appends normalised x/y coordinate channels so the conv knows each cell's absolute position, then 3x3 convs for local context. Hidden 64, 3 layers."
  },
  {
    "variant": "global_context",
    "family": "conv + global context",
    "activation": "silu",
    "params": 77798,
    "spatial": "per-pixel 1x1 conv + broadcast global average-pooled context",
    "summary": "Gives a per-pixel model global awareness cheaply: local 1x1 features concatenated with a global-pooled MLP summary broadcast back to every cell. Hidden 96."
  },
  {
    "variant": "convnext",
    "family": "convnext",
    "activation": "silu",
    "params": 357446,
    "spatial": "depthwise 7x7 per block at full res",
    "summary": "ConvNeXt blocks: depthwise 7x7 conv + channel LayerNorm + pointwise expand/contract MLP with residual. Hidden 96, 4 blocks."
  },
  {
    "variant": "se_resnet",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 1011062,
    "spatial": "5x... residual 3x3 conv blocks + channel attention",
    "summary": "Residual conv blocks with Squeeze-and-Excitation channel attention (global-pool -> bottleneck -> sigmoid rescale). Hidden 96, 4 blocks."
  },
  {
    "variant": "largekernel",
    "family": "conv (large kernel)",
    "activation": "silu",
    "params": 1809062,
    "spatial": "stack of full 7x7 convs, reach ~25x25",
    "summary": "Plain large-kernel CNN: four dense 7x7 convolutions, no dilation/downsampling. Hidden 64, 4 layers."
  },
  {
    "variant": "sepconv",
    "family": "conv (separable)",
    "activation": "silu",
    "params": 85574,
    "spatial": "depthwise 3x3 + pointwise 1x1 per layer",
    "summary": "Depthwise-separable (MobileNet-style) conv stack: cheap local spatial context with far fewer params than dense 3x3. Hidden 96, 5 layers."
  },
  {
    "variant": "fnet",
    "family": "fourier mixing",
    "activation": "silu",
    "params": 1468198,
    "spatial": "global token mixing via parameter-free 2-D FFT (patch 4)",
    "summary": "FNet: replaces self-attention with a 2-D FFT (real part) over 256 patch tokens + SiLU MLP. Global mixing, no learned attention. Dim 128, 4 blocks."
  },
  {
    "variant": "inception",
    "family": "inception",
    "activation": "silu",
    "params": 280454,
    "spatial": "parallel 1x1 / 3x3 / dilated-3x3 / pooled branches per block",
    "summary": "Inception block mixes four receptive-field scales in parallel and concatenates them. Hidden 96, 3 blocks."
  },
  {
    "variant": "conv_axial",
    "family": "conv + axial attn",
    "activation": "silu",
    "params": 585254,
    "spatial": "2 full-res 3x3 convs (local) then axial row/col attention (patch 2)",
    "summary": "Hybrid: conv stem mixes neighbours locally, then axial attention mixes globally over 32x32 tokens. Dim 64, 4 layers, 4 heads."
  },
  {
    "variant": "convmixer",
    "family": "convmixer",
    "activation": "silu",
    "params": 128678,
    "spatial": "depthwise 9x9 (spatial) + pointwise 1x1 (channel), repeated, full res",
    "summary": "ConvMixer: alternating residual depthwise 9x9 and pointwise 1x1 convs with GroupNorm. Strong all-conv design at full resolution. Dim 96, 5 blocks."
  },
  {
    "variant": "densenet",
    "family": "densenet",
    "activation": "silu",
    "params": 235686,
    "spatial": "densely-connected 3x3 convs (feature reuse), full res",
    "summary": "DenseNet: each 3x3 layer sees all previous layers' features concatenated, then 1x1 projection. Growth 32, 5 layers."
  },
  {
    "variant": "pspnet",
    "family": "pyramid pooling",
    "activation": "silu",
    "params": 510854,
    "spatial": "multi-scale: pool to 1/2/4/8 grids, upsample, concat with full-res",
    "summary": "PSPNet pyramid pooling module: injects context at four pooled scales then fuses with full-res features. Non-attention global context. Hidden 96."
  },
  {
    "variant": "aspp",
    "family": "atrous pyramid",
    "activation": "silu",
    "params": 428294,
    "spatial": "parallel atrous convs (dilations 1/6/12/18) + image-level global pool",
    "summary": "ASPP: parallel dilated convs at several rates plus a global-pool branch, fused 1x1. Multi-scale + global context in one layer. Hidden 96."
  },
  {
    "variant": "gmlp",
    "family": "gated MLP",
    "activation": "silu",
    "params": 1601318,
    "spatial": "Spatial Gating Unit (token-mixing linear gate) over 256 patch tokens",
    "summary": "gMLP: project up, then a Spatial Gating Unit gates half the channels by a learned spatial map of the other half. Spatial mixing without attention. Dim 128, 4 blocks."
  },
  {
    "variant": "bottleneck",
    "family": "resnet (bottleneck)",
    "activation": "silu",
    "params": 519334,
    "spatial": "1x1 squeeze -> 3x3 -> 1x1 expand residual blocks, full res",
    "summary": "ResNet bottleneck blocks: squeeze channels with 1x1, 3x3 conv in reduced space, expand back, residual. Hidden 128, mid 32, 4 blocks."
  },
  {
    "variant": "convnext_deep",
    "family": "convnext",
    "activation": "silu",
    "params": 674246,
    "spatial": "depthwise 7x7 per block at full res",
    "summary": "ConvNeXt (frontier leader) made deeper: hidden 96, 8 blocks. Exploits the winning design with more depth."
  },
  {
    "variant": "convnext_wide",
    "family": "convnext",
    "activation": "silu",
    "params": 923398,
    "spatial": "depthwise 7x7 per block at full res",
    "summary": "ConvNeXt made wider: hidden 160, 4 blocks. Exploits the winning design with more width."
  },
  {
    "variant": "se_resnet_deep",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 1686566,
    "spatial": "residual 3x3 + channel attention",
    "summary": "SE-ResNet made deeper: hidden 96, 8 blocks."
  },
  {
    "variant": "convmixer_deep",
    "family": "convmixer",
    "activation": "silu",
    "params": 181382,
    "spatial": "depthwise 9x9 + pointwise 1x1, repeated",
    "summary": "ConvMixer made deeper: dim 96, 8 blocks."
  },
  {
    "variant": "dilated_deep",
    "family": "spatial-cnn",
    "activation": "silu",
    "params": 750758,
    "spatial": "dilated convs, receptive field grows 1->2->4->8->16->32",
    "summary": "SiLU dilated CNN with an extra octave (dilations up to 32) for an even wider receptive field. Hidden 96, 6 layers."
  },
  {
    "variant": "repvgg",
    "family": "repvgg",
    "activation": "silu",
    "params": 741926,
    "spatial": "3x3 + 1x1 + identity branches summed per block, full res",
    "summary": "RepVGG-style multi-branch conv: each block sums 3x3, 1x1 and identity paths before activation. Hidden 96, 5 layers."
  },
  {
    "variant": "convnext_h64",
    "family": "convnext",
    "activation": "silu",
    "params": 245606,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 64, 6 blocks."
  },
  {
    "variant": "convnext_h128",
    "family": "convnext",
    "activation": "silu",
    "params": 607654,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 128, 4 blocks."
  },
  {
    "variant": "convnext_h128_deep",
    "family": "convnext",
    "activation": "silu",
    "params": 1161126,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 128, 8 blocks."
  },
  {
    "variant": "convnext_h192",
    "family": "convnext",
    "activation": "silu",
    "params": 1304678,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 192, 4 blocks."
  },
  {
    "variant": "convnext_b6",
    "family": "convnext",
    "activation": "silu",
    "params": 515846,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 96, 6 blocks."
  },
  {
    "variant": "convnext_b12",
    "family": "convnext",
    "activation": "silu",
    "params": 991046,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 96, 12 blocks."
  },
  {
    "variant": "convnext_wide_deep",
    "family": "convnext",
    "activation": "silu",
    "params": 1779078,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 160, 8 blocks (wide+deep)."
  },
  {
    "variant": "se_resnet_wide",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 2246918,
    "spatial": "(see base family)",
    "summary": "SE-ResNet hidden 128, 6 blocks."
  },
  {
    "variant": "se_resnet_h160",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 2432598,
    "spatial": "(see base family)",
    "summary": "SE-ResNet hidden 160, 4 blocks."
  },
  {
    "variant": "convmixer_wide",
    "family": "convmixer",
    "activation": "silu",
    "params": 305158,
    "spatial": "(see base family)",
    "summary": "ConvMixer dim 160, 6 blocks."
  },
  {
    "variant": "convmixer_h128_deep",
    "family": "convmixer",
    "activation": "silu",
    "params": 274598,
    "spatial": "(see base family)",
    "summary": "ConvMixer dim 128, 8 blocks."
  },
  {
    "variant": "convnext_h64_b4",
    "family": "convnext",
    "activation": "silu",
    "params": 172774,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 64, 4 blocks."
  },
  {
    "variant": "convnext_h64_b8",
    "family": "convnext",
    "activation": "silu",
    "params": 318438,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 64, 8 blocks."
  },
  {
    "variant": "convnext_h128_b6",
    "family": "convnext",
    "activation": "silu",
    "params": 884390,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 128, 6 blocks."
  },
  {
    "variant": "convnext_h160_b6",
    "family": "convnext",
    "activation": "silu",
    "params": 1351238,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 160, 6 blocks."
  },
  {
    "variant": "convnext_h192_b8",
    "family": "convnext",
    "activation": "silu",
    "params": 2528102,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 192, 8 blocks."
  },
  {
    "variant": "convnext_h224_b4",
    "family": "convnext",
    "activation": "silu",
    "params": 1751494,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 224, 4 blocks."
  },
  {
    "variant": "convnext_h256_b4",
    "family": "convnext",
    "activation": "silu",
    "params": 2263846,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 256, 4 blocks."
  },
  {
    "variant": "se_resnet_h64_b6",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 674966,
    "spatial": "(see base family)",
    "summary": "SE-ResNet hidden 64, 6 blocks."
  },
  {
    "variant": "se_resnet_h128_b8",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 2846758,
    "spatial": "(see base family)",
    "summary": "SE-ResNet hidden 128, 8 blocks."
  },
  {
    "variant": "se_resnet_h192_b4",
    "family": "resnet + SE",
    "activation": "silu",
    "params": 3367622,
    "spatial": "(see base family)",
    "summary": "SE-ResNet hidden 192, 4 blocks."
  },
  {
    "variant": "convmixer_h96_d12",
    "family": "convmixer",
    "activation": "silu",
    "params": 251654,
    "spatial": "(see base family)",
    "summary": "ConvMixer dim 96, 12 blocks."
  },
  {
    "variant": "convmixer_h128_d5",
    "family": "convmixer",
    "activation": "silu",
    "params": 192038,
    "spatial": "(see base family)",
    "summary": "ConvMixer dim 128, 5 blocks."
  },
  {
    "variant": "convmixer_h160_d8",
    "family": "convmixer",
    "activation": "silu",
    "params": 384198,
    "spatial": "(see base family)",
    "summary": "ConvMixer dim 160, 8 blocks."
  },
  {
    "variant": "bottleneck_wide",
    "family": "resnet (bottleneck)",
    "activation": "silu",
    "params": 726758,
    "spatial": "(see base family)",
    "summary": "Bottleneck hidden 160, mid 40, 6 blocks."
  },
  {
    "variant": "bottleneck_deep",
    "family": "resnet (bottleneck)",
    "activation": "silu",
    "params": 591270,
    "spatial": "(see base family)",
    "summary": "Bottleneck hidden 128, mid 32, 8 blocks."
  },
  {
    "variant": "dilated_h128",
    "family": "spatial-cnn",
    "activation": "silu",
    "params": 1037734,
    "spatial": "(see base family)",
    "summary": "SiLU dilated CNN, hidden 128, dilations 1..16."
  },
  {
    "variant": "repvgg_deep",
    "family": "repvgg",
    "activation": "silu",
    "params": 1018982,
    "spatial": "(see base family)",
    "summary": "RepVGG hidden 96, 8 layers."
  },
  {
    "variant": "repvgg_wide",
    "family": "repvgg",
    "activation": "silu",
    "params": 1646118,
    "spatial": "(see base family)",
    "summary": "RepVGG hidden 160, 5 layers."
  },
  {
    "variant": "convnext_h160",
    "family": "convnext",
    "activation": "silu",
    "params": 923398,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 160, 4 blocks. Width gap between h128 (best) and h192."
  },
  {
    "variant": "convnext_h320",
    "family": "convnext",
    "activation": "silu",
    "params": 3485158,
    "spatial": "(see base family)",
    "summary": "ConvNeXt hidden 320, 4 blocks. Pushes width past h256 to find the curve knee."
  },
  {
    "variant": "se_convnext",
    "family": "convnext+se",
    "activation": "silu",
    "params": 624614,
    "spatial": "7x7 depthwise + pointwise MLP + SE channel gate",
    "summary": "ConvNeXt blocks with Squeeze-Excitation gating. Combines the two frontier winners. Width 128."
  },
  {
    "variant": "eca_convnext",
    "family": "convnext+eca",
    "activation": "silu",
    "params": 607674,
    "spatial": "7x7 depthwise + pointwise MLP + ECA 1D-conv channel gate",
    "summary": "ConvNeXt blocks with parameter-light Efficient Channel Attention (1D conv over pooled channels). Width 128."
  },
  {
    "variant": "res2net",
    "family": "res2net",
    "activation": "silu",
    "params": 625446,
    "spatial": "hierarchical multi-scale split-conv-cascade",
    "summary": "Res2Net multi-scale residual blocks: channels split into 4 groups processed in a cascade for mixed receptive fields. Width 128."
  },
  {
    "variant": "ghost",
    "family": "ghost",
    "activation": "silu",
    "params": 126374,
    "spatial": "ghost module: 1x1 primary + cheap depthwise 3x3, concatenated",
    "summary": "GhostNet cheap-feature blocks (half channels from a cheap depthwise conv). Smallest conv model. Width 128."
  },
  {
    "variant": "nfnet",
    "family": "nfnet",
    "activation": "silu",
    "params": 1628070,
    "spatial": "normalizer-free scaled residual 3x3 convs",
    "summary": "Normalizer-free residual blocks (alpha/beta signal-propagation, no GroupNorm). Probes if norm is load-bearing. Width 128."
  },
  {
    "variant": "convnext_dw5",
    "family": "convnext",
    "activation": "silu",
    "params": 595366,
    "spatial": "5x5 depthwise + pointwise MLP",
    "summary": "ConvNeXt h128 with a 5x5 depthwise kernel (vs default 7x7). Probes spatial-context size."
  },
  {
    "variant": "convnext_dw9",
    "family": "convnext",
    "activation": "silu",
    "params": 624038,
    "spatial": "9x9 depthwise + pointwise MLP",
    "summary": "ConvNeXt h128 with a 9x9 depthwise kernel."
  },
  {
    "variant": "convnext_dw11",
    "family": "convnext",
    "activation": "silu",
    "params": 644518,
    "spatial": "11x11 depthwise + pointwise MLP",
    "summary": "ConvNeXt h128 with an 11x11 depthwise kernel."
  },
  {
    "variant": "convnext_e6",
    "family": "convnext",
    "activation": "silu",
    "params": 870822,
    "spatial": "7x7 depthwise + 6x pointwise MLP",
    "summary": "ConvNeXt h128 with 6x inverted-bottleneck expansion (vs 4x)."
  },
  {
    "variant": "convnext_e8",
    "family": "convnext",
    "activation": "silu",
    "params": 1133990,
    "spatial": "7x7 depthwise + 8x pointwise MLP",
    "summary": "ConvNeXt h128 with 8x inverted-bottleneck expansion."
  },
  {
    "variant": "convnext_h112",
    "family": "convnext",
    "activation": "silu",
    "params": 474358,
    "spatial": "(see base family)",
    "summary": "ConvNeXt fine-width probe at hidden 112 (just below the h128 sweet spot)."
  },
  {
    "variant": "convnext_h144",
    "family": "convnext",
    "activation": "silu",
    "params": 757334,
    "spatial": "(see base family)",
    "summary": "ConvNeXt fine-width probe at hidden 144 (just above the h128 sweet spot)."
  },
  {
    "variant": "poolformer",
    "family": "metaformer",
    "activation": "silu",
    "params": 583078,
    "spatial": "avgpool-minus-identity token mixer + channel MLP",
    "summary": "PoolFormer: parameter-free pooling token mixer in the MetaFormer norm/mix/MLP block. Width 128."
  },
  {
    "variant": "mobilenetv2",
    "family": "mobilenet",
    "activation": "silu",
    "params": 995750,
    "spatial": "inverted residual: 1x1 expand, 3x3 depthwise, 1x1 project",
    "summary": "MobileNetV2 inverted-residual (MBConv) blocks, expansion 4, residual on narrow ends. Width 128."
  },
  {
    "variant": "hornet",
    "family": "hornet",
    "activation": "silu",
    "params": 806822,
    "spatial": "gnconv recursive gated depthwise conv (order 2) + channel MLP",
    "summary": "HorNet high-order gated convolutions; gating kept inside a pre-normed residual with 1/order scale damp for stability. Width 128."
  },
  {
    "variant": "head_clickconv",
    "family": "head",
    "activation": "silu",
    "params": 884486,
    "spatial": "convnext_h128 body + 3x(3x3 conv) click decoder",
    "summary": "HEAD experiment: replaces the 1x1 click head with a 3-layer 3x3-conv decoder so the click map has a real receptive field. Targets the dominant coord loss."
  },
  {
    "variant": "head_coordconv",
    "family": "head",
    "activation": "silu",
    "params": 886214,
    "spatial": "convnext_h128 body + CoordConv 3x3 click decoder",
    "summary": "HEAD experiment: click decoder fed normalised (x,y) coordinate channels (CoordConv) so it can reason about absolute position. Targets coord."
  },
  {
    "variant": "head_separable",
    "family": "head",
    "activation": "silu",
    "params": 609718,
    "spatial": "convnext_h128 body + factorized x/y (outer-sum) click head",
    "summary": "HEAD experiment: click map = row-logit(y) + col-logit(x), a separable P(y)*P(x) localization prior. Targets coord."
  },
  {
    "variant": "head_unet",
    "family": "head",
    "activation": "silu",
    "params": 1308582,
    "spatial": "convnext_h128 body + U-Net click decoder (64->32->16->up, skips)",
    "summary": "HEAD experiment: multi-scale U-Net click decoder for coarse-to-fine localization. Targets coord."
  },
  {
    "variant": "head_attnpool",
    "family": "head",
    "activation": "silu",
    "params": 610492,
    "spatial": "convnext_h128 body + softmax attention pooling (action/color)",
    "summary": "HEAD experiment: replaces amax pooling with per-channel spatial attention pooling for action/color. Click stays 1x1 (isolated A/B). Targets action+color."
  },
  {
    "variant": "head_combo",
    "family": "head",
    "activation": "silu",
    "params": 892156,
    "spatial": "convnext_h128 body + attn pool + CoordConv separable+2D click",
    "summary": "HEAD experiment: kitchen-sink head stacking attention pooling and a CoordConv separable+2D click decoder. Candidate to clear the convnext_h128 plateau."
  }
];
