v19_dark_hbasin_sep

Base:
  v12_4_ropad_lite_gate_boostonly

Input:
  CAVITY_FORCE_V19_DARK_HBASIN_3CH=1
  --in_chans 3

Channels:
  ch0 = normalized CXR
  ch1 = dark residual
  ch2 = black h-dome / h-basin

Recommended env:
  CAVITY_FORCE_V19_DARK_HBASIN_3CH=1
  CAVITY_V19_HBASIN_H=0.08
  CAVITY_RESIDUAL_SIGMA=5.0
  CAVITY_V19_DECOUPLE_CLS_SEG=1
