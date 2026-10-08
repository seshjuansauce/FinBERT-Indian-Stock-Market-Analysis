# Volatility ladder — TCS

News corpus ends 2026-03-30; test period capped there. Placebo = mean of 20 within-year shuffles.


## validation

```
                           rung   n  qlike  rmse_ln  mz_r2        DM vs R0       DM vs R0c       DM vs R0b        DM vs R1      DM vs R2_A      DM vs R2_C     DM vs R2_At
                     R0 GJR raw 369 0.3394   0.9429 0.0688             NaN             NaN             NaN             NaN             NaN             NaN             NaN
             R0c GJR calibrated 369 0.3042   0.8360 0.0773 -3.33 (p=0.001)             NaN             NaN             NaN             NaN             NaN             NaN
               R0b +HAR +market 369 0.3095   0.8571 0.0793             NaN +0.74 (p=0.456)             NaN             NaN             NaN             NaN             NaN
                  R1 +attention 369 0.3005   0.8424 0.0907             NaN             NaN -0.77 (p=0.440)             NaN             NaN             NaN             NaN
             R2_A +sentiment(A) 369 0.3116   0.8580 0.0775             NaN             NaN +0.18 (p=0.859) +1.59 (p=0.111)             NaN             NaN             NaN
             R2_C +sentiment(C) 369 0.3164   0.8626 0.0763             NaN             NaN +0.56 (p=0.578)             NaN +0.68 (p=0.497)             NaN             NaN
           R2_At +sentiment(At) 369 0.3120   0.8581 0.0768             NaN             NaN +0.21 (p=0.835)             NaN             NaN -0.66 (p=0.508)             NaN
           R2_Ct +sentiment(Ct) 369 0.3160   0.8623 0.0774             NaN             NaN +0.52 (p=0.604)             NaN             NaN             NaN +0.60 (p=0.549)
P1 placebo (shuffled attention) 369 0.3104   0.8592 0.0816             NaN             NaN +0.82 (p=0.415)             NaN             NaN             NaN             NaN
```

## test

```
                           rung   n  qlike  rmse_ln  mz_r2        DM vs R0       DM vs R0c       DM vs R0b        DM vs R1      DM vs R2_A      DM vs R2_C     DM vs R2_At
                     R0 GJR raw 548 0.4398   0.9872 0.0863             NaN             NaN             NaN             NaN             NaN             NaN             NaN
             R0c GJR calibrated 548 0.4551   0.9100 0.0780 +0.96 (p=0.335)             NaN             NaN             NaN             NaN             NaN             NaN
               R0b +HAR +market 548 0.4377   0.8993 0.0969             NaN -0.94 (p=0.348)             NaN             NaN             NaN             NaN             NaN
                  R1 +attention 548 0.4273   0.8893 0.1120             NaN             NaN -0.81 (p=0.419)             NaN             NaN             NaN             NaN
             R2_A +sentiment(A) 548 0.4323   0.8869 0.1096             NaN             NaN -0.37 (p=0.710) +1.02 (p=0.310)             NaN             NaN             NaN
             R2_C +sentiment(C) 548 0.4331   0.8901 0.1063             NaN             NaN -0.31 (p=0.756)             NaN +0.41 (p=0.681)             NaN             NaN
           R2_At +sentiment(At) 548 0.4330   0.8873 0.1090             NaN             NaN -0.33 (p=0.741)             NaN             NaN -0.08 (p=0.938)             NaN
           R2_Ct +sentiment(Ct) 548 0.4331   0.8896 0.1068             NaN             NaN -0.31 (p=0.756)             NaN             NaN             NaN +0.05 (p=0.957)
P1 placebo (shuffled attention) 548 0.4378   0.9006 0.0953             NaN             NaN +0.11 (p=0.912)             NaN             NaN             NaN             NaN
```

## test · results days

```
                           rung  n  qlike  rmse_ln  mz_r2        DM vs R0       DM vs R0c       DM vs R0b        DM vs R1      DM vs R2_A      DM vs R2_C     DM vs R2_At
                     R0 GJR raw 20 0.7784   0.9942 0.0025             NaN             NaN             NaN             NaN             NaN             NaN             NaN
             R0c GJR calibrated 20 1.0515   1.0739 0.0001 +2.67 (p=0.008)             NaN             NaN             NaN             NaN             NaN             NaN
               R0b +HAR +market 20 1.1183   1.0812 0.0036             NaN +1.56 (p=0.118)             NaN             NaN             NaN             NaN             NaN
                  R1 +attention 20 0.5165   0.9852 0.0045             NaN             NaN -2.34 (p=0.019)             NaN             NaN             NaN             NaN
             R2_A +sentiment(A) 20 0.5089   0.9811 0.0048             NaN             NaN -2.35 (p=0.019) -0.57 (p=0.572)             NaN             NaN             NaN
             R2_C +sentiment(C) 20 0.5021   0.9824 0.0051             NaN             NaN -2.32 (p=0.020)             NaN -0.36 (p=0.716)             NaN             NaN
           R2_At +sentiment(At) 20 0.5175   0.9855 0.0029             NaN             NaN -2.35 (p=0.019)             NaN             NaN +0.71 (p=0.481)             NaN
           R2_Ct +sentiment(Ct) 20 0.5012   0.9833 0.0048             NaN             NaN -2.32 (p=0.021)             NaN             NaN             NaN -0.72 (p=0.470)
P1 placebo (shuffled attention) 20 1.1252   1.0859 0.0015             NaN             NaN +0.55 (p=0.581)             NaN             NaN             NaN             NaN
```

## test · no-news days

```
                           rung   n  qlike  rmse_ln  mz_r2        DM vs R0       DM vs R0c       DM vs R0b        DM vs R1      DM vs R2_A      DM vs R2_C     DM vs R2_At
                     R0 GJR raw 130 0.5103   1.0391 0.0556             NaN             NaN             NaN             NaN             NaN             NaN             NaN
             R0c GJR calibrated 130 0.5360   0.9604 0.0515 +0.57 (p=0.571)             NaN             NaN             NaN             NaN             NaN             NaN
               R0b +HAR +market 130 0.4644   0.9377 0.0839             NaN -1.03 (p=0.303)             NaN             NaN             NaN             NaN             NaN
                  R1 +attention 130 0.4812   0.9244 0.0836             NaN             NaN +1.15 (p=0.249)             NaN             NaN             NaN             NaN
             R2_A +sentiment(A) 130 0.4862   0.9139 0.0939             NaN             NaN +0.96 (p=0.337) +0.48 (p=0.630)             NaN             NaN             NaN
             R2_C +sentiment(C) 130 0.4900   0.9187 0.0866             NaN             NaN +1.05 (p=0.293)             NaN +1.37 (p=0.170)             NaN             NaN
           R2_At +sentiment(At) 130 0.4860   0.9148 0.0935             NaN             NaN +0.97 (p=0.333)             NaN             NaN -1.37 (p=0.172)             NaN
           R2_Ct +sentiment(Ct) 130 0.4885   0.9179 0.0878             NaN             NaN +1.02 (p=0.306)             NaN             NaN             NaN +1.07 (p=0.283)
P1 placebo (shuffled attention) 130 0.4628   0.9381 0.0833             NaN             NaN -0.91 (p=0.362)             NaN             NaN             NaN             NaN
```

## R1 coefficients, validation period (standardised features, HAC t-stats)

```
             beta (per 1 sd)  t (HAC)      p
const                 -0.605  -15.193  0.000
lrv_d                 -0.012   -0.244  0.807
lrv_w                  0.032    0.594  0.552
lrv_m                 -0.041   -0.804  0.421
lmkt_d                 0.171    3.239  0.001
lmkt_w                -0.060   -0.960  0.337
log_n                  0.009    0.099  0.921
log_primary           -0.029   -0.409  0.682
results_day            0.103    2.103  0.035
log_focus              0.082    1.459  0.144
no_news                0.021    0.380  0.704
```
