# P1 Fourier high-frequency pressure test

The fixed public dictionary has 33 columns: a constant plus sine/cosine
pairs at 16 selected integer multiples of pi. Selection is independent of
the sample's oracle omega. The baseline frequencies 1π–16π are modified by
replacing 16π with 24π or 40π, respectively. Both runs use the same seed-42
clean curve, M=31 and a direct square solve.

| Maximum frequency | Condition number | Dense RMSE | Status |
| --- | ---: | ---: | --- |
| 24π | 2.38245e10 | 6.18e-12 | numerical failure: unstable condition |
| 40π | 6.09393e10 | 3.54e-10 | numerical failure: unstable condition |

The tiny reconstruction errors do not override the specified numerical
failure threshold. The archived summaries include the complete selected
frequency lists, predictions, matrix diagnostics and singular values.
