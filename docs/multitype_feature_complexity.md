# Multitype feature-complexity experiment

This stage runs the formal, known-mapping experiment for six CIFAR-10 trigger
families: `badnets`, `blended`, `wanet`, `inputaware`, `low_frequency`, and
`ssba`. It keeps the existing UAP implementation and the `MDL-FEATURE-v1`
codec. There is no smoke or pilot mode in this workflow.

## Server commands

```bash
bash bash/09_train_multitype_assets.sh configs/multitype_feature_formal.yaml cuda:0
bash bash/10_multitype_observation.sh configs/multitype_feature_formal.yaml cuda:0
```

Fit one trigger/seed at a time on separate GPUs:

```bash
bash bash/11_multitype_fitting.sh configs/multitype_feature_formal.yaml cuda:0 0 badnets
bash bash/11_multitype_fitting.sh configs/multitype_feature_formal.yaml cuda:1 0 blended
bash bash/11_multitype_fitting.sh configs/multitype_feature_formal.yaml cuda:2 0 wanet
bash bash/11_multitype_fitting.sh configs/multitype_feature_formal.yaml cuda:3 0 inputaware
```

Repeat for `low_frequency` and `ssba` and seeds `1`, `2`, then summarize:

```bash
bash bash/12_multitype_report.sh configs/multitype_feature_formal.yaml
```

The fixed selection rule is validation `NRMSE <= 0.10`, followed by minimum
encoded bits. `K_act` additionally requires fitted ASR at least `0.90` and an
ASR gap no larger than `0.05`. The report stores a separate first-
indistinguishable layer for every trigger type; two consecutive layers are
required before stopping is confirmed.

