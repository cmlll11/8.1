# Known Trigger/UAP shallow-feature experiment

This repository implements the forward experiment described in
`浅层诱导特征复杂度正向实验计划.md`. The clean/backdoor classifiers and the UAP/trigger
input mappings are known and frozen. This project does not invert a trigger.

The first milestone provides:

- manifest and environment validation;
- model-agnostic shallow feature interception at named PyTorch modules;
- paired feature-change metrics;
- C0-C4 residual feature mappers;
- the fixed `MDL-FEATURE-v1` bit counter;
- a synthetic smoke test that checks the full software chain without training
  a classifier or searching a trigger.

## Server setup

The target server already has PyTorch 2.4.1+cu121. Install only the remaining
dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Then run the steps in order:

```bash
bash bash/00_check_environment.sh
bash bash/01_synthetic_smoke.sh cuda:0
bash bash/02_check_assets.sh configs/forward_smoke.yaml
```

The real-data commands are intentionally gated on the asset check. Paths in
`configs/forward_smoke.yaml` are explicit placeholders until the server's
existing model and mapping artifacts are connected.

## Scientific boundary

All reported complexities are empirical minima within a frozen layer, mapper
family, quantization scheme and error threshold. Runtime, GPU hours and search
queries are reported separately and are not included in encoded bits.
