# Experiment II: paper-backed feature fitting with MDL

## Scope

Experiment I is unchanged. Experiment II consumes its frozen layer and the
known paired feature tensors. It does not search for an unknown trigger.

This is an exploratory single-classifier, single-trigger demo. The reported
quantity is the shortest code found in a preregistered candidate set, not the
uncomputable global minimum description length.

## Fitting families

1. `mean_shift` is a deterministic average feature-change baseline. It stores
   the mean paired change and performs no gradient search.
2. `feature_re` uses FeatureRE's feature-space mask-pattern equation

   `z_mapped = (1 - mask) * z + mask * pattern`.

   The mask is projected to `[0, 1]` and trained with the paired feature MSE
   plus a frozen mask-size penalty.
3. `fitnets` uses a FitNets-style convolutional hint regressor. A compact
   convolutional network predicts a residual and is trained using squared
   feature matching loss.
4. `residual_adapter` uses the parallel 1x1 residual adapter of Rebuffi et al.
   and the same paired feature MSE.
5. `spatial_gated_fitnets` is the positive-control repair motivated by the
   first demo. It combines FeatureRE-style spatial gating with a FitNets-style
   input-dependent regressor:

   `z_mapped = z + support * (spatial_bias + regressor(z, x_coord, y_coord))`.

   `support` is derived only from the training-pair feature changes. It is
   frozen before validation/test evaluation and its exact binary positions are
   charged by the MDL code. The regressor loss is normalized over this support,
   preventing a fixed-location Trigger from being overwhelmed by unchanged
   spatial positions.

These are adaptations of published feature transformations to the forward
paired-regression question. They are not claims of reproducing the complete
FeatureRE reverse-engineering detector or the complete FitNets distillation
pipeline.

Primary references:

- FeatureRE, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/3f9bf45ea04c98ad7cb857f951f499e2-Abstract-Conference.html
- FitNets: https://arxiv.org/abs/1412.6550
- Residual Adapters, NeurIPS 2017: https://proceedings.neurips.cc/paper/2017/hash/e7b24b112a44fdd9ee93bdf998c6ca0e-Abstract.html

## Stable fitting and evaluation

Training minimizes ordinary feature MSE. NRMSE is not differentiated through;
it is a held-out global relative Frobenius error:

`NRMSE = ||g(z) - z_mapped||_F / max(||z_mapped - z||_F, epsilon)`.

The fixed denominator is aggregated over the whole split, so zero or tiny
per-example changes do not create NaN gradients. Gradients are clipped and
non-finite losses or parameters stop the run immediately.

## MDL code

Every dense fitted mapping is pruned and quantized first. The decoded mapping,
not the original float model, is then evaluated. The two-part code charges:

- protocol, layer and family identifiers;
- architecture fields and all tensor shapes;
- sparse coordinates under an enumerative mask code;
- the train-derived spatial support used by a gated mapper;
- one float32 scale per transmitted quantized tensor;
- the transmitted fp32, int8 or int4 parameter values.

Runtime, optimizer state, epochs and search queries are reported separately
and are not included in the mapping description. A candidate is admissible
only when its decoded test NRMSE is below the frozen threshold and its
feature-reinjection ASR differs from the source mapping by at most 0.05. The
reported `minimum_bits` is the smallest admissible code in this candidate set.

## Demo commands

Run the four conditions on four GPUs, using a distinct log for each process:

```bash
nohup bash bash/07_feature_fitting_demo.sh cuda:0 uap_clean 0 > outputs/gated_fit_uap_clean.log 2>&1 &
nohup bash bash/07_feature_fitting_demo.sh cuda:1 trigger_backdoor 0 > outputs/gated_fit_trigger_backdoor.log 2>&1 &
nohup bash bash/07_feature_fitting_demo.sh cuda:2 uap_backdoor 0 > outputs/gated_fit_uap_backdoor.log 2>&1 &
nohup bash bash/07_feature_fitting_demo.sh cuda:3 trigger_clean 0 > outputs/gated_fit_trigger_clean.log 2>&1 &
```

After all four report `status=completed`:

```bash
bash bash/08_summarize_feature_fitting_demo.sh 0
```
