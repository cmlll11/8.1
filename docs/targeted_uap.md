# Targeted UAP asset

The formal forward experiment uses a **single image-agnostic additive
perturbation**. It is trained against the frozen clean classifier and is known
before shallow-feature analysis; it is not a trigger-inversion result.

## Method

`projected_targeted_uap` is a CIFAR-10 adaptation of the targeted universal
objective in Poursaeed et al., *Generative Adversarial Perturbations* (CVPR
2018). Instead of retaining a generator whose input is fixed noise, this
implementation directly optimizes its decoded perturbation. Each optimizer
step projects that perturbation back into a frozen L-infinity ball.

The epsilon ladder is fixed in the config before training. Candidates are
trained and selected only on the validation set, from the smallest epsilon
upward. The test ASR is evaluated once for the selected candidate. Multiple
deterministic restarts reduce sensitivity to initialization.

This is an adaptation of the published method, not copied upstream code. The
upstream implementation targets older PyTorch and ImageNet assets, so using it
unchanged would not match this repository's CIFAR-10 model or artifact format.

## References

- Paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.html
- Upstream implementation: https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations
