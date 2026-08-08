Exit code: 0
Wall time: 6.9 seconds
Output:
# BackdoorBench attack provenance

The multitype training code does not use the previous hand-written
Input-Aware or SSBA substitutes.

## Input-Aware

The generator, mask generator, threshold function, mask-density objective and
diversity objective are adapted from the public BackdoorBench implementation:

- <https://github.com/SCLBD/BackdoorBench/blob/main/attack/inputaware.py>
- original release: <https://github.com/VinAIResearch/input-aware-backdoor-attack-release>

The project adapts only the data loader, CIFAR-10 tensor split and checkpoint
layout. The classifier, generator and mask are jointly optimized as in the
source method. The resulting checkpoint stores the classifier plus generator
and mask state dictionaries.

## SSBA

BackdoorBench's `attack/ssba.py` does **not** invent a pixel residual. It reads
sample-specific poisoned arrays produced by the official ISSBA/StegaStamp
preprocessing stage:

- <https://github.com/SCLBD/BackdoorBench/blob/main/attack/ssba.py>
- <https://github.com/tancik/StegaStamp>
- <https://github.com/Kooscii/ISSBA>

The project therefore requires these files and fails explicitly if they are
absent:

```text
artifacts/backdoorbench/ssba/cifar10_ssba_train_b1.npy
artifacts/backdoorbench/ssba/cifar10_ssba_test_b1.npy
```

No synthetic SSBA fallback is permitted. A missing official array makes the
SSBA type `unqualified`/`missing_official_asset`, rather than producing a
scientifically uninterpretable replacement trigger.

