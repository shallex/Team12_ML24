# Hard and Rare samples estimation with NN based models (Team-12, ML-24)

## Table of Contents
- [Introduction](#introduction)
- [Approaches](#approaches)
- [Results](#results)
- [Reproduction](#reproduction)

## Introduction

Large datasets often contain instances that do not equally contribute to the learning process. These instances may include mislabeled, difficult-to-learn, or redundant samples. Our objective is to minimize the size of the initial dataset by retaining only highly relevant samples. This approach facilitates faster convergence of the model, reduces storage requirements for useful data, and minimizes computational overhead. Our goal is to proactively filter the unlabeled dataset to reduce costs associated with data labeling, while ensuring the final quality of the model remains uncompromised.

## Approaches

- Supervised learning: SSFT
- Unsupervised learning: AutoEncoder

## Results

All results you can find in appropreate folders and notebooks.

### SSFT

* Network with lower number of parameters (0.3M vs 3M) tends to get decent SSFT metrics even without LR scheduling:

<img src="https://github.com/shallex/Team12_ML24/blob/main/images/ssft_supervised_cnn_300k.png?raw=true" alt="" width="70%"/>

* Visualization of samples taken from different clusters:

<img src="https://github.com/shallex/Team12_ML24/blob/main/images/ssft_supervised_cnn_300k_visualised.png?raw=true" width="70%" />

* In AutoEncoder scenario, we can look at learnt samples as samples with SSIM threshold higher than a fixed threshold (in our case it's 0.15):

<img src="https://github.com/shallex/Team12_ML24/blob/main/images/ae_ssim_before_after_training.png?raw=true" width="70%" />

* However, during second split training, SSIM on the samples from the first split is growing. Meaning that AE reconstruction task tends to generalize rather than overfit to the samples:

 <img src="https://github.com/shallex/Team12_ML24/blob/main/images/ssft_autoencoder.png?raw=true" width="70%" >

* Self and cross correlations are a bad metrics to separate a learnt and not yet learnt samples for the contrastive scenario in BarlowTwins:

<img src="https://github.com/shallex/Team12_ML24/blob/main/images/ssft_barlowtwins_correlations.png?raw=true" width="100%" />

### AutoEncoder

Mix 3000 images, example:

![alt text](https://github.com/shallex/Team12_ML24/blob/main/images/mixed_image.png?raw=true)

Training process:

![alt text](https://github.com/shallex/Team12_ML24/blob/main/images/ae-training.png?raw=true)

#### Metric - sum loss from 1 to 100 epoch

Distribution:

![alt text](https://github.com/shallex/Team12_ML24/blob/main/images/dist_sum_loss.png?raw=true)

Right tail:

![alt text](https://github.com/shallex/Team12_ML24/blob/main/images/0.999_percentile.png?raw=true)

#### Artificial label

Applied label "1" to mixed images

**SRCC of Non-supervised metrics with the artificial label**

- Modest correlation with the artificial target of non-supervised metrics.
- Entropy prevails probably because distorted images, indeed, are less “certain”.
- More complicated metrics yield better results.

SRCC with label:

| Non-supervised metric | SRCC |
| --- | --- |
| H_mean_from_0 | 0.2524 |
| LID_mean_from_10 | 0.2403 |
| H_mean_from_400 | 0.2298 |
| H_last | 0.2285 |
| LID_var_from_10 | 0.2124 |

**SRCC of Loss based metrics with the artificial label**

Loss based metrics exhibit greater correlation with the artificial target.

SRCC with label:

| loss metric | SRCC |
| --- | --- |
| loss_last | 0.3826 |
| loss_mean_from_50 | 0.3801 |
| loss_mean_from_20 | 0.3793 |
| loss_mean_from_0 | 0.3785 |
| loss_diff_last_20 | 0.3067 |

## Reproduction

- To train autoencoder and get bottleneck embeddings with samplewise loss values - run `ssl-ae/ae-reconstruction-L1.ipynb`.
- To compute non-supervised and loss based metrics and plot histograms, images from them - run `ssl-ae/ae-analysis.ipynb`. (The full run takes around hour)
- To get correlations with artificial label and SSFT metrics - run `ssl-ae/ae-correlations.ipynb`.


