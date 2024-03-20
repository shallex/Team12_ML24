# Hard and Rare samples estimation with NN based models (Team-12, ML-24)

## Table of Contents
- [Introduction](#introduction)
- [Approaches](#approaches)
- [Results](#results)

## Introduction

Large datasets often contain instances that do not equally contribute to the learning process. These instances may include mislabeled, difficult-to-learn, or redundant samples. Our objective is to minimize the size of the initial dataset by retaining only highly relevant samples. This approach facilitates faster convergence of the model, reduces storage requirements for useful data, and minimizes computational overhead. Our goal is to proactively filter the unlabeled dataset to reduce costs associated with data labeling, while ensuring the final quality of the model remains uncompromised.

## Approaches

- Supervised learning: SSFT
- Unsupervised learning: AutoEncoder

## Results

All results you can find in appropreate folders and notebooks.

### SSFT

Ramazan part

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