# COMP 5331 Group Project

## General Information

**Title:** Causal Inference via Style Transfer for Out-of-Distribution Generalisation

**Project Type:** Implementation

**Group Number:** 13

**Members:**

| Student ID | Student Name | Contribution |
| ---------- | ------------ | ------------ |
| 20504741 | LEUNG, Tin Long | Data visualization / traineval |
| 20583797 | TANG, Zheng | Main framework |
| 20596201 | LAU, Ying Yee Ava | NST model |
| 20607969 | WONG, Chi Ho | Classifier model |
| 20942785 | BAHARI, Maral | Main framework |
| 20943026 | BA SOWID, Badr Saleh Abdullah | Data config / preprocessing |

## Installation

### Requirements

- [Python](https://www.python.org/) 3.10
- [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 11.8+

### Setup

1. Activate a virtual Python environment (e.g. [conda](https://docs.conda.io/en/latest/)).
2. Install the dependencies.
```sh
python -m pip install torch==2.1.* torchvision==0.16.* torchaudio==2.1.* --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Usage

### Experiments

To train the model with the **baseline** method (standard Empirical Risk Minimization):

```sh
# By default, checkpoints are stored in `./checkpoints/erm`
python run_erm.py [OPTIONS]
```


To train the model with the **proposed** method (Front-door Adjustment via Neural Style Transfer):

```sh
# By default, checkpoints are stored in `./checkpoints/fast`
python run_fast.py [OPTIONS]
```

## References

Our code is based on the GitHub repository for the following papers:
- [Causal Inference via Style Transfer for Out-of-distribution Generalisation](https://github.com/nktoan/Causal-Inference-via-Style-Transfer-for-OOD-Generalisation)
- [AdaIN](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer)
- [Image Transfer Network](https://github.com/zuotengxu/Image-Neural-Style-Transfer-With-Preserving-the-Salient-Regions)
- [Vision Transformer](https://github.com/google-research/vision_transformer)
