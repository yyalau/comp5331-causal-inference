# COMP 5331 Group Project

## General Information

**Title:** Causal Inference via Style Transfer for Out-of-Distribution Generalisation

**Project Type:** Implementation

**Group Number:** 13

**Members:**

| Student ID | Student Name | Contribution |
| ---------- | ------------ | ------------ |
| 20504741 | LEUNG, Tin Long | Training, evaluation, and logging framework |
| 20583797 | TANG, Zheng | Main framework |
| 20596201 | LAU, Ying Yee Ava | NST model |
| 20607969 | WONG, Chi Ho | Classifier model |
| 20942785 | BAHARI, Maral | Main framework |
| 20943026 | BA SOWID, Badr Saleh Abdullah | Data config / preprocessing |

## Installation

### Requirements

- Linux operating system
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

### Baseline Method (ERM)

1. Train a standard ERM (Empirical Risk Minimization) classifier:
```sh
# By default, checkpoints are stored in `./experiments/erm`
python run_erm.py [OPTIONS]
```

#### Example

### Proposed Method (FAGT)

1. *(Optional)* Train a NST (Neural Style Transfer) model:
```sh
# By default, checkpoints are stored in `./experiments/nst`
python run_nst.py [OPTIONS]
```
#### Example
```sh
python "run_nst.py" fit \
        --data "config/data/nst/pacs.yaml" \
        --model "config/model/nst/ada_in.yaml" \
        --trainer.devices [0] \
        --trainer.accelerator gpu \
        --trainer.max_epochs 70 \
        --data.batch_size 32 \
        --data.dataset_config.train_domains ["art_painting","cartoon","photo"] \
        --data.dataset_config.val_domains ["art_painting","cartoon","photo"] \
        --data.dataset_config.test_domains ["sketch"]
```
or directly run the shell script
```sh
bash run_nst.sh
```

2. Train a FA (Front-door Adjustment) classifier:
```sh
# You can use the NST model from Step 1 or download a pretrained one
# By default, checkpoints are stored in `./experiments/fa`
python run_fa.py [OPTIONS]
```

#### Example

## File Structure

### Main Files

- `config/`: Contains our YAML configuration files.
    - `data/`: Contains the configuration for loading each dataset.
        - e.g.: `classification/erm/digits.yaml` is the configuration file for loading the Digits-DG dataset for ERM task.
    - `model/`: Contains the configuration for training each model.
        - e.g.: `classification/erm/digits/conv.yaml` is the configuration file for training the small CNN on the Digits-DG dataset for ERM task.
- `src/`: Contains our source code.
    - `datamodules/`: TBD
    - `dataops/`: TBD
    - `models/`: 
        - `classification`:
            - `erm`:
                - `base.py`: Defines the input data type of the ERM models and its base class. 
                - `cnn.py`: Defines the architecture of small CNN and its submodules.
                - `resnet18.py`: Defines the architecture of ResNet-18 and its submodules.
                - `vit_wrapper.py`: Defines the additional properties. such as checkpoint loading, of ViT classifier.
                - `vit.py`: Defines the architecture of ViT classifier and its submodules.
            - `fa`: 
                - `base.py`: Defines the input data type of the FA frameworks and its base class
                - `faft.py`: Defines the forward passing of FAFT.
                - `fagt.py`: Defines the forward passing of FAGT.
                - `fast.py`: Defines the forward passing of FAST.
                - `fourier.py`: Defines the forward passing of FST.
            - `base.py`: Defines a base classification model 
        - `nst`:
            - `ada_in.py`: Defines the architecture of AdaIN NST model and its submodules
            - `base.py`: Defines the input data type of NST models, base method for NST models, and base method for pretrained modules.
        - `base.py`: Defines a partial interface for `torch.nn.Module`
    - `tasks/`:
        - `base.py`: Defines the base task.
        - `classification/`: Defines each classification task.
            - `base.py`: Defines the base classification task.
            - `erm.py`: Defines the ERM task (for the ERM method).
            - `fa.py`: Defines the FA task (for the FAST, FAFT and FAGT methods).
        - `nst/`: Defines each style transfer task.
            - `ada_in.py`: Defines the NST task for training AdaIN model.

- `pyproject.toml`, `setup.cfg`: Configures the code linters used in this project.
- `requirements.txt`: Lists out the package dependencies for this project.
- `run_erm.py`, `run_erm.sh`: Command-line script for training and evaluating a model for ERM task.
- `run_nst.py`, `run_nst.sh`: Command-line script for training and evaluating a model for NST task.
- `run_fa.py`, `run_fa.sh`: Command-line script for training and evaluating a model for FA task.

### Supplementary Files

- `data/`: Contains each dataset. *(Created when downloading the datasets)*
- `experiments/`: Contains the TensorBoard logs. *(Created when running the experiments)*
- `weights/`: Contains the model weights. *(Created when downloading the pretrained models)*

## References

Our code is based on the GitHub repository for the following papers:
- [Causal Inference via Style Transfer for Out-of-distribution Generalisation](https://github.com/nktoan/Causal-Inference-via-Style-Transfer-for-OOD-Generalisation)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
- [Deep Residual Learning for Image Recognition (ResNet)](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://github.com/google-research/vision_transformer)
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (AdaIN)](https://github.com/MAlberts99/PyTorch-AdaIN-StyleTransfer)
