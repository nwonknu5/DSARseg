# Domain-Invariant Continual Segmentation against Acquisition Shift

Implementation code for paper: Domain-Invariant Continual Segmentation against Acquisition Shift.

## ACKNOWLEDGEMENT
Our method and comparison methods are mainly implemented on the lifelong nnUNet framework. Please refer to https://github.com/MECLabTUDA/Lifelong-nnUNet to understand the more detailed lifelong-nnUNet network architecture, training class definitions, comparison methods, and training & testing commands.

## Dataset
* Prostate Dataset: Multi-centre, multi-vendor and multi-disease cardiac segmentation: the M&MS challenge. 
* Fundus Dataset: https://github.com/liuquande/FedDG-ELCFS

## Code files
* ``network_architecture/generic_UNet.py`` implementation implementation of a generic nnUNet model embedded with a shape codebook (SCB).
* ``network_architecture/modules.py`` includes implementation of VQVAE module.
* ``network_training/nnUNetTrainerDSAR.py`` includes the specific implementation of Domain-Specificity Aware Regularization (DSAR) and Continual Domain Alignment (CDA) strategy.
* ``network_training/train_shape_codebook.py`` includes the training program for shape codebook (SCB).
* ``loss_functions/deep_supervision.py`` includes the loss function of our DSAR module, CDA strategy and SCB module.

## Tips: code corresponding to the methodology section of the paper.
* Shape Codebook (SCB)
    * Training of Shape Codebook - see ``network_training/train_shape_codebook.py`` （line 24-50; line 84-148）
    * Integration of Shape Codebook - see ``network_architecture/generic_UNet.py`` (line 358-434)
* Domain-Specificity Aware Regularization (DSAR)
    * Simulation of Acquisition Shift - see ``network_training/nnUNetTrainerDSAR.py`` (line 268-291)
    * Calculation of Domain Sensitivity - see ``network_training/nnUNetTrainerDSAR.py`` (line 293-302)
    * Penalty on Domain-specific Parameters - see ``loss_functions/deep_supervision.py`` (line 50-64)
* Continual Domain Alignment strategy (CDA)
    * Estimation of Distribution - see ``network_training/nnUNetTrainerDSAR.py`` (line 331-412)
    * Pseudo Samples Generation - see ``loss_functions/deep_supervision.py`` (line 146-180)


## Requirements

    pytest
    einops==0.3.2
    batchgenerators==0.23
    nnunet==1.7.1
    hiddenlayer==0.2
    timm == 0.5.4
    matplotlib
    IPython
    scikit-learn

## Usage
Please first preprocess the dataset according to the method required by nnUNet. After processing the prostate datasets, their task IDs range from 71 to 76 (The ids can be freely specified).  

### Setup

```
python setup.py install
```

### train Shape Codebook

```
python continual_train_vqvae.py --data-folder ./save/dataset/ --dataset prostate --output-folder  CDBv2_h128k512 --hidden-size 128 --k 512 --batch-size 16 --num-epochs 100 --device cuda:4
```

### train nnUNet
```
nnUNet_train_dsar 2d -t 71 72 73 74 75 76 -f 0 -dsar_lambda 10 -cda_beta 0.01 -num_epoch 100 -d 0 -save_interval 100 -s seg_outputs --store_csv
```

### validate
```
nnUNet_evaluate 2d nnUNetTrainerDSAR -trained_on 71 72 73 74 75 76 -f 0 -use_model 71 -evaluate_on 71 72 73 74 75 76 -d 1 --store_csv 
nnUNet_evaluate 2d nnUNetTrainerDSAR -trained_on 71 72 73 74 75 76 -f 0 -use_model 71 72 -evaluate_on 71 72 73 74 75 76 -d 1 --store_csv 
nnUNet_evaluate 2d nnUNetTrainerDSAR -trained_on 71 72 73 74 75 76 -f 0 -use_model 71 72 73 -evaluate_on 71 72 73 74 75 76 -d 1 --store_csv 
nnUNet_evaluate 2d nnUNetTrainerDSAR -trained_on 71 72 73 74 75 76 -f 0 -use_model 71 72 73 74 -evaluate_on 71 72 73 74 75 76 -d 1 --store_csv 
nnUNet_evaluate 2d nnUNetTrainerDSAR -trained_on 71 72 73 74 75 76 -f 0 -use_model 71 72 73 74 75 -evaluate_on 71 72 73 74 75 76 -d 1 --store_csv 
nnUNet_evaluate 2d nnUNetTrainerDSAR -trained_on 71 72 73 74 75 76 -f 0 -use_model 71 72 73 74 75 76 -evaluate_on 71 72 73 74 75 76 -d 1 --store_csv 
```