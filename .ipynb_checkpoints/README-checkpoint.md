# Equivariant Imaging Biomarkers for Robust Unsupervised Segmentation of Histopathology
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![arXiv:2505.05689](https://img.shields.io/badge/arXiv-2505.05689-B31B1B.svg)](https://arxiv.org/abs/2505.05689)

 *Fuyao Chen, Yuexi Du, Tal Zeevi, Nicha C. Dvornek, John A. Onofrey*
 
 *Yale University*

![teaser](assets/teaser.gif)

## News
- **July 2025:** *Paper accepted by MIDL 2025!*

## Abstract
> Histopathology evaluation of tissue specimens through microscopic examination is essential for accurate disease diagnosis and prognosis. However, traditional manual analysis by specially trained pathologists is time-consuming, labor-intensive, cost-inefficient, and prone to inter-rater variability, potentially affecting diagnostic consistency and accuracy. As digital pathology images continue to proliferate, there is a pressing need for automated analysis to address these challenges. Recent advancements in artificial intelligence-based tools such as machine learning (ML) models, have significantly enhanced the precision and efficiency of analyzing histopathological slides. However, despite their impressive performance, ML models are invariant only to translation, lacking invariance to rotation and reflection. This limitation restricts their ability to generalize effectively, particularly in histopathology, where images intrinsically lack meaningful orientation. In this study, we develop robust, equivariant histopathological biomarkers through a novel symmetric convolutional kernel via unsupervised segmentation. The approach is validated using prostate tissue micro-array (TMA) images from 50 patients in the Gleason 2019 Challenge public dataset. The biomarkers extracted through this approach demonstrate enhanced robustness and generalizability against rotation compared to models using standard convolution kernels, holding promise for enhancing the accuracy, consistency, and robustness of ML models in digital pathology. Ultimately, this work aims to improve diagnostic and prognostic capabilities of histopathology beyond prostate cancer through equivariant imaging.

## Installation 
Please refer to [SRE-Conv page](https://github.com/XYPB/SRE-Conv.git) for detailed instructions on installation and model pretraining. 

## Datasets

The experiments used Gleason 2019 Challenge tissue micro-array (TMA) images: [Download Here](https://gleason2019.grand-challenge.org/).
For image preprocessing:

```bash
cd ./preprocess
python data_preprocess.py \
  --input_folder ./input_images \
  --output_folder ./masked_images \
  --output_csv ./metadata.csv
```

## Usage
### Intra-Subject Analysis

```bash
python src/intra_subject.py \
  --csv_path data/traindata.csv \
  --n_clusters 3 \
  --n_samples 2000 \
  --seed 42 \
  --weight_path pretrained_models/model.pth \
  --output_path results/
```

### Inter-Subject Analysis

```bash
python src/inter_subject.py \
  --csv_path data/train_metadata.csv \
  --test_data_path data/test_metadata.csv \
  --n_clusters 3 \
  --n_samples 2000 \
  --seed 42 \
  --weight_path pretrained_models/sre_model.pth \
  --output_path results/
```

**Arguments:**

- `--csv_path`: Path to a CSV file with metadata and image paths  
- `--n_clusters`: Number of Kmeans clusters for unsupervised segmentation  
- `--n_samples`: Number of features to subsample per image  
- `--seed`: Random seed for reproducibility  
- `--weight_path`: Path to pretrained model weights  
- `--output_path`: Directory where segmentation outputs will be saved
- `csv_path`: Path to a CSV file with metadata and image paths
- `test_data_path`: (Optional) For inter-subject analysis only. Path to test data CSV file. 

## Reference

```
@ARTICLE{Chen2025-kc,
  title         = "Equivariant imaging biomarkers for robust unsupervised
                   segmentation of histopathology",
  author        = "Chen, Fuyao and Du, Yuexi and Zeevi, Tal and Dvornek, Nicha
                   C and Onofrey, John A",
  month         =  may,
  year          =  2025,
  copyright     = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
  archivePrefix = "arXiv",
  primaryClass  = "eess.IV",
  eprint        = "2505.05689"
}

```