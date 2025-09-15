# GBCI: Generative Breast Cancer Imaging

## Overview
GBCI is a pipeline for synthetic mammogram generation using **Latent Diffusion Models** (LDM) and risk prediction modules, with semantic conditioning on clinical and anatomical attributes. The project supports simulation, augmentation, and in-silico patient profiling, focusing on attributes such as density, category, BIRADS, and lesion coordinates.

## Main Features
- **Image Generation**: LDM model with encoder/decoder and transformer cross-attention, conditioned on attributes and masks.
- **Automated Evaluation**: SSIM and PSNR metrics for synthetic image quality.
- **Risk Prediction**: ViT-based module for BIRADS classification from images and attributes.
- **Dataset Management**: Loader for VinDr-Mammo (DICOM + CSV), robust attribute and mask parsing.
- **Model Comparison**: Script to evaluate and compare different checkpoints.
- **Visualization**: Utilities to display real, masked, and reconstructed images.

## Repository Structure
```
GBCI/
├── notebooks/           # Jupyter notebooks for demos and experiments
├── src/                 # Model implementation, training, evaluation
│   ├── model.py         # LDM, transformer, embedding models
│   ├── train.py         # Training for generation and risk prediction
│   ├── predict_risk.py  # BIRADS risk prediction
│   ├── compare_model.py # Model comparison (SSIM/PSNR)
│   ├── generate.py      # Conditional image generation
│   ├── data_loader.py   # DICOM + CSV loader, attribute parsing
│   ├── metrics.py       # SSIM, PSNR, FID (placeholder)
│   ├── utils.py         # Visualization and saving utilities
│   ├── config_tuning.py # Global configuration
│   ├── requirements.txt # Dependencies
├── data/                # Preprocessing scripts (no raw data)
├── docs/                # Documentation
├── README.md            # This file
└── LICENSE              # License
```

## Getting Started
1. **Clone the repository**  
   ```bash
   git clone https://github.com/nickystark/GBCI.git
   cd GBCI
   ```
2. **Install dependencies**  
   ```bash
   pip install -r src/requirements.txt
   ```
3. **Prepare the dataset**  
   - Place the VinDr-Mammo zip file in your chosen folder.
   - Update the path in `config_tuning.py` if needed.

4. **Launch notebooks**  
   ```bash
   cd notebooks
   jupyter notebook
   ```

## Usage Examples
- **Generation Training**:  
  See `src/train.py` and the main notebook.
- **Model Evaluation**:  
  Use `src/compare_model.py` to compare checkpoints.
- **Conditional Image Generation**:  
  Use `src/generate.py` with specific attributes.
- **Risk Prediction (BIRADS)**:  
  Use `src/predict_risk.py` for classification.

## Models & Architecture
- **Latent Diffusion Transformer**:  
  CNN encoder, decoder, ViT for masks, attribute embedding, transformer denoising with cross-attention.
- **Risk Predictor**:  
  ViT + attributes for BIRADS classification.

## Evaluation Metrics
- **SSIM**: Structural similarity between real and synthetic images.
- **PSNR**: Peak signal-to-noise ratio.
- **FID**: Placeholder for future extensions.

## Dataset
- **VinDr-Mammo**: DICOM mammograms + CSV annotations, automatic attribute and mask parsing.

## Contributing
- Open **issues** and **pull requests** for improvements.
- Follow the branching strategy for stable development.

## Resources & References
- [ISPAMM Lab Code Repository](https://github.com/orgs/ispamm/repositories)
- [High-Resolution Image Synthesis](https://www.notion.so/High-Resolution-Image-Synthesis-with-Latent-Diffusion-Models-568cdba7f3c2415a989673ceef9ca20f?pvs=21)
- [Segmentation-Guided Diffusion Models](https://www.notion.so/Anatomically-Controllable-Medical-Image-Generation-with-Segmentation-Guided-Diffusion-Models-2067d0e6cb8980fdafc6f454642ae25b?pvs=21)
- [Generalizable Tumor Synthesis](https://www.notion.so/Towards-Generalizable-Tumor-Synthesis-2067d0e6cb8980cfa090dad1e627e177?pvs=21)

## Notes
For questions, suggestions, or extension requests, check the internal documentation or open an issue on GitHub.
