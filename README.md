# GBCI: Generative Breast Cancer Imaging

## Overview
Generative models are transforming medical imaging by enabling **simulation, augmentation, and in-silico patient profiling**. This project focuses on developing a **latent diffusion model (LDM) and/or a VAE-based pipeline** trained on mammograms (e.g., VinDr-Mammo or INbreast) to generate **synthetic cancer-positive and -negative mammograms**. The goal is to explore **semantic conditioning**, ensuring clinically relevant attributes are captured, such as **mass density, calcifications, and tissue asymmetries**.

## Project Goals
- Implement a **diffusion-based generative model** for mammogram synthesis.
- Explore **latent-space representations** that capture cancer-relevant features.
- **Condition** image generation using metadata such as BI-RADS scores, lesion type, or patient profiles.
- Compare **VAE-based approaches** and evaluate generated samples for diagnostic reliability.

## Team Contributions
- **Feature Conditioning:** Cross-attention or FiLM layers to incorporate metadata and radiology reports.
- **Reconstruction Evaluation:** Testing synthetic images through classification/segmentation models for diagnostic robustness.
- **Domain Translation:** Generating lesion-containing images from healthy scans to simulate **disease progression**.

## Honors Extensions
For advanced implementation:
- **Longitudinal Modeling:** Simulating lesion evolution using **flow-matching** or **temporal diffusion models**.
- **Counterfactual Synthesis:** Generating alternative scenarios based on real patient data.
- **Multimodal Generation:** Producing coherent mammogram-biopsy image pairs.

## Repository Structure
📂 generative-cancer-imaging ├── notebooks/           # Jupyter/Colab notebooks ├── src/                 # Model implementation ├── data/                # Dataset processing scripts (not raw data) ├── docs/                # Project documentation ├── requirements.txt     # Dependencies ├── README.md            # Project overview └── LICENSE              # License information

## Getting Started
1. Clone the repository:
git clone https://github.com/nickystark/GBCI.git cd GBCI
2. Install dependencies:
pip install -r requirements.txt
3. Open the notebooks:
cd notebooks/ jupyter notebook
4. Follow project development using [GitHub Projects](https://github.com/nickystark/GBCI.git/projects).

## References & Resources
- **ISPAMM Lab Code Repository**: [GitHub](https://github.com/orgs/ispamm/repositories)
- **Medical Generative Models Research**:
- [High-Resolution Image Synthesis](https://www.notion.so/High-Resolution-Image-Synthesis-with-Latent-Diffusion-Models-568cdba7f3c2415a989673ceef9ca20f?pvs=21)
- [Segmentation-Guided Diffusion Models](https://www.notion.so/Anatomically-Controllable-Medical-Image-Generation-with-Segmentation-Guided-Diffusion-Models-2067d0e6cb8980fdafc6f454642ae25b?pvs=21)
- [Generalizable Tumor Synthesis](https://www.notion.so/Towards-Generalizable-Tumor-Synthesis-2067d0e6cb8980cfa090dad1e627e177?pvs=21)

##Dataset
VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography

## Contributing
- Submit **issues and pull requests** for improvements.
- Follow the **branching strategy** for stable development.
