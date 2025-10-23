# DINO-ECHO: A Foundation Model for Echocardiography Video Interpretation and Report Generation

**DINO-ECHO** is a multimodal foundation model designed to understand and interpret **echocardiography videos** through self-supervised and vision-language pretraining.  
Our approach integrates **DINOv2**, **CLIP**, and **LLaVA** to achieve robust visual representation, fine-grained vision-language alignment, and automatic report generation.

We demonstrate that **ECHO** significantly outperforms existing baselines including **EchoCLIP**, **EchoPrime**, **BiomedGPT**, and **BiomedCLIP** across a range of echocardiographic understanding and reporting benchmarks.

---

## 🌟 Highlights

- **End-to-end echocardiography foundation model** combining DINOv2, CLIP, and LLaVA for video understanding and report generation.  
- **Self-supervised and multimodal learning** — captures spatial-temporal cardiac features and aligns them with clinical text using contrastive learning.  
- **Automated structured reporting** — generates diagnostic summaries and surgical indications directly from echocardiographic videos.  
- **Strong generalization** — achieves state-of-the-art results on linear probing, zero-shot classification, and report generation tasks.  
- **Clinically meaningful impact** — enhances efficiency, accuracy, and scalability of echocardiographic interpretation in practice.

---

## 🏗️ Repository Structure

```bash
ECHO/
├── CLIP/ # CLIP fine-tuning module
│ ├── scripts/ # Bash scripts for CLIP fine-tuning
│ └── ... # Modified CLIP training code
│
├── LLaVA/ # LLaVA-based report generation module
│ ├── scripts/ # Bash scripts for LLaVA training
│ └── ... # Submodule + modified pretrain/fine-tune code
│
├── scripts/ # Bash scripts for running training/evaluation
│
├── custom_util/ # Utility functions and custom tools
│
├── dataset/ # Dataset loading and preprocessing code
│
├── models/ # Model definitions and architecture modules
│
├── other/ # Miscellaneous tools and helper scripts
│
├── requirements.txt # Python dependencies
├── run.py # Main training script
├── run_engine.py # Training/evaluation engine
└── bootstrap_metrics.py # Metric computation and bootstrapping
```
