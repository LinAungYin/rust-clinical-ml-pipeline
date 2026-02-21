# rust-clinical-ml-pipeline
High-performance, memory-safe machine learning pipeline for clinical risk prediction in Rust. Features a custom Logistic Regression engine built from first principles. Demonstrates systems-level engineering capabilities and execution speed suitable for large-scale biomedical and genomic data processing.

# High-Performance Clinical ML Pipeline (Rust) 🦀

**Project:** Computational Deep Phenotyping for Clinical Risk Prediction  
**Context:** PhD Application Technical Showcase (NUS Yong Loo Lin School of Medicine)  
**Author:** Lin Aung Yin

## 📌 Overview
This repository features a custom-built, bare-metal machine learning pipeline written in Rust. While Python is the standard for prototyping ML models, processing massive clinical cohorts or high-dimensional multi-omics datasets requires systems-level performance. 

This project demonstrates the ability to write highly efficient, memory-safe algorithms capable of handling large-scale biomedical data. It implements a Logistic Regression classifier, Batch Gradient Descent, and K-Fold Cross-Validation entirely from scratch.

## ⚙️ Engineering Focus
* **Memory Safety & Speed:** Leveraging Rust's ownership model to ensure zero memory leaks during iterative gradient descent training.
* **Algorithmic Transparency:** Hard-coding Z-score standardization and log-loss optimization to prove a deep understanding of ML mechanics, rather than calling external crates.
* **Robust Evaluation:** Ensuring the model's predictive stability across unseen patient data folds.

## 🚀 How to Run
This project requires Rust and Cargo to be installed on your system.
1. Clone the repository.
2. Navigate to the project directory.
3. Build and run the executable:
   ```bash
   cargo run
