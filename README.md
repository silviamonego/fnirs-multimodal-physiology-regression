# Multimodal Physiological Noise Regression in fNIRS

MATLAB computational pipeline for systemic physiological noise modeling, feature extraction, and regularized regression in functional near-infrared spectroscopy (fNIRS).

Developed as part of the Master's Thesis titled *"Multimodal Physiological Noise Regression in fNIRS: Methodological Validation Through a Novel Finger-Tapping and Breath-Holding Dataset"* in Bioengineering at the University of Padova (Department of Information Engineering - DEI).

---

## Overview

Systemic autonomic fluctuations (arterial cardiac pulsatility, respiration, blood pressure Mayer waves, hypercapnic cerebral vasodilation, and sympathetic electrodermal bursts) introduce non-neural confounds into functional near-infrared spectroscopy (fNIRS) recordings. When cognitive or motor tasks coincide with systemic physiological changes, these confounds distort hemodynamic response functions (HRFs) and can generate false-positive cortical activations.

This repository provides an end-to-end processing pipeline that:
1. Extracts and synchronizes multimodal autonomic biosignals (Lead II ECG, earlobe PPG, thoracic respiration, palmar EDA, and pulse oximetry) with multi-channel fNIRS recordings.
2. Evaluates low-frequency 1/f noise properties and demonstrates the effect of block-by-block linear detrending on spurious correlations.
3. Quantifies state-dependent coupling between superficial short-separation (SC) channels and peripheral autonomic regressors.
4. Implements a nested regularized ridge regression framework (models M0–M6) with multi-lag expansions (0 to 12 s), trained on hypercapnic Breath-Holding (BH) blocks via Leave-One-Block-Out Cross-Validation (LOBO-CV) and transferred to an independent dual-task condition (COMB: simultaneous finger tapping and breath holding).
5. Benchmarks decontaminated activations against an independent SC-corrected Homer GLM reference.
6. Computes group-level non-parametric inferential statistics (Wilcoxon signed-rank tests with step-down Holm-Bonferroni correction) and evaluates the one-standard-error (1-SE) parsimony model selection rule across subjects.

---

## Repository Structure

* `physio_general.m`  
  Preprocesses raw auxiliary physiological recordings (`.wings` and `.snirf`). Implements Savitzky-Golay filtering on thoracic respiration, apnea compliance screening for BH and COMB tasks, downsampling and convex optimization for EDA (cvxEDA tonic and phasic components), peak detection for PPG and ECG (Lead II) with instantaneous heart rate derivation, and SpO2 cleaning. Generates `<subject>_ProcessedPhysio.csv`, `<subject>_ValidTrials.mat`, and Satori design matrix files (`.sdm`).

* `noise_simulation.m`  
  Evaluates power spectral densities (PSD) of fNIRS short channels and physiological signals against a synthetic 1/f^beta noise model (beta = 1.7). Runs a Monte Carlo simulation (1000 iterations) to quantify spurious cross-correlations in raw versus block-detrended segments.

* `task_correlation.m`  
  Computes dynamic 8x8 correlation matrices between superficial scalp hemodynamics (SC PCA for HbO and HbR) and peripheral autonomic regressors across experimental states (Baseline, Fingertapping, Breath Holding, Combined). Generates state-specific correlation heatmaps and saves `<subject>_TaskCorrelation.mat`.

* `integrated_spatial_agreement_validation.m`  
  Core subject-level modeling script. Fits nested ridge regression models (M0–M6) on hypercapnic BH blocks using nested LOBO-CV to select regularization parameters (lambda) and identify the optimal model via the 1-SE rule. Applies frozen regression weights to COMB epochs, extracts trial-by-trial amplitudes, FDR-corrected activation maps, spatial Pearson correlations with the Homer GLM FT reference, and C3 motor ROI HRF RMSE.

* `group_analysis.m`  
  Aggregates subject-level model summaries (`*_IntegratedSpatial_ModelSummary.csv`). Computes median and interquartile ranges across models, executes planned pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction, summarizes 1-SE model selection frequencies, and outputs summary tables and publication figures.

* `group_baseline_analysis.m`  
  Computes group-level trial retention statistics, respiratory pre-screening compliance, and task-induced physiological changes across subjects.

---

## Computational Workflow

1. **Physiological Preprocessing**: Run `physio_general.m` for each subject folder. This validates apnea trial compliance and exports aligned physiological regressors.
2. **Task Correlation Analysis**: Run `task_correlation.m` to generate the 8-regressor matrix and task-dependent correlation structures.
3. **Noise Simulations (Optional)**: Run `noise_simulation.m` to inspect spectral slopes and run Monte Carlo detrending tests.
4. **Subject-Level Validation**: Run `integrated_spatial_agreement_validation.m` for each participant to perform nested LOBO-CV ridge regression and benchmark against the Homer GLM reference.
5. **Group-Level Statistics**: Run `group_analysis.m` and `group_baseline_analysis.m` on the directory containing all subject outputs to obtain group tables and figures.

---

## Requirements and Dependencies

* MATLAB (tested on R2022b and later)
* Signal Processing Toolbox
* Statistics and Machine Learning Toolbox
* [cvxEDA](https://github.com/lciti/cvxEDA) (required for electrodermal activity decomposition; CVX optimization package must be installed and configured in MATLAB)
* [Homer2](https://homer-fnirs.org/) (specifically `hmrDeconvHRF_DriftSS.m` for simultaneous GLM deconvolution with short-channel regressors)

---

## Data Availability

The raw and preprocessed experimental datasets are available from the author upon reasonable request for non-commercial scientific and academic research.

---

## Acknowledgments

Data was acquired using equipment kindly provided by NIRx Medical Technologies.

---

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License. See the `LICENSE` file for details.
