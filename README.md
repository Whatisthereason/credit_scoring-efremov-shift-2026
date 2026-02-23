# Credit Risk Classification --- Machine Learning

Binary classification pipeline implemented with CatBoost.

## Pipeline Structure

config → preprocessing → feature_engineering → validation → tuning → final_training → submission

- **Config**: Centralized parameters, fixed seeds, dataset fingerprinting  
- **Preprocessing**: Numerical/categorical separation, high-missing filtering, deterministic feature ordering  
- **Feature Engineering**: Morphological normalization (`pymorphy3`), role/domain/seniority extraction  
- **Validation**: 3 fixed stratified hold-outs, identical subset + split per run  
- **Tuning**: Coarse grid search on fixed subset → validation across 3 seeds  
- **Final Training**: Early stopping, best iteration selection, full-train refit  
- **Submission**: Deterministic prediction, output validation, artifact packaging  

## Implemented

- Feature-set comparison (All / numerical-only / filtered)  
- High-cardinality categorical transformation  
- Mean ± std ROC-AUC reporting  
- Cached feature matrices (Parquet)  
- Automated generation of `submisson.csv` and `requirements.txt`  

## Result

ROC-AUC ≈ 0.75 (200k+ samples, stratified hold-out). There were better results (commented cells), but the laptop had powered off during training due to system protection.

## Data

The original train and test datasets are ~1 GB in total and exceed GitHub’s 25 MB file size limit.  
Only datasets with 50 entries (as a samples) are uploaded.

## Stack

Python · CatBoost · scikit-learn · pandas · NumPy · pymorphy3
