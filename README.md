## Research Summary

This repository contains the complete implementation of a 
physics-informed digital twin and reinforcement learning 
control system for a single-phase induction motor (SPIM).

### Digital Twin Results
- Architecture: Residual LSTM (64 → 32 hidden units)
- R-squared accuracy: 97.78%
- RMSE: 2.12 RPM
- MAE: 1.67 RPM
- Training data: Multi-load Simulink simulation (1,000,001 samples)

### Motor Specifications
- Power: 746W (1HP)
- Voltage: 230V AC
- Frequency: 50Hz
- Poles: 4
- Synchronous speed: 1500 RPM

## Data Availability

Full training dataset available on request.
Preprocessed data and trained model weights stored 
in Google Drive (linked in notebooks).

## Requirements

- Python 3.10+
- PyTorch 2.11
- Stable-Baselines3
- Gymnasium
- scikit-learn
- pandas, numpy, matplotlib

## Citation

If you use this code, please cite our paper:
[Citation will be added upon publication]

## Author

Timothy A. Adeyi, PhD Researcher — NUIST
Supervisor: Professor Adrian D. Cheok
