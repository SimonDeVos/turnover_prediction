# Predicting Employee Turnover: Scoping and Benchmarking the State-of-the-Art</br><sub><sub>Simon De Vos, Chris Rickermann, Jente Van Belle, Wouter Verbeke [[2024]](https://doi.org/10.1007/s12599-024-00898-z)</sub></sub>  

This paper addresses the need for predictive analytics in workforce management by scoping and benchmarking the state-of-the-art research on employee turnover prediction. Through an extensive benchmarking experiment involving 14 classification methods and 9 datasets, we highlight the challenges posed by inconsistent methodologies and experimental setups in existing studies. Our findings provide a unified perspective to advance both academic research and practical applications in human resource management. The code and public datasets are made available on GitHub to encourage further research and collaboration.

## Repository Structure
This repository is organized as follows:
```bash
|- data/
    |- ds.csv              # Dataset for experiments
    |- ibm.csv             # IBM HR dataset
    |- kaggle1.csv         # Kaggle dataset 1
    |- kaggle3.csv         # Kaggle dataset 3
    |- kaggle4.csv         # Kaggle dataset 4
    |- kaggle5.csv         # Kaggle dataset 5
|- experiments/
    |- experiment.py       # Script for conducting experiments
    |- main.py             # Main entry point for running experiments
|- performance_metrics/
    |- performance_metrics.py  # Module for evaluating model performance
```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Instructions:
- In ['main.py'](https://github.com/SimonDeVos/turnover_prediction/blob/master/experiments/main.py):
  - Set the project directory to your custom folder. E.g., `DIR = r'C:\Users\...\...\...'`
  - Specify experiment configuration in `settings = {'folds': 2, 'repeats': 5, ...}`
  - Specify dataset used in `datasets = {'real1': False, 'ibm': True, ...}`. The public datasets can be found in the [data folder](https://github.com/SimonDeVos/turnover_prediction/tree/7f6389ff91b39770fd232205a7f02fbab1758361/data
). The datasets _Real1_, _Real2_, and _Real3_ are not publicly available.
  - Specify the classifications methods in `methodologies = {'ab': True,'ann': True,'bnb': True, ... }`
  - Hyperparameter grids can be adapted in `hyperparameters = {'ab': {'n_estimators': [50, 100, 200], ...}, 'ann': {...} ...}`. It is recommended to put some hyperparameter specifications in comment, as running the current specified grid takes a long time.
- Run 'main.py' to reproduce our results. Results will be written to a text file in `DIR = r'C:\Users\...\...\...'`

## Citing
Please cite our paper and/or code as follows:

```tex

@article{de2024predicting,
  title={Predicting Employee Turnover: Scoping and Benchmarking the State-of-the-Art},
  author={De Vos, Simon and Bockel-Rickermann, Christopher and Van Belle, Jente and Verbeke, Wouter},
  journal={Business \& Information Systems Engineering},
  pages={1--20},
  year={2024},
  publisher={Springer}
}

```
