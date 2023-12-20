This is the code associated with the paper '_Employee turnover analytics: Scoping and benchmarking the state-of-the-art_'.

**Instructions:**
- In ['main.py'](https://github.com/SimonDeVos/turnover_prediction/blob/master/experiments/main.py):
  - Set the project directory to your custom folder. E.g., `DIR = r'C:\Users\...\...\...'`
  - Specify experiment configuration in `settings = {'folds': 2, 'repeats': 5, ...}`
  - Specify dataset used in `datasets = {'real1': False, 'ibm': True, ...}`. The public datasets can be found in the [data folder](https://github.com/SimonDeVos/turnover_prediction/tree/7f6389ff91b39770fd232205a7f02fbab1758361/data
). The datasets _Real1_, _Real2_, and _Real3_ are not publicly available.
  - Specify the classifications methods in `methodologies = {'ab': True,'ann': True,'bnb': True, ... }`
  - Hyperparameter grids can be adapted in `hyperparameters = {'ab': {'n_estimators': [50, 100, 200], ...}, 'ann': {...} ...}`. It is recommended to put some hyperparameter specifications in comment, as running the current specified grid takes a long time.
- Run 'main.py' to reproduce our results. Results will be written to a text file in `DIR = r'C:\Users\...\...\...'`

**About the authors:** 

_anonymized repo_

**Cite as**

_anonymized repo_
