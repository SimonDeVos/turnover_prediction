This is the code associated with the paper '_Employee turnover analytics: Scoping and benchmarking the state-of-the-art_'.

**Instructions:**
- Set the project directory in 'main.py' to your custom folder. `DIR = r'C:\Users\...\...\...'`
- Specify experiment configuration in `settings = {'folds': 2, 'repeats': 5, ...}`
- Specify the dataset used in `datasets = {'real1': False, 'ibm': True, ...}`. The public datasets can be found in the [data](https://github.com/SimonDeVos/turnover_prediction/tree/7f6389ff91b39770fd232205a7f02fbab1758361/data
) folder. The datasets _Real1_, _Real2_, and _Real3_ are not publicly available.
- Specify the classifications methods in `methodologies = {'ab': True,'ann': True,'bnb': True, ... }`
- Hyperparameter grids can be adapted in `hyperparameters = {'ab': {'n_estimators': [50, 100, 200], ...}, 'ann': {...} ...}`. Running the current specified grid takes a long time.
- Run 'main.py' to reproduce our results.

**About the authors:** 

_anonymized repo_

**Cite as**

_anonymized repo_
