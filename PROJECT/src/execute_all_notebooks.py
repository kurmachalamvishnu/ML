import subprocess
import sys
import os

notebooks = [
    "notebook/01_Dataset_Exploration_and_Audit.ipynb",
    "notebook/02_Exploratory_Data_Analysis_EDA.ipynb",
    "notebook/03_Data_Preprocessing_and_Cleaning.ipynb",
    "notebook/04_Feature_Engineering_and_Typologies.ipynb",
    "notebook/08_Classification_Models_and_Baselines.ipynb",
    "notebook/09_Ensemble_Learning_RandomForest_XGBoost.ipynb",
    "notebook/10_Hyperparameter_Tuning_and_Optimization.ipynb",
    "notebook/11_Model_Evaluation_and_Comparison.ipynb",
    "notebook/Final_preprocessing_complete_steps.ipynb",
    "notebook/P3_SLR.ipynb",
    "notebook/P3_MLR.ipynb",
    "notebook/P3_GD.ipynb",
    "notebook/AML_Detection_Complete_Project.ipynb"
]

jupyter_path = os.path.join(".venv", "Scripts", "jupyter.exe")

for nb in notebooks:
    print(f"\n========================================================")
    print(f"Executing: {nb} ...")
    print(f"========================================================")
    cmd = [jupyter_path, "nbconvert", "--to", "notebook", "--execute", "--inplace", nb]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"SUCCESS: {nb} executed with 0 errors!")
    else:
        print(f"ERROR executing {nb}:")
        print(res.stderr)
        sys.exit(1)

print("\nALL 13 NOTEBOOKS EXECUTED CLEANLY AND SUCCESSFULLY!")
