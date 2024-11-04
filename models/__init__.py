from models.linear import LogisticRegressionManager
from models.neighbors import KNeighborsClassifierManager
from models.svm import SVCManager

# A map containing all Optuna Managers within this module, for easy reference
MANAGER_MAP = {
    'LogisticRegression': LogisticRegressionManager,
    'SVC': SVCManager,
    'KNNC':  KNeighborsClassifierManager,
}