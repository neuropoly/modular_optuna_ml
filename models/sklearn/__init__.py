from models.sklearn.linear import LogisticRegressionManager
from models.sklearn.neighbors import KNeighborsClassifierManager
from models.sklearn.svm import SVCManager

MANAGER_MAP = {
    'LogisticRegression': LogisticRegressionManager,
    'SVC': SVCManager,
    'KNNC':  KNeighborsClassifierManager,
}