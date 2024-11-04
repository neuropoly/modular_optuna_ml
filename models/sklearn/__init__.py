from models.sklearn.linear import LogisticRegressionManager
from models.sklearn.neighbors import KNeighborsClassifierManager
from models.sklearn.svm import SVCManager
from models.sklearn.ensemble import AdaBoostClassifierManager, RandomForestClassifierManager

MANAGER_MAP = {
    'LogisticRegression': LogisticRegressionManager,
    'SVC': SVCManager,
    'KNNC':  KNeighborsClassifierManager,
    'AdaBoostClassifier': AdaBoostClassifierManager,
    'RFC': RandomForestClassifierManager
}