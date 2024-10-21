from models.linear import LogisticRegressionFactory

# A map containing all Optuna Managers within this module, for easy reference
FACTORY_MAP = {
    'LogisticRegression': LogisticRegressionFactory
}