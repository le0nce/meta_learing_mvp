import logging
import random
import time


DEFAULT_PRIMARY_METRIC = "area_under_roc_curve"
ADDITIONAL_METRICS = [
    "predictive_accuracy",
    "log_loss",
    "balanced_accuracy",
]

BASE_CLASS_WHITELIST = {
    "LogisticRegression", "RidgeClassifier", "LinearSVC", "SVC",
    "KNeighborsClassifier", "NearestCentroid",
    "GaussianNB", "MultinomialNB", "BernoulliNB",
    "DecisionTreeClassifier", "ExtraTreeClassifier",
    "RandomForestClassifier", "ExtraTreesClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier", "AdaBoostClassifier",
    "MLPClassifier",
    "QuadraticDiscriminantAnalysis", "LinearDiscriminantAnalysis",
}

DISALLOWED_TOKENS = [
    "Pipeline", "FeatureUnion", "ColumnTransformer", "SearchCV",
    "CalibratedClassifier", "Stacking", "Bagging", "Voting",
]

ENSEMBLE_CLASS_NAMES = {
    "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
    "HistGradientBoostingClassifier", "AdaBoostClassifier",
}

def with_retry(fn, *, retries: int = 3, base_sleep: float = 1.5, logger: logging.Logger = None):
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if i == retries:
                if logger:
                    logger.error(f"OpenML call failed after {retries} retries: {e}")
                raise
            sleep = base_sleep * (2 ** i) + random.uniform(0, 0.25)
            if logger:
                logger.warning(f"OpenML call failed (attempt {i+1}/{retries}); retrying in {sleep:.1f}s: {e}")
            time.sleep(sleep)

def parse_algo_class(flow_name: str) -> str:
    cls = flow_name.split(".")[-1]
    cls = cls.split(" ")[0]
    return cls