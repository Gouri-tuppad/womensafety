"""app/ml – Machine-learning pipeline for SafeWay risk prediction."""

# Ensure compatibility between scikit-learn >=1.6 and xgboost <=2.1.4
try:
    from sklearn.base import ClassifierMixin, RegressorMixin

    if not hasattr(RegressorMixin, "_estimator_type"):
        RegressorMixin._estimator_type = "regressor"
    if not hasattr(ClassifierMixin, "_estimator_type"):
        ClassifierMixin._estimator_type = "classifier"
except ImportError:
    pass
