from sklearn.metrics import classification_report

class ModelEvaluator:
    def evaluate(self, model, X_test, y_test):
        """
        Output classification_report.
        """
        pred = model.predict(X_test)
        print(classification_report(y_test, pred))