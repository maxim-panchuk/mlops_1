from sklearn.metrics import classification_report
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

output_file = config['train']['output_file']

def evaluate(model, X_test, y_test):
    """
    Output classification_report.
    """
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))


class ModelEvaluator:
    pass