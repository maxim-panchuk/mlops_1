# src/train.py
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        """
        Обучаем DecisionTreeClassifier.
        """
        self.model = DecisionTreeClassifier()
        self.model.fit(X_train, y_train)
        return self.model

    def export_tree(self, feature_names, class_names_list, output_file='tree.dot'):
        """
        Сохраняем структуру дерева в файл tree.dot.
        """
        export_graphviz(
            self.model,
            out_file=output_file,
            feature_names=feature_names,
            class_names=class_names_list
        )

    def save_model(self, model_path='model.joblib'):
        """
        Сохраняем обученную модель на диск.
        """
        joblib.dump(self.model, model_path)