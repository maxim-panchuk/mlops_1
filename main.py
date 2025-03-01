from src.preprocess import Preprocessor, split_data
from src.train import ModelTrainer
from src.utils import evaluate

data_path = "./data"
preprocessor = Preprocessor(data_path)

# 1. Load and splitting data
data = preprocessor.load_data()
X_train, X_test, y_train, y_test = split_data(data)

# 2. Training
trainer = ModelTrainer()
model = trainer.train(X_train, y_train)

# 3. Export tree struct
class_names_list = [str(cls) for cls in data['class'].unique().tolist()]
trainer.export_tree(
    feature_names=['variance', 'skewness', 'curtosis', 'entropy'],
    class_names_list=class_names_list,
    output_file='tree.dot'
)

# 4. Eval
evaluate(model, X_test, y_test)

# 5. Save model
trainer.save_model('model.joblib')