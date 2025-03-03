import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into X_train, X_test, y_train, y_test.
    
    Args:
        data: Input DataFrame
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple containing train-test split of inputs
    """
    logger.info("Splitting data into features and target")
    X = data[['variance', 'skewness', 'curtosis', 'entropy']]
    y = data['class']
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Split completed. Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


class Preprocessor:
    def __init__(self, data_path: str):
        """
        Initialize Preprocessor with data path
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        logger.info(f"Initialized Preprocessor with data path: {data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV from self.data_path.
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            logger.info(f"Columns: {', '.join(data.columns)}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

