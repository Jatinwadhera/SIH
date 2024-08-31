import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error("The file was not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        raise
    except pd.errors.ParserError:
        logging.error("Error parsing the file.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def main():
    # Load the dataset
    df = load_data("Crop_recommendation.csv")
    
    # Validate that the required columns are present
    required_columns = [" N ", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    if not all(column in df.columns for column in required_columns):
        logging.error(f"The dataset must contain the following columns: {required_columns}")
        return
    
    # Select independent and dependent variables
    X = df[[" N ", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    
    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Instantiate the model
    classifier = RandomForestClassifier(random_state=50)
    
    # Fit the model
    classifier.fit(X_train, y_train)
    logging.info("Model training completed.")
    
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.2f}")
    logging.info("Classification report:\n" + classification_report(y_test, y_pred))
    logging.info("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    
    # Make pickle file of our model
    try:
        with open("model.pkl", "wb") as model_file:
            pickle.dump(classifier, model_file)
        logging.info("Model saved successfully as model.pkl")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")

if __name__ == "__main__":
    main()
