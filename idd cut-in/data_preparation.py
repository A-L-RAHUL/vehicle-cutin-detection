import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def prepare_data(file_path):
    # Load the data
    df = pd.read_csv(file_path, dtype={'width': float, 'height': float, 'xmin': float, 'ymin': float, 'xmax': float, 'ymax': float})
    
    print(df.columns)
    
    # Drop non-numeric columns
    df = df.drop(columns=['filename', 'folder', 'fully_qualified_path', 'file_exists'])
    
    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Prepare training data
train_path = 'C:/Users/Rahul/Desktop/vehicle_cut_in_detection/your_project/data/train.csv'
X_train, y_train = prepare_data(train_path)

# Prepare validation data
validation_path = 'C:/Users/Rahul/Desktop/vehicle_cut_in_detection/your_project/data/validation.csv'
X_validation, y_validation = prepare_data(validation_path)

def prepare_lstm_data():
    # Combine and normalize features
    X = pd.concat([pd.read_csv(train_path), pd.read_csv(validation_path)], axis=0)
    X = X.drop(columns=['filename', 'folder', 'fully_qualified_path', 'file_exists', 'class'])
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Combine and extract labels
    y = pd.concat([pd.read_csv(train_path), pd.read_csv(validation_path)], axis=0)['class']
    
    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Reshape data for LSTM
    timesteps = 10  # Adjust based on your sequence length
    features = X_train.shape[1] // timesteps

    X_train = X_train.reshape((X_train.shape[0], timesteps, features))
    X_val = X_val.reshape((X_val.shape[0], timesteps, features))
    X_test = X_test.reshape((X_test.shape[0], timesteps, features))

    # One-hot encode labels if necessary
    num_classes = len(y_train.unique())
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_lstm_data()
