import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Path to the dataset
DATA_FILE = 'Term_Deposit_Predictor\\bank-additional-full.csv'

# Load the dataset
data = pd.read_csv(DATA_FILE, sep=';', na_values='?')

# Drop all data pointer with missing values
data.dropna(inplace=True)

# Encode all categorical variables into numerical values
data['y'] = data['y'].map({'yes': 1, 'no': 0})
data_encode = pd.get_dummies(data, drop_first=True)

# Splits the data into targte and features
y = data["y"]
x = data_encode.drop("y", axis=1)

# Scale the features for standardization and equal weighting 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators= 500, random_state=42)
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)
print(y_predict)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy: " + str(accuracy))
print(classification_report(y_test, y_predict))




