import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers)

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female':'1', 'male':'0'})

# Fill the nan values in the age column
passengers['Age'].fillna(value=round(passengers['Age'].mean()), inplace = True)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda p: 1 if p == 2 else 0 )
print(passengers)
# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
print(features)
print(survival)
# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(features, survival)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scale = StandardScaler()
train_features = scale.fit_transform(train_features)
test_features = scale.transform(test_features)

# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Score the model on the train data
print(model.score(train_features, train_labels))

# Score the model on the test data
print(model.score(test_features, test_labels))

# Analyze the coefficients
print(model.coef_)
print(list(zip(['Sex', 'Age', 'FirstClass', 'SecondClass'], model.coef_[0])))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Selim = np.array([0.0,21.0,1.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Selim])

# Scale the sample passenger features
sample_passengers = scale.transform(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
