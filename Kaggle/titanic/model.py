import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

test_passenger_ids = test_data["PassengerId"]

train_data["Age"].fillna(train_data["Age"].median(), inplace=True)
test_data["Age"].fillna(test_data["Age"].median(), inplace=True)

train_data['Cabin'].fillna('U', inplace=True)
test_data['Cabin'].fillna('U', inplace=True)

train_data["CabinLetter"] = train_data["Cabin"].apply(lambda x: x[0])
test_data["CabinLetter"] = test_data["Cabin"].apply(lambda x: x[0])

test_data["Fare"].fillna(test_data["Fare"].mode()[0], inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})

train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data["Title"] = train_data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

train_data["Title"] = train_data["Title"].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data["Title"] = train_data["Title"].replace(["Mlle", "Ms"], "Miss")
train_data["Title"] = train_data["Title"].replace("Mme", "Mrs")
train_data = pd.get_dummies(train_data, columns=["Title"], drop_first=True)
test_data["Title"] = test_data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data["Title"] = test_data["Title"].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data["Title"] = test_data["Title"].replace(["Mlle", "Ms"], "Miss")
test_data["Title"] = test_data["Title"].replace("Mme", "Mrs")
test_data = pd.get_dummies(test_data, columns=["Title"], drop_first=True)

train_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
test_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

train_data["IsAlone"] = (train_data["FamilySize"] == 1).astype(int)
test_data["IsAlone"] = (test_data["FamilySize"] == 1).astype(int)

train_data["Fare"] = np.log1p(train_data["Fare"]) 
test_data["Fare"] = np.log1p(test_data["Fare"]) 

train_data = pd.get_dummies(train_data, columns=["CabinLetter"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["CabinLetter"], drop_first=True)

train_data["Pclass_Fare"] = train_data["Pclass"] * train_data["Fare"]
test_data["Pclass_Fare"] = test_data["Pclass"] * test_data["Fare"]

from sklearn.model_selection import train_test_split

# Train verisini X ve y olarak ayır
X_train = train_data.drop(columns=["Survived"])
y_train = train_data["Survived"]

# Test verisini X olarak ayır (çünkü "Survived" sütunu yok)
X_test = test_data.copy()

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Eğitim setini daha küçük bir doğrulama setiyle test etmek için bölebiliriz
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Modeli oluştur
model = LogisticRegression(max_iter=200)

# Modeli eğit
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_valid)

# Modelin doğruluk oranını hesapla
accuracy = accuracy_score(y_valid, y_pred)
print(f"Lojistik Regresyon Doğruluk Oranı: {accuracy:.4f}")

final_predictions = model.predict(X_test)
# Tahmin sonuçlarını CSV dosyası olarak kaydetme
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": final_predictions})
submission.to_csv("submission.csv", index=False)