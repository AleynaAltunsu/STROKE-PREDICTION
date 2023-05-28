# -*- coding: utf-8 -*-
"""
Created on Fri May 19 03:14:21 2023

@author: aleyn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Veri okuma
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Eksik değer kontrolü
data.isna().sum()

# Kategorik değişkenleri dönüştürme
cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in cols:
    data[col] = le.fit_transform(data[col])

# Verileri ölçeklendirme
scaler = StandardScaler()
X = data.drop(['id', 'stroke'], axis=1)
X_scaled = scaler.fit_transform(X)
y = data['stroke']

# Eğitim ve test verilerine bölme
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model eğitimi
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Doğrulama verileri üzerinde tahmin yapma
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Doğrulama verileri doğruluk oranı: {:.2f}".format(accuracy))

# Sentetik test verileri oluşturma (örnek amaçlı)
test_data = pd.DataFrame({
    "gender": [0, 1, 1],
    "age": [45, 60, 35],
    "hypertension": [0, 1, 0],
    "heart_disease": [0, 0, 1],
    "ever_married": [1, 1, 0],
    "work_type": [2, 3, 1],
    "Residence_type": [0, 1, 1],
    "avg_glucose_level": [85, 110, 95],
    "bmi": [24, 30, 28],
    "smoking_status": [1, 0, 2]
})

# Verileri ölçeklendirme
test_scaled = scaler.transform(test_data)

# Tahmin yapma
predictions = lr.predict(test_scaled)
print("Tahminler:", predictions)

# Sonucu elde etme
test_data["stroke"] = predictions
submission = test_data[["id", "stroke"]]
submission.to_csv("submission.csv", index=False)
