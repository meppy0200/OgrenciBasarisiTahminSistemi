import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Veri yükleniyor...")
df = pd.read_csv("data/StudentsPerformance.csv")

print(f"Toplam satır sayısı: {len(df)}")
print(f"Sütunlar: {list(df.columns)}")
print("\nİlk 5 satır:")
print(df.head())

df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["success"] = df["average_score"].apply(lambda x: 1 if x >= 60 else 0)

print(f"\nBaşarılı öğrenci sayısı: {df['success'].sum()}")
print(f"Başarısız öğrenci sayısı: {len(df) - df['success'].sum()}")

print("\n🔄 Kategorik değişkenler dönüştürülüyor...")
X = pd.get_dummies(df.drop(["math score", "reading score", "writing score", "average_score", "success"], axis=1))
y = df["success"]

feature_names = X.columns.tolist()
print(f"Toplam özellik sayısı: {len(feature_names)}")
print("Özellikler:")
for i, feature in enumerate(feature_names):
    print(f"  {i+1}. {feature}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nModel Eğitimi...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test doğruluğu: {accuracy:.3f}")

print("\nDetaylı performans raporu:")
print(classification_report(y_test, y_pred, target_names=['Başarısız', 'Başarılı']))

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nÖzellik önemleri (ilk 10):")
print(feature_importance.head(10))

print("\nModel kaydediliyor...")
os.makedirs("model", exist_ok=True)
joblib.dump({
    "model": model, 
    "features": feature_names,
    "feature_importance": feature_importance
}, "model/student_model.pkl")

print("Model başarıyla eğitildi ve kaydedildi!")
print(f"Model dosyası: model/student_model.pkl")
print(f"Özellik sayısı: {len(feature_names)}")