import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# Veriyi yükleme
file_path = 'C:\\Users\\LENOVO\\Desktop\\student\\student placement data.csv'  
data = pd.read_csv(file_path)

# 1. Akademik Performans Analizi
# Sayısal sütunları belirleme
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Ortalama yüzdeleri hesaplama
academic_performance = data[numeric_columns].mean()

# Görselleştirme: Akademik performans yüzdeleri
plt.figure(figsize=(12, 6))
academic_performance.sort_values().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Akademik Performans Yüzdeleri', fontsize=16)
plt.xlabel('Dersler', fontsize=12)
plt.ylabel('Ortalama Yüzde', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Korelasyon Analizi
# Korelasyon matrisi hesaplama
correlation_matrix = data[numeric_columns].corr()

# Görselleştirme: Korelasyon matrisi
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Korelasyon Matrisi', fontsize=16)
plt.show()

# 3. Kategorik Analiz
# Hackathon katılımı ve iş tercihi ilişkisi
if 'hackathons' in data.columns and 'Job/Higher Studies?' in data.columns:
    hackathon_vs_job = data.groupby('hackathons')['Job/Higher Studies?'].value_counts(normalize=True).unstack()

    # Görselleştirme
    hackathon_vs_job.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title('Hackathon Katılımı ve İş Tercihi Dağılımı', fontsize=16)
    plt.xlabel('Hackathon Katılım Sayısı', fontsize=12)
    plt.ylabel('Oran', fontsize=12)
    plt.legend(title='İş / Yüksek Lisans', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.show()
else:
    print("Hackathon veya iş tercihi sütunları bulunamadı.")

# Eksik veri kontrolü
missing_data = data.isnull().sum()
print("Eksik veri sayısı:\n", missing_data)

#Hedef değişken ve özellik seçimi
# Örnek: "Acedamic percentage in Operating Systems" tahmin edilecek hedef
target_column = 'Acedamic percentage in Operating Systems'

# Kategorik değişkenleri sayısal hale getirme
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Bağımsız ve bağımlı değişkenler
X = data.drop(columns=[target_column])  # Özellikler
y = data[target_column]  # Hedef değişken

# Veriyi eğitim ve test seti olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regresyon modeli oluşturma
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performansı:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Tahmin-Gözlem Karşılaştırması
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Gerçek Değerler vs. Tahminler", fontsize=16)
plt.xlabel("Gerçek Değerler", fontsize=12)
plt.ylabel("Tahmin Edilen Değerler", fontsize=12)
plt.grid(alpha=0.5)
plt.show()


# ---------------- Polinomsal Regresyon ---------------- #
print("\n--- Polinomsal Regresyon ---")
# Polinomsal özellikler oluşturma
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Polinomsal regresyon modeli
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Tahmin yapma
y_pred_poly = poly_model.predict(X_test_poly)

# Performans değerlendirme
mse_poly = mean_squared_error(y_test, y_pred_poly)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"Mean Absolute Error (MAE): {mae_poly:.2f}")
print(f"R² Score: {r2_poly:.2f}")

# ---------------- SVR ---------------- #
print("\n--- Destek Vektör Regresyonu (SVR) ---")
# Veriyi standartlaştırma (SVR'de genellikle önemlidir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR modeli
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Tahmin yapma
y_pred_svr = svr_model.predict(X_test_scaled)

# Performans değerlendirme
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"Mean Squared Error (MSE): {mse_svr:.2f}")
print(f"Mean Absolute Error (MAE): {mae_svr:.2f}")
print(f"R² Score: {r2_svr:.2f}")

# ---------------- Tahmin Karşılaştırma Görselleştirme ---------------- #
plt.figure(figsize=(14, 6))

# Polinomsal Regresyon
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_poly, color='blue', edgecolor='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Polinomsal Regresyon: Gerçek vs. Tahmin", fontsize=14)
plt.xlabel("Gerçek Değerler", fontsize=12)
plt.ylabel("Tahmin Edilen Değerler", fontsize=12)
plt.grid(alpha=0.5)

# SVR
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svr, color='green', edgecolor='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("SVR: Gerçek vs. Tahmin", fontsize=14)
plt.xlabel("Gerçek Değerler", fontsize=12)
plt.ylabel("Tahmin Edilen Değerler", fontsize=12)
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()

# XGBoost modeli tanımlama
xgb_model = XGBRegressor(random_state=42)

# Hiperparametre ızgarası
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# GridSearch ile hiperparametre optimizasyonu
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# En iyi modeli seçme
best_xgb_model = grid_search.best_estimator_

# Test seti üzerinde tahmin yapma
y_pred_xgb = best_xgb_model.predict(X_test)

# Performans değerlendirme
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\n--- XGBoost ---")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
print(f"R² Score: {r2_xgb:.2f}")