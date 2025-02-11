import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Charger les données
train_df = pd.read_csv('train_data.csv')  # Assurez-vous de fournir le bon fichier

# 1. Suppression des variables fortement corrélées
numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = train_df[numeric_cols].corr()
high_corr_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns 
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.9]
cols_to_drop = list(set([col1 for col1, col2 in high_corr_pairs]))
train_df.drop(columns=cols_to_drop, inplace=True)

# 2. Test de Khi-deux pour les variables catégorielles
categorical_cols = train_df.select_dtypes(include=['object']).columns
chi2_results = {}
for col1 in categorical_cols:
    for col2 in categorical_cols:
        if col1 != col2:
            contingency_table = pd.crosstab(train_df[col1], train_df[col2])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            chi2_results[(col1, col2)] = p_value
dependent_categorical_vars = [(col1, col2) for (col1, col2), p in chi2_results.items() if p < 0.05]

# 3. Normalisation et PCA
numeric_data = train_df.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(scaled_data)

# 4. Transformation de la variable cible
bins = [0, 150000, 250000, 350000, float('inf')]
labels = ['bas', 'moyen', 'élevé', 'très élevé']
train_df['PriceCategory'] = pd.cut(train_df['SalePrice'], bins=bins, labels=labels)
y = train_df['PriceCategory']

# 5. Encodage et rééchantillonnage
X = pd.get_dummies(train_df.drop('SalePrice', axis=1))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 6. Entraînement du modèle
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Évaluation du modèle
y_pred = model.predict(X_val)
print("Matrice de confusion :")
print(confusion_matrix(y_val, y_pred))
print("\nRapport de classification :")
print(classification_report(y_val, y_pred))

# Sauvegarde du modèle
import joblib
joblib.dump(model, 'trained_model.pkl')
print("Modèle sauvegardé sous 'trained_model.pkl'")
