import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- Load dataset ---
file_path = 'dropped_leads.csv'
full_data = pd.read_csv(file_path)

# --- Handle missing values for numeric columns ---
num_cols = full_data.select_dtypes(include=['float64', 'int64']).columns
full_data[num_cols] = full_data[num_cols].fillna(0)

# --- Fill missing for object columns ---
obj_cols = full_data.select_dtypes(include=['object']).columns.tolist()
full_data[obj_cols] = full_data[obj_cols].fillna('Unknown')

# --- Group rare categories in LF-Industry ---
def group_top_categories(df, column, top_n=15, new_label='Other'):
    top_categories = df[column].value_counts().nlargest(top_n).index
    df[column] = df[column].apply(lambda x: x if x in top_categories else new_label)
    return df

full_data = group_top_categories(full_data, 'LF-Industry', top_n=15)

# --- One-hot encode LF-Industry ---
full_data = pd.get_dummies(full_data, columns=['LF-Industry'], drop_first=True)

# --- Label encode remaining object columns ---
for col in full_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    full_data[col] = le.fit_transform(full_data[col].astype(str))

# --- Prepare features and target ---
X = full_data.drop(columns=['Converted'])
y = full_data['Converted']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest Classifier ---
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# --- Evaluate ---
y_pred = rfc.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- Feature Importance ---
importances = rfc.feature_importances_
feature_names = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(14,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='PuBu', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()