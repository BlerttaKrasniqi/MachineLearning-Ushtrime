import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("dataset.csv")

# Fill missing values
df.fillna({
    "Vict Age": df["Vict Age"].median(),
    "Vict Sex": "Unknown",
    "Vict Descent": "Unknown"
}, inplace=True)

# Feature selection
X = df.select_dtypes(include=[np.number]).drop(columns=["Crm Cd"])
y = df["Crm Cd"]

# Balance classes
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

# Remove outliers
iso = IsolationForest(contamination=0.01)
mask = iso.fit_predict(X_res) != -1
X_clean = X_res[mask]
y_clean = y_res[mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# === Supervised ===
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate_classification(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro'),
        "Recall": recall_score(y_true, y_pred, average='macro'),
        "F1 Score": f1_score(y_true, y_pred, average='macro')
    }

lr_metrics = evaluate_classification(y_test, y_pred_lr)
rf_metrics = evaluate_classification(y_test, y_pred_rf)

# === Unsupervised ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

def evaluate_clustering(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        return {
            "Silhouette Score": silhouette_score(X[mask], labels[mask]),
            "Davies-Bouldin Index": davies_bouldin_score(X[mask], labels[mask])
        }
    else:
        return {
            "Silhouette Score": np.nan,
            "Davies-Bouldin Index": np.nan
        }

kmeans_metrics = evaluate_clustering(X_scaled, kmeans_labels)
dbscan_metrics = evaluate_clustering(X_scaled, dbscan_labels)

# === Visualization ===

# Confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), ax=axs[0], cmap="Blues", cbar=False)
axs[0].set_title("Logistic Regression - Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), ax=axs[1], cmap="Greens", cbar=False)
axs[1].set_title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()

# Bar plot of metrics
df_metrics = pd.DataFrame([lr_metrics, rf_metrics], index=["Logistic Regression", "Random Forest"])
df_metrics.plot(kind='bar', figsize=(10,6), title="Performance Metrics - Supervised Learning")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Scatter plots for clustering
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='tab10', s=10)
plt.title("KMeans Clustering")

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='tab10', s=10)
plt.title("DBSCAN Clustering")

plt.tight_layout()
plt.show()

# Summary print
print("\n--- METRIKAT ---")
print("Logistic Regression:", lr_metrics)
print("Random Forest:", rf_metrics)
print("KMeans Clustering:", kmeans_metrics)
print("DBSCAN Clustering:", dbscan_metrics)
