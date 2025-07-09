from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Data
X = np.array([[100, 3],
              [110, 4],
              [98, 3],
              [102, 3],
              [500, 1]])

# LOF model
lof = LocalOutlierFactor(n_neighbors=2)  # Tetangga terdekat yang dipertimbangkan
y_pred = lof.fit_predict(X)
scores = lof.negative_outlier_factor_

for i, (point, score, label) in enumerate(zip(X, scores, y_pred)):
    status = 'Outlier' if label == -1 else 'Inlier'
    print(f"Data {i+1}: {point}, LOF Score: {score:.2f}, Status: {status}")
