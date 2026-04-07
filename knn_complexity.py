from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score

import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
data=load_iris()
print(data)
X=data.data
y=data.target

X_train, X_test, y_train, y_test=train_test_split ( X, y, test_size=0.3,
                                               random_state=42)
#2. testing different k values
k_range = range(1,31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, X_train, y_train, cv=5)
    print(f"K={k}, CV Scores: {scores}, Mean CV Score: {scores.mean()}")
    cv_scores.append(scores.mean())

plt.plot(k_range, cv_scores, marker='*', color = 'green')
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding Optimal K (Bias-Variance Tradeoff)')
# plt.grid(True)
plt.show()