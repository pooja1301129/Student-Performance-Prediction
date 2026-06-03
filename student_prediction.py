import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("student_data.csv")

# Convert target column
data['Performance'] = data['Performance'].map({'Fail': 0, 'Pass': 1})

# Features & Target
X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Performance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ----------------------------
# 2. Train Multiple Models
# ----------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

print("\n📊 Model Accuracy Comparison:")
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name}: {acc:.2f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ----------------------------
# 3. Visualization
# ----------------------------
data['Performance'].value_counts().plot(kind='bar')
plt.title("Pass vs Fail Distribution")
plt.xlabel("Performance")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 4. User Input Prediction
# ----------------------------
print("\n🎓 Student Performance Prediction System")

h = int(input("Enter Hours Studied: "))
a = int(input("Enter Attendance (%): "))
p = int(input("Enter Previous Score: "))

prediction = best_model.predict(pd.DataFrame([[h, a, p]],
                    columns=['Hours_Studied', 'Attendance', 'Previous_Score']))

# ----------------------------
# 5. Output Result
# ----------------------------
print("\n----------------------------")

if prediction[0] == 1:
    print("🎉 Result: Student will PASS")
else:
    print("❌ Result: Student will FAIL")

print("----------------------------")
print(f"Best Model Used: {type(best_model).__name__}")
print(f"Accuracy: {best_accuracy:.2f}")