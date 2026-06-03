import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------
# LOAD DATA
# -----------------------
data = pd.read_csv("student_data.csv")

data['Performance'] = data['Performance'].map({'Fail': 0, 'Pass': 1})

X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Performance']

# -----------------------
# TRAIN MODEL
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------
# UI START
# -----------------------
st.title("🎓 Student Performance Prediction System")
st.write("This AI predicts whether a student will PASS or FAIL")

st.sidebar.header("Enter Student Details")

hours = st.sidebar.number_input("Hours Studied", 0, 24, 5)
attendance = st.sidebar.number_input("Attendance (%)", 0, 100, 75)
score = st.sidebar.number_input("Previous Score", 0, 100, 60)

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict Result"):
    result = model.predict([[hours, attendance, score]])

    if result[0] == 1:
        st.success("🎉 Student will PASS")
    else:
        st.error("❌ Student will FAIL")

# -----------------------
# MODEL ACCURACY
# -----------------------
st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# -----------------------
# GRAPH (BONUS FEATURE)
# -----------------------
st.subheader("📈 Pass vs Fail Distribution")

fig, ax = plt.subplots()
data['Performance'].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel("Performance")
ax.set_ylabel("Count")

st.pyplot(fig)