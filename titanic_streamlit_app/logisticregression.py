import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Titanic Logistic Regression App", layout="centered")

# Title
st.title(" Titanic Survival Prediction App")

# Step 1: Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic_train.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df

df = load_data()

# Step 1c: Visualizations
st.subheader(" Data Visualization (Exploration)")

tab1, tab2, tab3 = st.tabs(["Survival by Sex", "Fare Distribution", "Age vs Survival"])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Survived", hue="Sex", ax=ax1)
    ax1.set_title("Survival Count by Sex")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x="Fare", bins=30, kde=True, ax=ax2)
    ax2.set_title("Fare Distribution")
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Survived", y="Age", ax=ax3)
    ax3.set_title("Age Distribution by Survival")
    st.pyplot(fig3)

# Step 2: Feature + Label split
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Model evaluation
y_pred = model.predict(X_val)
st.subheader(" Model Performance")
st.write("Accuracy:", accuracy_score(y_val, y_pred))
st.write("F1 Score:", f1_score(y_val, y_pred))

# Step 5: Single user input prediction
st.subheader(" Make a Manual Prediction")

pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": 0 if sex == "male" else 1,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": {"S": 0, "C": 1, "Q": 2}[embarked]
}])

if st.button("Predict for Passenger"):
    pred = model.predict(input_data)[0]
    st.success("Prediction: Survived" if pred == 1 else "Prediction: Did Not Survive")

# Step 6: Batch prediction from Titanic_test.csv
st.subheader(" Predict on Titanic_test.csv")

if st.checkbox("Run model on test dataset"):
    try:
        test_df = pd.read_csv("Titanic_test.csv")

        # Preprocessing
        test_df["Age"] = test_df["Age"].fillna(df["Age"].median())
        test_df["Fare"] = test_df["Fare"].fillna(df["Fare"].median())
        test_df["Embarked"] = test_df["Embarked"].fillna("S")
        test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
        test_df["Embarked"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

        X_test = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        test_df["Predicted_Survived"] = model.predict(X_test)

        st.write(test_df[["PassengerId", "Predicted_Survived"]].head())

        csv = test_df[["PassengerId", "Predicted_Survived"]].to_csv(index=False)
        st.download_button(" Download Predictions", csv, "titanic_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing test dataset: {e}")
