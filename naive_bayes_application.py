import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# تحميل البيانات وتدريب النموذج
df = pd.read_excel('churn_dataset.xlsx')

# ترميز البيانات
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['Churn'] = encoder.fit_transform(df['Churn'])

X = df[['Age', 'Tenure', 'Sex']]
y = df['Churn']

# تقسيم الداتا
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# تدريب الموديل
model = GaussianNB()
model.fit(X_train, y_train)


# -------------------------------------
# Streamlit App
# -------------------------------------

st.title("Customer Churn Prediction App 🚀")
st.write("Enter customer information to predict if they will churn or not.")

# أخذ مدخلات من المستخدم
age = st.number_input("Enter Age:", min_value=0, max_value=100, value=30)
tenure = st.number_input("Enter Tenure (Years as Customer):", min_value=0, max_value=50, value=5)
sex = st.selectbox("Select Gender:", options=["Male", "Female"])

# تحويل الجنس إلى أرقام (زي اللي عملناه وقت التدريب)
sex_encoded = 1 if sex == "Male" else 0

# زر للتوقع
if st.button("Predict"):
    # تجهيز الداتا المدخلة
    input_data = np.array([[age, tenure, sex_encoded]])
    input_data_scaled = scaler.transform(input_data)

    # التنبؤ
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0]

    # عرض النتيجة
    if prediction == 1:
        st.error(f"❌ The customer is likely to churn with probability {prediction_proba[1]:.2f}")
    else:
        st.success(f"✅ The customer is likely to stay with probability {prediction_proba[0]:.2f}")


    # عرض الاحتمالات لكل كلاس
    st.subheader("Prediction Probabilities:")
    st.write(f"- Not Churn (0): {prediction_proba[0]:.2f}")
    st.write(f"- Churn (1): {prediction_proba[1]:.2f}")
