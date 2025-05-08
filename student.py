import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# Simulated user database (in-memory)
# ----------------------------
if 'users' not in st.session_state:
    st.session_state.users = {'admin': 'admin123'}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# ----------------------------
# Sidebar - Login/Register
# ----------------------------
st.sidebar.title("🔐 User Authentication")

auth_mode = st.sidebar.radio("Choose Action:", ["Login", "Register"])

if auth_mode == "Register":
    new_user = st.sidebar.text_input("👤 Create Username")
    new_pass = st.sidebar.text_input("🔑 Create Password", type="password")
    if st.sidebar.button("Register"):
        if new_user in st.session_state.users:
            st.sidebar.warning("Username already exists.")
        elif new_user and new_pass:
            st.session_state.users[new_user] = new_pass
            st.sidebar.success("Registration successful! You can now log in.")
        else:
            st.sidebar.warning("Fill all fields to register.")

elif auth_mode == "Login":
    username = st.sidebar.text_input("👤 Username")
    password = st.sidebar.text_input("🔑 Password", type="password")
    if st.sidebar.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid credentials")

# ----------------------------
# If Logged In: Show Main App
# ----------------------------
if st.session_state.logged_in:

    st.title("📚 Student Performance Predictor")
    st.markdown("This app predicts student performance using a trained **Machine Learning Model** and gives **smart feedback**.")

    try:
        # Load dataset
        df = pd.read_csv('student_performance_dataset.csv')

        # Split data
        X = df[['academic_score', 'attendance', 'phone_usage', 'concentration_power']]
        y = df['final_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        lr_model = LinearRegression().fit(X_train, y_train)
        ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
        lasso_model = Lasso(alpha=0.1).fit(X_train, y_train)

        # Evaluate models
        models = {'Linear Regression': lr_model, 'Ridge Regression': ridge_model, 'Lasso Regression': lasso_model}
        best_model = None
        best_r2 = -np.inf

        for name, model in models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            st.write(f"{name} → R²: {r2:.2f}, RMSE: {rmse:.2f}")
            if r2 > best_r2:
                best_r2 = r2
                best_model = model

        st.success(f"✅ Best model selected: {type(best_model).__name__} (R²: {best_r2:.2f})")

        st.header("📥 Enter Your Details")

        english = st.number_input("English Marks (out of 100)", 0, 100, 0)
        tamil = st.number_input("Tamil Marks (out of 100)", 0, 100, 0)
        math = st.number_input("Mathematics Marks (out of 100)", 0, 100, 0)
        science = st.number_input("Science Marks (out of 100)", 0, 100, 0)
        social = st.number_input("Social Science Marks (out of 100)", 0, 100, 0)

        attendance = st.slider("Attendance Percentage (0-100)", 0, 100, 75)
        phone_usage = st.slider("Phone Usage (hrs/day)", 0.0, 12.0, 4.0)
        concentration_power = st.slider("🧠 Concentration Power (0–100)", 0, 100, 70)

        academic_score = (english + tamil + math + science + social) / 5

        if st.button("🔍 Predict Performance"):
            input_data = pd.DataFrame([[academic_score, attendance, phone_usage, concentration_power]],
                                      columns=['academic_score', 'attendance', 'phone_usage', 'concentration_power'])
            try:
                predicted_score = best_model.predict(input_data)[0]
                predicted_score = max(0, min(predicted_score, 100))

                st.subheader("📊 Predicted Final Score")
                st.success(f"{predicted_score:.2f} / 100")

                st.subheader("📝 Personalized Feedback")
                feedback = ""

                # Academic feedback
                if academic_score >= 85:
                    feedback += "📘 Academic Performance: Excellent! Keep it up.\n"
                elif academic_score >= 70:
                    feedback += "📘 Academic Performance: Good, but there's room for improvement.\n"
                else:
                    feedback += "📘 Academic Performance: Needs improvement. Focus on studies.\n"

                # Attendance feedback
                if attendance >= 90:
                    feedback += "🎯 Attendance: Great job maintaining high attendance!\n"
                elif attendance >= 75:
                    feedback += "🎯 Attendance: Satisfactory, try to attend more regularly.\n"
                else:
                    feedback += "🎯 Attendance: Low attendance, which may affect performance.\n"

                # Phone usage feedback
                if phone_usage <= 2:
                    feedback += "📱 Phone Usage: Excellent self-control!\n"
                elif phone_usage <= 5:
                    feedback += "📱 Phone Usage: Moderate usage, try to reduce a bit.\n"
                else:
                    feedback += "📱 Phone Usage: Too much phone usage! Consider cutting down.\n"

                # Concentration power feedback
                if concentration_power >= 85:
                    feedback += "🧠 Concentration: Excellent focus and mental stamina.\n"
                elif concentration_power >= 60:
                    feedback += "🧠 Concentration: Good, but can improve with fewer distractions.\n"
                else:
                    feedback += "🧠 Concentration: Needs work — try mindfulness or study breaks.\n"

                st.info(feedback)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    except FileNotFoundError:
        st.error("⚠️ Dataset file 'student_performance_dataset_updated.csv' not found. Please upload it to the app directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.experimental_rerun()

else:
    st.warning("🔒 Please login or register from the sidebar to continue.")
