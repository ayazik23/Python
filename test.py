import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dataset
df = pd.read_csv(r'C:\Users\kamta\Desktop\AI1\data.csv')
df = df.fillna('-')

# Grouping functions
def striking_accuracy_group(acc):
    if acc < 40:
        return 'Low Striking Accuracy'
    elif acc <= 60:
        return 'Medium Striking Accuracy'
    else:
        return 'High Striking Accuracy'

def takedown_accuracy_group(acc):
    if acc < 40:
        return 'Low Takedown Accuracy'
    elif acc <= 60:
        return 'Medium Takedown Accuracy'
    else:
        return 'High Takedown Accuracy'

def stance_group(stance):
    known_stances = ['Orthodox', 'Southpaw', 'Switch']
    return stance if stance in known_stances else 'Other Stance'

def age_group(dob):
    try:
        year_of_birth = int(dob.split('-')[0]) if '-' in dob else 2000
    except:
        year_of_birth = 2000
    age = datetime.now().year - year_of_birth
    return 'Young Fighter' if age < 30 else 'Old Fighter'

df['striking_accuracy_group'] = df['significant_striking_accuracy'].apply(striking_accuracy_group)
df['takedown_accuracy_group'] = df['takedown_accuracy'].apply(takedown_accuracy_group)
df['stance_group'] = df['stance'].apply(stance_group)
df['age_group'] = df['date_of_birth'].apply(age_group)

# ML Functions
def prepare_ml_data():
    features = ['significant_striking_accuracy', 'takedown_accuracy']
    df_ml = df.copy()
    df_ml['stance_encoded'] = df_ml['stance_group'].astype('category').cat.codes
    df_ml['age_encoded'] = df_ml['age_group'].astype('category').cat.codes
    X = df_ml[features + ['stance_encoded', 'age_encoded']]
    y = df_ml['striking_accuracy_group'].astype('category').cat.codes
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = prepare_ml_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, accuracy_score(y_test, y_pred)

# Filtering Function
def filter_fighters(striking_accuracy=None, takedown_accuracy=None, stance=None, age_group=None):
    filtered_df = df.copy()
    if striking_accuracy:
        filtered_df = filtered_df[filtered_df['striking_accuracy_group'] == striking_accuracy]
    if takedown_accuracy:
        filtered_df = filtered_df[filtered_df['takedown_accuracy_group'] == takedown_accuracy]
    if stance:
        filtered_df = filtered_df[filtered_df['stance_group'] == stance]
    if age_group:
        filtered_df = filtered_df[filtered_df['age_group'] == age_group]
    return filtered_df

# Streamlit UI
st.title("Fighter search")

# Filters
st.sidebar.header("Filter fighters")

striking_accuracy = st.sidebar.selectbox("Striking Accuracy", ["All", "Low Striking Accuracy", "Medium Striking Accuracy", "High Striking Accuracy"])
takedown_accuracy = st.sidebar.selectbox("Takedown Accuracy", ["All", "Low Takedown Accuracy", "Medium Takedown Accuracy", "High Takedown Accuracy"])
stance = st.sidebar.selectbox("Stance", ["All", "Orthodox", "Southpaw", "Switch", "Other Stance"])
age_group = st.sidebar.selectbox("Age Group", ["All", "Young Fighter", "Old Fighter"])

if striking_accuracy == "All":
    striking_accuracy = None
if takedown_accuracy == "All":
    takedown_accuracy = None
if stance == "All":
    stance = None
if age_group == "All":
    age_group = None

filtered_df = filter_fighters(striking_accuracy, takedown_accuracy, stance, age_group)

st.subheader("Filtered Fighters")
st.dataframe(filtered_df[['name', 'nickname', 'striking_accuracy_group', 'takedown_accuracy_group', 'stance_group', 'age_group']])

# Search
search_query = st.text_input("Search by name or nickname")
if search_query:
    results = filtered_df[filtered_df['name'].str.contains(search_query, case=False) |
                          filtered_df['nickname'].str.contains(search_query, case=False)]
    if results.empty:
        st.warning("No fighters found for your search.")
    else:
        st.subheader("Search Results")
        st.dataframe(results[['name', 'nickname', 'striking_accuracy_group', 'takedown_accuracy_group', 'stance_group', 'age_group']])

# Train model
if st.button("Train Model"):
    model, acc = train_model()
    st.success(f"Model trained successfully! Accuracy: {acc:.2f}")

st.write("Made by Miras Kamatay")
