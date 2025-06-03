import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import datetime
from sklearn.model_selection import train_test_split  
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="logo.png",
    layout="wide",  # or "wide, centered"
    initial_sidebar_state="expanded" # collapsed, expanded , auto
)

INR_TO_PKR = 3.28

def plot_feature_importance(model, user_input):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_names = list(user_input.keys())
        sorted_idx = importance.argsort()[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=sorted_importance, y=sorted_features, palette="Blues_r", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("This model doesn't support feature importances.")

def plot_price_distribution(y_test, prediction):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y_test, kde=True, bins=30, color='skyblue', ax=ax)
    ax.axvline(prediction, color='red', linestyle='--', label='Your Prediction')
    ax.set_title("Selling Price Distribution")
    ax.set_xlabel("Selling Price")
    ax.legend()
    st.pyplot(fig)

def show_similar_price_stats(data, user_input):
    df = data.copy()
    mask = (
        (df["fuel"] == user_input["fuel"]) &
        (df["transmission"] == user_input["transmission"]) &
        (df["seller_type"] == user_input["seller_type"]) &
        (abs(df["year"] - user_input["year"]) <= 2)
    )
    similar = df[mask]
    if len(similar) > 0:
        avg_price = similar["selling_price"].mean()
        min_price = similar["selling_price"].min()
        max_price = similar["selling_price"].max()

        st.markdown("### Similar Cars Price Stats")
        st.metric("Average Price", f"{avg_price * INR_TO_PKR:,.0f} PKR")
        st.metric("Price Range", f"{min_price * INR_TO_PKR:,.0f} PKR - {max_price * INR_TO_PKR:,.0f} PKR")
    else:
        st.info("No similar cars found to compare.")

def avg_price_by_fuel(df):
    avg_prices = df.groupby('fuel')['selling_price'].mean().sort_values()
    st.subheader("Average Selling Price by Fuel Type")
    st.bar_chart(avg_prices * INR_TO_PKR)

def price_vs_year(df):
    avg_by_year = df.groupby('year')['selling_price'].mean().sort_index()
    st.subheader("Average Price vs. Manufacturing Year")
    st.line_chart(avg_by_year * INR_TO_PKR)

def km_driven_by_price(df):
    st.subheader("Average Price vs. KM Driven")

    # Bin km_driven into readable ranges
    bins = [0, 20000, 40000, 60000, 80000, 100000, 150000, 200000, 300000]
    labels = ['0–20K', '20K–40K', '40K–60K', '60K–80K', '80K–100K', '100K–150K', '150K–200K', '200K+']
    df['km_range'] = pd.cut(df['km_driven'], bins=bins, labels=labels, include_lowest=True)

    # Calculate average price in PKR
    avg_price = df.groupby('km_range')['selling_price'].mean().reset_index()
    avg_price['selling_price'] = avg_price['selling_price'] * 3.28

    # Plot line chart
    fig, ax = plt.subplots(figsize=(6, 5))  # Adjusted size
    sns.lineplot(data=avg_price, x='km_range', y='selling_price', marker='o', ax=ax, color='teal')

    ax.set_title("KM Driven vs Average Selling Price", fontsize=12)
    ax.set_xlabel("KM Driven Range", fontsize=10)
    ax.set_ylabel("Average Price (PKR)", fontsize=10)
    ax.tick_params(axis='x', rotation=45)  # Rotate x labels for readability
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


    
def transmission_pie(df):
    counts = df['transmission'].value_counts()
    st.subheader("Transmission Type Distribution")
    st.pyplot(counts.plot.pie(autopct='%1.1f%%', ylabel='', figsize=(4, 4)).figure)

def popular_engines(df):
    top_engines = df['engine'].value_counts().nlargest(10)
    st.subheader("Most Common Engine Sizes")
    st.bar_chart(top_engines)
    
# Helper Functions
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except FileNotFoundError:
        st.error("Employees.csv file not found.")
        return None

@st.cache_resource

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('model.joblib')
        return model
    except FileNotFoundError:
        st.error("model.joblib file not found.")
        return None
    
def preprocess_user_input(user_data):
    """Preprocess user input to match training data format"""
    # Create DataFrame from user input
    df_t = pd.read_csv("cars_processed.csv")
    # df_t.drop(["selling_price"],inplace=True)
    df_t.drop(columns=["selling_price"],inplace=True)
    training_columns = df_t.columns
    df = pd.DataFrame([user_data])
    label_encoders = joblib.load("label_encoders.joblib")

    # Step 2: Apply LabelEncoders
    for col in ['fuel', 'seller_type', 'transmission', 'owner']:
        le = label_encoders.get(col)
        if le:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # Handle unknown label gracefully by using the most frequent (or 0)
                df[col] = 0
                print(f"[Warning] Unknown value for {col}. Replaced with default 0.")

    # Step 3: Reorder columns
    df = df[training_columns]

    return df

def main():

    model = load_model()
    raw_data = load_data("cars_data.csv")
    data = load_data("cars_processed.csv")
    
    with st.sidebar:    
        col1,col2 = st.columns([1,4])
        with col1:
            logo = 'logo.png'
            st.image(logo,width=60)
        with col2:
            st.markdown("# **Car Worth AI**")    
        st.markdown("### About This App")
        st.markdown("""
        Predict the **resale value** of your used car using a trained machine learning model.
        
        - Based on 1000s of real car listings  
        - Powered by **Random Forest Regressor**
        - Reliable, fast, and easy to use!
        """)

        avg_price = int(raw_data["selling_price"].mean())
        total_cars = len(raw_data)
        top_fuel = raw_data["fuel"].mode()[0]

        st.markdown(f"""
        **Dataset Overview**
        - Total Cars: `{total_cars}`
        - Avg Price:  `{avg_price * INR_TO_PKR:,} PKR`
        - Most Common Fuel: `{top_fuel}`
        """)
        
    #MAIN
    st.title("Used Car Price Predictor")
    st.subheader("Get an estimated market value for your used car in seconds!")

    st.markdown("""
    Welcome to the **Used Car Price Predictor** app!  
    This tool helps you estimate the resale price of your car based on key features like mileage, fuel type, engine power, and more.

    """)

    # ### How it Works:
    # - Enter your car's details below
    # - Our trained machine learning model predicts its price
    # - See instant, data-driven results!

    # ---


    st.info("📌 Tip: For the most accurate prediction, try to provide realistic values.")
    if model is None:
        st.stop()
    
    tab1,tab2 = st.tabs(['Prediction','Key Insights'])
    with tab1:
        # Create input form
        # col1,col2 = st.columns([2,1]) 
        col1,col2 = st.columns(2,gap='large')
        with col1 :
            st.markdown("###### Fill out the form below to predict price")
            with st.form('car_data'):
                # st.markdown("**Manufacturing Year**")
                year = st.number_input("Manufacturing Year", min_value=1986, max_value=datetime.datetime.now().year, value=2010)
                km_driven = st.number_input("Meter Rating (km)", min_value=0 , value=1300)
                fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol","LPG","CNG"])
                seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
                transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])
                owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'])
                # mileage = st.number_input("Mileage (kmpl) : ", min_value=0 , value=23.4)
                mileage = st.number_input("Mileage (kmpl) : ", min_value=0.0 , value=23.0,step=0.1)
                engine = st.number_input("Engine (CC) : ", min_value=0.0 , value=1493.0,step=0.1)
                max_power = st.number_input("Max Power (bhp) : ", min_value=0.0 , value=63.0,step=0.1)
                seats = st.number_input("Seats : ", min_value=2 , max_value=130,value=6)
                submit_button = st.form_submit_button("Predict Price", use_container_width=True)
                
        with col2:
            transmission_pie(raw_data)
            price_vs_year(raw_data)
            
        if submit_button:
            user_input = {
                'year':year,
                'km_driven':km_driven,
                'fuel':fuel,
                'seller_type':seller_type,
                'transmission':transmission,
                'owner':owner,
                'mileage':mileage,
                'engine':engine,
                'max_power':max_power,
                'seats':seats
            }
            processed_input = preprocess_user_input(user_input)
            prediction = model.predict(processed_input)[0] #in indian rupees (INR)
            predict_pkr = prediction * INR_TO_PKR # 1 INR = 3.28 PKR
            st.success(f"Estimated Selling Price: {int(predict_pkr):,} PKR")
            
            st.markdown("### Model Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                # plot_feature_importance(model,user_input)
                plot_feature_importance(model,processed_input)
            with col2:
                X = data.drop('selling_price', axis=1)
                y = data['selling_price']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                plot_price_distribution(y_test * INR_TO_PKR ,predict_pkr) 
                
            show_similar_price_stats(raw_data,user_input)
            
        with tab2:
            col1,col2 = st.columns(2)
            with col1:   
                km_driven_by_price(raw_data)
            with col2:
                popular_engines(raw_data)
                 
                 
            col1,col2 = st.columns(2)
            with col1:   
                avg_price_by_fuel(raw_data)     
            with col2:
                pass
            
    # Footer
    st.markdown("""
    ---
    © 2025 Car Worth | Used Car Price Predictor
    """)

if __name__ == "__main__":
    main()    