import gradio as gr
import sqlite3
import bcrypt
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import time
import threading
import random
import requests
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load tokenizer and model for Flan-T5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Create a pipeline
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# File to store feedback
FEEDBACK_FILE = "user_feedback.csv"

def huggingface_chatbot(user_input):
    try:
        result = generator(user_input, max_length=150, temperature=0.7, do_sample=True)
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif "generated_text" in result:
            return result["generated_text"]
        elif "text" in result[0]:
            return result[0]["text"]
        else:
            return "⚠️ Could not parse model response."
    except Exception as e:
        return f"Error: {str(e)}"




# Database setup for user authentication
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def register(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return "✅ Registration Successful! You can now log in."
    except sqlite3.IntegrityError:
        return "⚠️ Username already exists. Try another."
    finally:
        conn.close()

def login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode(), result[0]):
        return "✅ Login Successful! Welcome to the marketplace."
    else:
        return "❌ Incorrect username or password. Try again."

# Initialize the feedback CSV if it doesn't exist
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=[
        "Satisfaction", "Useful_Features", "Issues", "Suggestions", "Recommendation", "Name", "Email"
    ]).to_csv(FEEDBACK_FILE, index=False)

def submit_feedback(satisfaction, features, issues, suggestions, recommendation, name, email):
    feedback = pd.read_csv(FEEDBACK_FILE)
    new_entry = {
        "Satisfaction": satisfaction,
        "Useful_Features": ", ".join(features),
        "Issues": issues,
        "Suggestions": suggestions,
        "Recommendation": recommendation,
        "Name": name,
        "Email": email
    }
    feedback = pd.concat([feedback, pd.DataFrame([new_entry])], ignore_index=True)
    feedback.to_csv(FEEDBACK_FILE, index=False)
    return "✅ Thank you for your feedback!"

# Load datasets (using sample data if files don't exist)
try:
    df_lifecycle = pd.read_csv("ecommerce_product_dataset.csv")
except:
    # Sample data if file not found
    df_lifecycle = pd.DataFrame({
        "Category": ["Electronics", "Plastic", "Metal", "Wood", "Composite"] * 20,
        "ProductName": ["Product " + str(i) for i in range(100)],
        "Price": np.random.uniform(10, 500, 100),
        "Rating": np.random.uniform(1, 5, 100),
        "NumReviews": np.random.randint(0, 1000, 100),
        "StockQuantity": np.random.randint(0, 500, 100),
        "Discount": np.random.uniform(0, 50, 100),
        "Sales": np.random.uniform(1, 20, 100)
    })

try:
    df_pricing = pd.read_csv("dynamic_pricing_data_5000.csv")
except:
    df_pricing = pd.DataFrame({
        "Product Name": ["iPhone 13", "Nike Shoes", "Samsung TV", "Adidas Jacket"] * 25,
        "Category": ["Electronics", "Fashion", "Electronics", "Fashion"] * 25,
        "Base Price": np.random.uniform(100, 1000, 100),
        "Competitor Price": np.random.uniform(80, 950, 100),
        "Demand": np.random.choice(["Low", "Medium", "High"], 100),
        "Stock": np.random.randint(0, 500, 100),
        "Reviews": np.random.randint(0, 5000, 100),
        "Rating": np.random.uniform(1, 5, 100),
        "Season": np.random.choice(["Holiday", "Summer", "Winter", "Off-season"], 100),
        "Discount": np.random.uniform(0, 30, 100),
        "Final Price": np.random.uniform(50, 1200, 100)
    })

try:
    df_recommendation = pd.read_csv("synthetic_product_data_5000.csv")
except:
    df_recommendation = pd.DataFrame({
        "product_id": range(100),
        "product_condition": np.random.choice(["New", "Used - Like New", "Used - Good", "Used - Fair"], 100),
        "price": np.random.uniform(10, 500, 100),
        "category": np.random.choice(["Electronics", "Furniture", "Clothing", "Books", "Toys"], 100)
    })

# Preprocessing for product lifecycle prediction
categorical_features_lifecycle = ['Category']
numeric_features_lifecycle = ['Price', 'Rating', 'NumReviews', 'StockQuantity', 'Discount']

preprocessor_lifecycle = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_lifecycle),
    ('num', 'passthrough', numeric_features_lifecycle)
])

# Fit the preprocessor on training data
X_lifecycle = df_lifecycle[["Category", "ProductName", "Price", "Rating", "NumReviews", "StockQuantity", "Discount"]]
y_lifecycle = df_lifecycle["Sales"]  # Target variable

X_transformed_lifecycle = preprocessor_lifecycle.fit_transform(X_lifecycle)

# Train the model
model_lifecycle = RandomForestRegressor(n_estimators=100, random_state=42)
model_lifecycle.fit(X_transformed_lifecycle, y_lifecycle)

def preprocess_input_lifecycle(Category, ProductName, Price, Rating, NumReviews, StockQuantity, Discount):
    input_df = pd.DataFrame([[Category, ProductName, Price, Rating, NumReviews, StockQuantity, Discount]],
                            columns=["Category", "ProductName", "Price", "Rating", "NumReviews", "StockQuantity", "Discount"])
    input_processed = preprocessor_lifecycle.transform(input_df)
    return input_processed

def predict_lifecycle(Category, ProductName, Price, Rating, NumReviews, StockQuantity, Discount):
    try:
        input_data = preprocess_input_lifecycle(Category, ProductName, Price, Rating, NumReviews, StockQuantity, Discount)
        prediction = model_lifecycle.predict(input_data)[0]
        return f"Predicted Product Lifecycle: {round(prediction, 2)} years"
    except Exception as e:
        return f"Error: {str(e)}"

# Encode categorical variables for dynamic pricing
label_encoders = {}
for col in ["Product Name", "Category", "Demand", "Season"]:
    le = LabelEncoder()
    df_pricing[col] = le.fit_transform(df_pricing[col])
    label_encoders[col] = le

# Scale numerical features for dynamic pricing
scaler = StandardScaler()
num_cols = ["Base Price", "Competitor Price", "Stock", "Reviews", "Rating", "Discount"]
df_pricing[num_cols] = scaler.fit_transform(df_pricing[num_cols])

# Train model for dynamic pricing
X_pricing = df_pricing.drop(columns=["Final Price"])
y_pricing = df_pricing["Final Price"]

model_pricing = RandomForestRegressor(n_estimators=100, random_state=42)
model_pricing.fit(X_pricing, y_pricing)

def predict_price(product_name, category, base_price, competitor_price, demand, stock, reviews, rating, season, discount):
    try:
        # Encode categorical features
        category = label_encoders["Category"].transform([category])[0]
        demand = label_encoders["Demand"].transform([demand])[0]
        season = label_encoders["Season"].transform([season])[0]
        product_name = label_encoders["Product Name"].transform([product_name])[0]

        # Scale numerical features
        features = np.array([base_price, competitor_price, stock, reviews, rating, discount]).reshape(1, -1)
        features = scaler.transform(features)

        # Combine features
        final_features = np.concatenate((features.flatten(), [category, demand, season, product_name])).reshape(1, -1)

        # Predict
        predicted_price = model_pricing.predict(final_features)[0]
        return f"Optimal Price: ₹{round(predicted_price, 2)}"
    except Exception as e:
        return f"Error: {str(e)}"

# Preprocessing for product recommendation
categorical_features_recommendation = ['product_condition', 'category']
numeric_features_recommendation = ['price']

preprocessor_recommendation = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features_recommendation),
        ('num', 'passthrough', numeric_features_recommendation)
    ])

product_features = preprocessor_recommendation.fit_transform(df_recommendation[['product_condition', 'price', 'category']])

# Fit NearestNeighbors model
knn = NearestNeighbors(n_neighbors=5)
knn.fit(product_features)

def recommend_products(category):
    try:
        filtered_df = df_recommendation[df_recommendation['category'] == category]
        if filtered_df.empty:
            return "No products found in this category."
        random_product = random.choice(filtered_df.index)
        product = product_features[random_product].reshape(1, -1)
        _, indices = knn.kneighbors(product)
        recommended = df_recommendation.iloc[indices[0]]
        recommended = recommended[recommended['category'] == category]
        return recommended[['product_id', 'product_condition', 'price', 'category']]
    except Exception as e:
        return f"Error: {str(e)}"

# Circular Economy Analytics Dashboard
def load_data():
    try:
        return pd.read_csv("synthetic_marketplace_data_2000.csv")
    except:
        return pd.DataFrame({
            "Category": np.random.choice(["Electronics", "Plastic", "Metal", "Wood", "Composite"], 100),
            "LifecycleYears": np.random.uniform(1, 20, 100),
            "Price": np.random.uniform(10, 500, 100),
            "NumReviews": np.random.randint(0, 1000, 100)
        })

def update_live_data():
    df = load_data()
    new_entry = {
        "Category": np.random.choice(["Electronics", "Plastic", "Metal", "Wood", "Composite"]),
        "LifecycleYears": round(np.random.uniform(1, 20), 2),
        "Price": round(np.random.uniform(10, 500), 2),
        "NumReviews": np.random.randint(0, 1000)
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv("synthetic_marketplace_data_2000.csv", index=False)

def generate_dashboard():
    df = load_data()
    lifecycle_fig = px.bar(df.groupby('Category')['LifecycleYears'].mean().reset_index(),
                          x='Category', y='LifecycleYears', title='Average Product Lifecycle by Category')
    price_trend_fig = px.line(df.groupby('Category')['Price'].mean().reset_index(),
                             x='Category', y='Price', title='Average Price Trends by Category')
    engagement_fig = px.bar(df.groupby('Category')['NumReviews'].sum().reset_index(),
                           x='Category', y='NumReviews', title='Total User Reviews per Category')
    df['Sustainability Score'] = np.random.uniform(0, 100, len(df))
    sustainability_fig = px.scatter(df, x='Price', y='Sustainability Score', color='Category',
                                  title='Sustainability Score vs. Product Price')
    return lifecycle_fig, price_trend_fig, engagement_fig, sustainability_fig

# Gradio Application
with gr.Blocks(title="Circular Economy Marketplace") as app:
    # Add a logo or banner image
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>♻️ Circular Economy Marketplace</h1>
        <p>Sustainable product lifecycle management and recommendations</p>
    </div>
    """)
    
    # Login/Register Tab
    with gr.Tab("🔐 Login/Register"):
        with gr.Tab("Register"):
            reg_username = gr.Textbox(label="Username")
            reg_password = gr.Textbox(label="Password", type="password")
            reg_btn = gr.Button("Register")
            reg_output = gr.Textbox()
            reg_btn.click(register, inputs=[reg_username, reg_password], outputs=reg_output)
        
        with gr.Tab("Login"):
            log_username = gr.Textbox(label="Username")
            log_password = gr.Textbox(label="Password", type="password")
            log_btn = gr.Button("Login")
            log_output = gr.Textbox()
            log_btn.click(login, inputs=[log_username, log_password], outputs=log_output)

    # Product Lifecycle Prediction Tab
    with gr.Tab("📈 Product Lifecycle"):
        gr.Markdown("### Predict the lifecycle of your product")
        with gr.Row():
            with gr.Column():
                lifecycle_inputs = [
                    gr.Dropdown(["Plastic", "Metal", "Wood", "Composite", "Electronics"], label="Category"),
                    gr.Textbox(label="Product Name"),
                    gr.Number(label="Price"),
                    gr.Slider(1, 5, label="Rating", step=0.1),
                    gr.Number(label="Number of Reviews"),
                    gr.Number(label="Stock Quantity"),
                    gr.Slider(0, 50, label="Discount (%)", step=1)
                ]
            with gr.Column():
                lifecycle_output = gr.Textbox(label="Prediction", interactive=False)
                lifecycle_btn = gr.Button("Predict Lifecycle", variant="primary")
        
        lifecycle_btn.click(predict_lifecycle, inputs=lifecycle_inputs, outputs=lifecycle_output)

    # Dynamic Pricing Tab
    with gr.Tab("💰 Dynamic Pricing"):
        gr.Markdown("### Get optimal pricing recommendations")
        with gr.Row():
            with gr.Column():
                pricing_inputs = [
                    gr.Dropdown(["iPhone 13", "Nike Shoes", "Samsung TV", "Adidas Jacket", "Dell Laptop", 
                               "Sony Headphones", "Apple Watch", "LG Refrigerator", "HP Printer", "Bose Speaker"], 
                              label="Product Name"),
                    gr.Dropdown(["Electronics", "Fashion", "Home Appliances"], label="Category"),
                    gr.Number(label="Base Price"),
                    gr.Number(label="Competitor Price"),
                    gr.Dropdown(["Low", "Medium", "High"], label="Demand Level"),
                    gr.Number(label="Stock Available"),
                    gr.Number(label="Number of Reviews"),
                    gr.Slider(1, 5, label="Rating", step=0.1),
                    gr.Dropdown(["Holiday", "Summer", "Winter", "Off-season"], label="Season"),
                    gr.Slider(0, 30, label="Discount (%)", step=1)
                ]
            with gr.Column():
                pricing_output = gr.Textbox(label="Optimal Price", interactive=False)
                pricing_btn = gr.Button("Calculate Optimal Price", variant="primary")
        
        pricing_btn.click(predict_price, inputs=pricing_inputs, outputs=pricing_output)

    # Product Recommendation Tab
    with gr.Tab("🛍️ Product Recommendations"):
        gr.Markdown("### Discover similar products in our marketplace")
        with gr.Row():
            with gr.Column():
                recommendation_input = gr.Dropdown(
                    choices=df_recommendation['category'].unique().tolist(), 
                    label="Select Product Category"
                )
                recommendation_btn = gr.Button("Find Recommendations", variant="primary")
            with gr.Column():
                recommendation_output = gr.Dataframe(
                    headers=["ID", "Condition", "Price", "Category"],
                    datatype=["str", "str", "number", "str"],
                    interactive=False
                )
        
        recommendation_btn.click(recommend_products, inputs=recommendation_input, outputs=recommendation_output)

    # Circular Economy Analytics Tab
    with gr.Tab("📊 Marketplace Analytics"):
        gr.Markdown("### Real-time marketplace insights and trends")
        with gr.Row():
            dashboard_btn = gr.Button("Refresh Dashboard", variant="primary")
        with gr.Row():
            dashboard_outputs = [
                gr.Plot(label="Product Lifecycle Analytics"),
                gr.Plot(label="Dynamic Pricing Insights")
            ]
        with gr.Row():
            dashboard_outputs.extend([
                gr.Plot(label="User Engagement Trends"),
                gr.Plot(label="Sustainability Insights")
            ])
        
        dashboard_btn.click(generate_dashboard, inputs=[], outputs=dashboard_outputs)

    # AI Chatbot Tab
    with gr.Tab("💬 AI Assistant"):
        gr.Markdown("### Ask our AI assistant anything about circular economy")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_input = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about product lifecycle, pricing, recommendations..."
                )
                chatbot_btn = gr.Button("Ask", variant="primary")
            with gr.Column(scale=4):
                chatbot_output = gr.Textbox(label="AI Response", interactive=False)
        
        chatbot_btn.click(huggingface_chatbot, inputs=chatbot_input, outputs=chatbot_output)

    # Feedback Tab
    with gr.Tab("📝 Feedback"):
        gr.Markdown("## We value your feedback!")
        gr.Markdown("Help us improve by sharing your experience with our platform.")
        
        with gr.Row():
            with gr.Column():
                satisfaction = gr.Radio(
                    ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"], 
                    label="1. Overall Satisfaction"
                )
                features = gr.CheckboxGroup(
                    ["Lifecycle Prediction", "Dynamic Pricing", "Product Recommendation", 
                     "Marketplace Analytics", "AI Assistant", "Other"],
                    label="2. Most Useful Features"
                )
                issues = gr.Textbox(
                    label="3. Issues Faced (if any)", 
                    placeholder="Describe any problems you encountered..."
                )
            with gr.Column():
                suggestions = gr.Textbox(
                    label="4. Suggestions for Improvement", 
                    placeholder="How can we make this platform better?"
                )
                recommendation = gr.Radio(
                    ["Yes", "No", "Maybe"], 
                    label="5. Would you recommend this platform to others?"
                )
                name = gr.Textbox(label="6. Your Name (Optional)")
                email = gr.Textbox(label="7. Email (Optional)")
        
        submit_btn = gr.Button("Submit Feedback", variant="primary")
        feedback_output = gr.Textbox(label="Submission Status", interactive=False)
        
        submit_btn.click(
            fn=submit_feedback,
            inputs=[satisfaction, features, issues, suggestions, recommendation, name, email],
            outputs=feedback_output
        )

# Simulate real-time data updates
def live_update():
    while True:
        update_live_data()
        time.sleep(5)

threading.Thread(target=live_update, daemon=True).start()

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)
