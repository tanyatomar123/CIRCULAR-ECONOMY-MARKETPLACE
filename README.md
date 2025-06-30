# 🌿 Circular Economy Marketplace Platform (CEMP)

An AI-powered Gradio app that brings sustainability into e-commerce using machine learning and NLP.


## 📌 Overview

CEMP helps users make eco-friendly decisions by predicting product lifespan, recommending prices, suggesting sustainable alternatives, and offering real-time dashboard analytics—all integrated into one seamless web app.


## 💡 Features

- 🔍 Product Lifecycle Prediction (Random Forest)
- 💸 Dynamic Pricing Engine (Gradient Boosting)
- 🤝 Smart Recommendations (KNN)
- 📊 Circular Analytics Dashboard (Plotly)
- 🤖 AI Chatbot (Hugging Face – FLAN-T5)
- 🔐 Secure Login (SQLite + bcrypt)
- 📬 Feedback Collection Form


## 🛠 Tech Stack

**Frontend/UI**: Gradio  
**ML Models**: Scikit-learn  
**Visualization**: Plotly  
**NLP**: Hugging Face Transformers  
**Database**: SQLite  
**Deployment**: Hugging Face Spaces  
**Dev Env**: Google Colab + GitHub


🚀 Live Demo
https://huggingface.co/spaces/tanya17/marketplace

🧠 Architecture (Flow)
mermaid
Copy
Edit
graph LR
User --> Gradio
Gradio --> MLModels
Gradio --> Chatbot(Hugging Face API)
Gradio --> Dashboard(Plotly)
Gradio --> Auth(SQLite)

📎 Usage Instructions
Login/Register to create your personalized session.

Navigate through the tabs:

Lifecycle Prediction: Fill in product details to get lifespan predictions.

Dynamic Pricing: Enter base price, demand, stock, and competitor info.

Recommendations: Choose a category to discover similar items.

Dashboard: View real-time trends and analytics.

Chatbot: Ask questions like “How can I reuse old electronics?”

Feedback: Submit suggestions, bugs, or improvements.


📷 Screenshots

![image](https://github.com/user-attachments/assets/a7eb4b02-7a75-46cd-9320-35e3eb339055)

![image](https://github.com/user-attachments/assets/f96d0e32-15f7-4519-a400-9ada2063436a)

![image](https://github.com/user-attachments/assets/9a6ffba2-d42e-4d44-b79e-fbc465443f14)

![image](https://github.com/user-attachments/assets/82861eee-8de6-48b1-aab1-18bfbf92e247)

![image](https://github.com/user-attachments/assets/cd874776-c492-45e3-943d-f477e254ebc4)



👥 Credits
Developer: Tanya Tomar

Tools: Gradio, Hugging Face, scikit-learn, Plotly




