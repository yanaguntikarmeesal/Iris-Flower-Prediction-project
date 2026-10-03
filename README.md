 

```markdown
# 🌸 Iris Flower Prediction - ML Web App

# link to open = 
https://iris-flower-prediction-project-rybnfvbzjyfud3zvk6iv2b.streamlit.app/

An elegant and interactive Machine Learning web application built with **Streamlit** to predict the species of an Iris flower based on its measurements. 

> Built with a clean UI, custom styling, and a trained scikit-learn model with 6 engineered features.

 

## 📸 Preview
A modern, responsive UI featuring:
- Gradient hero section
- Sidebar number inputs for measurements
- Real-time prediction with confidence score
- Probability distribution table
- Dataset preview & Feature Importance

## ✨ Features

- **Accurate Prediction:** Uses a trained ML model (`iris_model.pkl`) to classify into Setosa, Versicolor, or Virginica
- **Feature Engineering:** Uses 6 features - 4 original + 2 engineered (`sepal_area`, `petal_area`)
- **Interactive UI:** Beautiful custom CSS with Poppins font, gradient theme, and responsive layout
- **Confidence Metrics:** Shows prediction probability distribution
- **Data Insights:** Built-in dataset preview and model feature importance visualization
- **Optimized Performance:** Uses `@st.cache_resource` for fast model loading

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Custom CSS, HTML
- **Backend / ML:** Python, NumPy, Pandas, Scikit-learn, Joblib
- **Dataset:** Iris Dataset from `sklearn.datasets`

## 📁 Project Structure

```
iris-flower-prediction/
│
├── app.py                 # Main Streamlit application
├── iris_model.pkl         # Trained classification model
├── scaler.pkl             # StandardScaler for feature scaling
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/iris-flower-prediction.git
cd iris-flower-prediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
streamlit
numpy
pandas
scikit-learn
joblib
```

### 4. Run the App
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

## 🧠 Model Information

The model is trained on 6 features to improve accuracy:

| Feature | Description |
| :--- | :--- |
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |
| `sepal_area` | `sepal_length * sepal_width` (Engineered) |
| `petal_area` | `petal_length * petal_width` (Engineered) |

All features are scaled using `StandardScaler` before prediction.

**Training snippet (if you need to re-train):**
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

iris = load_iris()
# Add engineered features and train...
# scaler.fit(X_train)
# model.fit(X_train_scaled, y_train)
# joblib.dump(model, "iris_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
```

## 💻 How to Use

1.  Enter the 4 flower measurements in the sidebar.
2.  Click on **🔮 Predict Species**.
3.  View the predicted species, confidence score, and probability distribution.

## 📊 Dataset

- **Source:** Fisher's Iris Dataset
- **Samples:** 150
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Features:** 4 original measurements

## 🌐 Deployment

You can deploy this app easily on:
- **Streamlit Community Cloud**
 

## 👨‍💻 Author

**Yanaguntikar Meesal**

- Email: yanaguntikarm@gmail.com
- GitHub: [@yanaguntikar](https://github.com/yanaguntikar)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<p align="center">Built with ❤ using Streamlit & Scikit-learn</p>
```

 
3. Badges with your actual GitHub username

Just tell me your GitHub username and I'll update the README links.
