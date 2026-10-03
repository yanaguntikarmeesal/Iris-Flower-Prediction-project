## 📁 Project Structure

iris-flower-prediction/
├── app.py                # Streamlit app (UI + prediction logic)
├── model.pkl        # Trained scikit-learn model
├── requirements.txt      # Python dependencies
└── README.md

## 🚀 Getting Started

### 1. Clone the repository
git clone https://github.com/<your-username>/iris-flower-prediction.git
cd iris-flower-prediction

### 2. Install dependencies
pip install -r requirements.txt

### 3. (Optional) Retrain the model
python train_model.py

### 4. Run the app
streamlit run app.py

The app will open at http://localhost:8501.

## 🧠 How It Works

1. **Input:** Enter sepal length/width and petal length/width in the sidebar.
2. **Feature engineering:** Two extra features are computed:
   - `sepal_area = sepal_length × sepal_width`
   - `petal_area = petal_length × petal_width`
3. **Prediction:** The 6-feature vector goes to the trained model, which returns the species and class probabilities.
4. **Output:** The app shows the predicted species, a confidence score, and a probability table.

## 🌼 Classes

| Species | Label |
|---------|-------|
| Setosa | 0 |
| Versicolor | 1 |
| Virginica | 2 |

## 📦 Requirements

streamlit
numpy
pandas
scikit-learn
joblib

## 🔮 Future Improvements

- Add input validation with realistic min/max ranges
- Show interactive plots (Plotly) of the input against the dataset
- Compare multiple models (Random Forest, SVM, KNN)
- Add Docker support for deployment

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)
