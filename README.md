# 🎓 Online Course Completion Predictor

An AI-powered machine learning application that predicts whether a student will complete an online course (MOOC) based on their engagement metrics, demographics, and learning behavior patterns.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project addresses the critical challenge of student dropout rates in online learning platforms. By analyzing various engagement metrics and demographic factors, the model predicts course completion probability and provides actionable recommendations to improve student retention.

### Problem Statement

Online courses face high dropout rates (often 85-90%). Early identification of at-risk students enables timely interventions to improve completion rates.

### Solution

A Random Forest machine learning model that:
- Predicts completion probability with 85-90% accuracy
- Identifies key factors influencing course completion
- Provides personalized recommendations for struggling students
- Enables batch prediction for entire cohorts

## ✨ Features

### 🔮 Prediction Capabilities
- **Single Student Prediction**: Interactive form for individual predictions
- **Batch Prediction**: Upload CSV files for cohort-level analysis
- **Real-time Results**: Instant predictions with probability scores
- **Visual Analytics**: Interactive gauges and charts

### 📊 Analysis Features
- Comprehensive exploratory data analysis
- Feature importance visualization
- Model performance comparison
- ROC curve analysis
- Correlation heatmaps

### 💡 Intelligent Recommendations
- Personalized suggestions based on engagement gaps
- Specific action items for improvement
- Risk factor identification
- Success pattern recognition

### 🎨 User Interface
- Clean, professional Streamlit interface
- Interactive visualizations using Plotly
- Responsive design
- Easy-to-use navigation

## 🛠️ Tech Stack

**Machine Learning:**
- scikit-learn (Random Forest, Logistic Regression, Gradient Boosting)
- pandas & NumPy for data manipulation
- Model persistence with pickle

**Visualization:**
- Matplotlib & Seaborn for static plots
- Plotly for interactive visualizations
- Streamlit for web interface

**Development:**
- Jupyter Notebook for experimentation
- Python 3.8+

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/abhigyaabs2/Online-Course-Completion-Predictor.git
cd Online-Course-Completion-Predictor
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Train the model**
```bash
jupyter notebook
# Open course_completion_analysis.ipynb and run all cells
```

4. **Launch the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🚀 Usage

### Single Prediction

1. Navigate to the **Prediction** tab
2. Enter student demographics (age, gender, education)
3. Input engagement metrics (videos watched, assignments, etc.)
4. Click **Predict Completion**
5. View results, probability gauge, and recommendations

### Batch Prediction

1. Navigate to the **Batch Prediction** tab
2. Download the sample CSV template
3. Fill in student data following the template format
4. Upload your CSV file
5. Click **Predict All**
6. Download results with predictions

### Model Training

Open `course_completion_analysis.ipynb` in Jupyter to:
- Explore the dataset
- Perform feature engineering
- Train and compare multiple models
- Evaluate model performance
- Save the best model

## 📁 Project Structure

```
online-course-predictor/
│
├── course_completion_analysis.ipynb  # ML model development
├── course.py                            # Streamlit web application
├── README.md                         # Project documentation
│
│   course_data.csv              # Training dataset
│
├
│   course_completion_model.pkl  # Trained model
│
└── screenshots/                      # Application screenshots
    ├── prediction.png
    ├── batch.png
    └── dashboard.png
```

## 📊 Model Performance

### Performance Metrics

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Random Forest | 89.2% | 0.94 | 0.87 | 0.91 |
| Gradient Boosting | 87.5% | 0.92 | 0.85 | 0.89 |
| Logistic Regression | 84.1% | 0.88 | 0.82 | 0.86 |

### Top Features (by importance)

1. **Assignments Completed** (23.5%)
2. **Quiz Scores Average** (18.2%)
3. **Days Active** (15.8%)
4. **Videos Watched** (13.4%)
5. **Time Spent Hours** (11.3%)

## 📊 Dataset

### Features Used

**Demographics:**
- Age
- Gender
- Education Level
- Previous Courses Completed

**Engagement Metrics:**
- Videos Watched
- Assignments Completed
- Forum Posts
- Login Frequency
- Days Active

**Performance Indicators:**
- Average Quiz Scores
- Total Time Spent
- Certification Goal

**Engineered Features:**
- Engagement Rate
- Assignment Completion Rate
- Average Time per Day
- Interaction Score

### Data Sources

This project uses a synthetic dataset for demonstration. For real-world applications, you can use:

- [Kaggle MOOC Datasets](https://www.kaggle.com/search?q=mooc)
- [Open University Learning Analytics](https://analyse.kmi.open.ac.uk/open_dataset)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM, Transformers)
- [ ] Real-time monitoring dashboard
- [ ] Email alerts for at-risk students
- [ ] A/B testing framework for interventions
- [ ] Mobile app version
- [ ] Multi-language support
- [ ] Integration with LMS platforms
- [ ] Advanced feature engineering (NLP on forum posts)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn documentation and community
- Streamlit for the amazing framework
- MOOC platforms for inspiring this project
- All contributors and supporters

---

⭐ If you found this project helpful, please give it a star!

📫 For questions or feedback, feel free to reach out or open an issue.

**Built with ❤️ using Python, scikit-learn, and Streamlit**
