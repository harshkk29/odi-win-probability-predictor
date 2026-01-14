# 🏆 ODI Cricket Win Probability Predictor

A machine learning-powered web application that predicts the win probability of One Day International (ODI) cricket matches in real-time using 15 advanced features including team statistics, player strength, ELO ratings, and live match conditions.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🌟 Features

### Advanced ML Model
- **15 Feature Logistic Regression Model** trained on historical ODI match data
- Ball-by-ball prediction capability
- Real-time probability updates based on match situation

### Key Prediction Features
**Live Match Features:**
- Current Score & Wickets Fallen
- Overs Completed
- Run Rates (Current vs Required)
- Runs Required & Balls Remaining

**Team Statistics (Historical):**
- Historical Win Ratio
- Recent Form (Last 5 Matches)
- Player Strength Index
- Head-to-Head Record
- ELO Rating Difference

### Interactive Visualizations
- 📊 **Probability Gauges** - Real-time win probability for both teams
- 📈 **Run Rate Comparison** - Current vs Required run rate
- 🎯 **Team Statistics Radar Chart** - Comparative team strengths
- 💡 **Match Insights** - AI-generated match analysis

## 🚀 Live Demo

**Deploy on Streamlit Cloud:** [Your App URL Here]

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- ODI match data in JSON format (for training)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/odi-win-predictor.git
cd odi-win-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
Place your ODI match JSON files in a folder (e.g., `odis_male_json/`)

### 4. Train the Model
```bash
python train_predictor.py
```
When prompted, enter the path to your JSON match folder.

This will generate:
- `worldcup_win_predictor.pkl` - Trained model
- `team_features.csv` - Team statistics
- `test_predictions.csv` - Test set predictions
- `merged_matches.csv` - Match metadata

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Project Structure

```
odi-win-predictor/
│
├── app.py                          # Streamlit web application
├── train_predictor.py              # Model training script
├── analyez.py                      # Data analysis utilities
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── worldcup_win_predictor.pkl      # Trained ML model (generated)
├── team_features.csv               # Team statistics (generated)
├── test_predictions.csv            # Model predictions (generated)
├── merged_matches.csv              # Match metadata (generated)
│
├── odis_male_json/                 # Match data folder (JSON files)
│   ├── match1.json
│   ├── match2.json
│   └── ...
│
└── visualizations/                 # Generated plots
    ├── 01_confusion_matrix_heatmap.png
    ├── 02_roc_curve.png
    ├── 03_correlation_heatmap.png
    └── ...
```

## 🎯 How to Use

### Step 1: Configure Match Details
In the sidebar, select:
- **Batting Team** - Team chasing the target
- **Bowling Team** - Team defending the target
- **Target Score** - Runs to win
- **Current Score** - Current runs scored
- **Wickets Fallen** - Wickets lost
- **Overs Completed** - Overs bowled

### Step 2: View Team Statistics
The app automatically loads:
- Historical win percentages
- Recent form
- Player strength
- Head-to-head records
- ELO ratings

### Step 3: Predict
Click **"🔮 PREDICT WIN PROBABILITY"** to get:
- Win probability for both teams
- Match situation analysis
- Run rate comparison
- Detailed insights and recommendations

## 🧠 Model Details

### Algorithm
**Logistic Regression** with StandardScaler normalization

### Training Process
1. **Data Loading** - Parse JSON match files
2. **Feature Engineering** - Extract 15 advanced features
3. **Chronological Processing** - Maintain temporal order for historical stats
4. **Train-Test Split** - 80-20 split by matches
5. **Model Training** - Logistic Regression with max_iter=1000
6. **Evaluation** - Accuracy, Precision, Recall, F1-Score

### Feature List
```python
features = [
    'current_score', 'wickets_fallen', 'overs_completed',
    'runs_required', 'balls_remaining',
    'current_run_rate', 'required_run_rate',
    'batting_historical_win_ratio', 'bowling_historical_win_ratio',
    'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
    'elo_difference',
    'batting_player_strength', 'bowling_player_strength'
]
```

## 📈 Model Performance

The model is evaluated on:
- **Accuracy** - Overall prediction correctness
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision and recall

Results are displayed after training and saved in `test_predictions.csv`

## 🌐 Deployment on Streamlit Cloud

### Quick Deploy Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Required Files for Deployment**
   - `app.py` ✅
   - `requirements.txt` ✅
   - `worldcup_win_predictor.pkl` ✅
   - `team_features.csv` ✅

**Note:** Make sure to train the model locally first and commit the generated `.pkl` and `.csv` files to GitHub.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Your Name**
Suchir Dharmadhikari - Data collection, Variable refinement
Aarushi Malani - Variable refinement, Statistical interpretation
Harshvardhan Khot - Data processing, Modeling 
Akshat Kumar - Data analysis, Literature review
Sarah Poonattu - Data analysis, Study design


## 🙏 Acknowledgments

- Cricket match data from [Cricsheet](https://cricsheet.org/)
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)

## 📧 Contact

For questions or feedback, please open an issue or contact harshvardhankhot12@gmail.com

---

⭐ **Star this repo** if you find it helpful!
