# 🚀 Deployment Guide

This guide will help you deploy your ODI Win Probability Predictor to Streamlit Cloud and prepare it for GitHub.

## 📦 Pre-Deployment Checklist

Before deploying, ensure you have:

- [x] Trained the model (`worldcup_win_predictor.pkl`)
- [x] Generated team features (`team_features.csv`)
- [x] All required files in the repository
- [x] Updated `requirements.txt`
- [x] Created `.gitignore`
- [x] Written `README.md`

## 🌐 Option 1: Deploy to Streamlit Cloud (Recommended)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done)
   ```bash
   cd "/Applications/PROBABILITY WIN MDOEL 2"
   git init
   ```

2. **Add all files**
   ```bash
   git add .
   ```

3. **Commit your changes**
   ```bash
   git commit -m "Initial commit: ODI Win Probability Predictor"
   ```

### Step 2: Push to GitHub

1. **Create a new repository on GitHub**
   - Go to https://github.com/new
   - Name it: `odi-win-probability-predictor`
   - Don't initialize with README (you already have one)
   - Click "Create repository"

2. **Link your local repo to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/odi-win-probability-predictor.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository: `odi-win-probability-predictor`
   - Set branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment**
   - Streamlit will install dependencies
   - Your app will be live at: `https://YOUR_USERNAME-odi-win-probability-predictor.streamlit.app`

### Step 4: Share Your App

Once deployed, you'll get a public URL like:
```
https://your-app-name.streamlit.app
```

Update your `README.md` with this URL!

## 🐳 Option 2: Deploy with Docker (Advanced)

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t odi-predictor .
docker run -p 8501:8501 odi-predictor
```

## 🖥️ Option 3: Deploy to Heroku

### Create Procfile

```
web: sh setup.sh && streamlit run app.py
```

### Create setup.sh

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Deploy

```bash
heroku create your-app-name
git push heroku main
```

## 📊 Important Files for Deployment

### Required Files
✅ `app.py` - Main Streamlit application  
✅ `requirements.txt` - Python dependencies  
✅ `worldcup_win_predictor.pkl` - Trained model  
✅ `team_features.csv` - Team statistics  

### Optional but Recommended
✅ `README.md` - Project documentation  
✅ `.gitignore` - Exclude unnecessary files  
✅ `LICENSE` - Open source license  
✅ `.streamlit/config.toml` - Streamlit configuration  

### Not Required for Deployment
❌ `train_predictor.py` - Training script (run locally)  
❌ `analyez.py` - Analysis utilities (optional)  
❌ `odis_male_json/` - Raw data (too large, train locally)  
❌ `test_predictions.csv` - Test results (optional)  
❌ PNG visualization files (optional)  

## 🔧 Troubleshooting

### Issue: App crashes on startup

**Solution:** Check that all required files are present:
```bash
ls -la worldcup_win_predictor.pkl
ls -la team_features.csv
```

### Issue: Module not found errors

**Solution:** Verify `requirements.txt` has all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Model file too large for GitHub

**Solution:** Use Git LFS (Large File Storage):
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add worldcup_win_predictor.pkl
git commit -m "Add model with LFS"
git push
```

### Issue: Streamlit Cloud build fails

**Solution:** Check Python version compatibility:
- Streamlit Cloud uses Python 3.9 by default
- Ensure your dependencies support Python 3.9

## 📝 Post-Deployment

### Update README with Live URL

```markdown
## 🌟 Live Demo

**Try it now:** [ODI Win Predictor](https://your-app-url.streamlit.app)
```

### Monitor Your App

- Check Streamlit Cloud dashboard for logs
- Monitor resource usage
- Track visitor analytics (if enabled)

### Share Your Project

- Tweet about it with #Streamlit
- Post on LinkedIn
- Share in cricket analytics communities
- Add to your portfolio

## 🎉 Success!

Your ODI Win Probability Predictor is now live and accessible to the world!

---

**Need Help?**
- Streamlit Docs: https://docs.streamlit.io
- Streamlit Community: https://discuss.streamlit.io
- GitHub Issues: Create an issue in your repository
