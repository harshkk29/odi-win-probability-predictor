# 🎯 Quick Setup Instructions

Follow these steps to get your ODI Win Probability Predictor ready for GitHub and Streamlit Cloud.

## ✅ Step 1: Verify Your Files

Check that you have all required files:

```bash
cd "/Applications/PROBABILITY WIN MDOEL 2"
ls -la
```

You should see:
- ✅ `app.py` - Streamlit application
- ✅ `train_predictor.py` - Model training script
- ✅ `requirements.txt` - Dependencies
- ✅ `worldcup_win_predictor.pkl` - Trained model
- ✅ `team_features.csv` - Team statistics
- ✅ `odis_male_json/` - Your match data folder
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules

## 🚀 Step 2: Initialize Git Repository

```bash
# Navigate to your project folder
cd "/Applications/PROBABILITY WIN MDOEL 2"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Make your first commit
git commit -m "Initial commit: ODI Win Probability Predictor"
```

## 📤 Step 3: Push to GitHub

### Option A: Using GitHub Desktop (Easiest)
1. Download GitHub Desktop: https://desktop.github.com
2. Open GitHub Desktop
3. Click "Add Local Repository"
4. Select your project folder
5. Click "Publish repository"
6. Uncheck "Keep this code private" if you want it public
7. Click "Publish Repository"

### Option B: Using Command Line
```bash
# Create a new repository on GitHub first at: https://github.com/new
# Then run these commands:

git remote add origin https://github.com/YOUR_USERNAME/odi-win-predictor.git
git branch -M main
git push -u origin main
```

## ☁️ Step 4: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app" button
   - Repository: Select your `odi-win-predictor` repo
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Wait for Deployment** (2-5 minutes)
   - Streamlit will install dependencies
   - Your app will be live!

4. **Get Your URL**
   - You'll get a URL like: `https://your-app.streamlit.app`
   - Share it with the world! 🎉

## ⚠️ Important Notes

### About the Data Folder (`odis_male_json/`)

The `odis_male_json/` folder contains your raw match data. This folder is:
- ✅ **Needed for training** the model locally
- ❌ **NOT needed for deployment** (the trained model `.pkl` file is enough)
- 📦 **May be too large** for GitHub (2500+ files)

**Recommendation:** 
- The `.gitignore` file is already configured to exclude large data folders
- If you want to include it, you can, but it will make your repo larger
- For deployment, only `worldcup_win_predictor.pkl` and `team_features.csv` are needed

### If GitHub Upload Fails Due to File Size

If you get errors about file size limits:

```bash
# Option 1: Don't commit the data folder (recommended)
# Already handled by .gitignore

# Option 2: Use Git LFS for large files
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add LFS tracking"
git push
```

## 🧪 Step 5: Test Your Deployed App

Once deployed:
1. Open your Streamlit Cloud URL
2. Select teams from the sidebar
3. Configure match details
4. Click "PREDICT WIN PROBABILITY"
5. Verify predictions work correctly

## 🎨 Step 6: Customize (Optional)

Update these in your files:
- `README.md` - Add your name, GitHub username, live app URL
- `LICENSE` - Add your name
- `app.py` - Customize colors, team names, etc.

## 📊 File Size Reference

Typical file sizes:
- `app.py` - ~18 KB ✅
- `train_predictor.py` - ~28 KB ✅
- `worldcup_win_predictor.pkl` - ~2 KB ✅
- `team_features.csv` - ~332 KB ✅
- `odis_male_json/` - Large (2500+ files) ⚠️

**Total without data folder:** < 1 MB ✅  
**Total with data folder:** Varies (could be 100+ MB) ⚠️

## 🆘 Troubleshooting

### Problem: Git is not installed
**Solution:** Download from https://git-scm.com/downloads

### Problem: Model file not found in Streamlit Cloud
**Solution:** Make sure `worldcup_win_predictor.pkl` is committed to GitHub:
```bash
git add worldcup_win_predictor.pkl
git commit -m "Add trained model"
git push
```

### Problem: App crashes with "No module named 'streamlit'"
**Solution:** Check `requirements.txt` is in the root folder and properly formatted

### Problem: Repository too large for GitHub
**Solution:** Exclude the `odis_male_json/` folder:
```bash
# Add to .gitignore (already done)
echo "odis_male_json/" >> .gitignore
git rm -r --cached odis_male_json/
git commit -m "Remove large data folder"
git push
```

## ✨ Success Checklist

- [ ] Git repository initialized
- [ ] All files committed
- [ ] Pushed to GitHub
- [ ] Deployed to Streamlit Cloud
- [ ] App is accessible via public URL
- [ ] Predictions working correctly
- [ ] README updated with live URL

## 🎉 You're Done!

Your ODI Win Probability Predictor is now live and ready to share!

**Next Steps:**
- Share your app URL on social media
- Add it to your portfolio
- Get feedback from cricket fans
- Iterate and improve!

---

**Need Help?** Check `DEPLOYMENT.md` for detailed deployment options.
