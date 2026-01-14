#!/bin/bash

# ODI Win Predictor - GitHub Upload Script
# This script helps you upload your project to GitHub

echo "🏆 ODI Win Probability Predictor - GitHub Upload Script"
echo "========================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project directory."
    exit 1
fi

echo "✅ Found project files"
echo ""

# Ask user about data folder
echo "📁 Data Folder Options:"
echo "Your project contains the 'odis_male_json/' folder with match data."
echo ""
echo "Choose an option:"
echo "  1) Include data folder (may be large, slower upload)"
echo "  2) Exclude data folder (recommended - faster upload, model already trained)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "2" ]; then
    echo ""
    echo "📝 Excluding data folder from Git..."
    # Uncomment the data folder line in .gitignore
    sed -i.bak 's/# odis_male_json\//odis_male_json\//' .gitignore
    echo "✅ Updated .gitignore to exclude odis_male_json/"
else
    echo ""
    echo "📦 Including data folder (this may take longer)..."
fi

echo ""
echo "🔧 Initializing Git repository..."
git init

echo ""
echo "📋 Adding files to Git..."
git add .

echo ""
echo "💬 Creating initial commit..."
git commit -m "Initial commit: ODI Win Probability Predictor"

echo ""
echo "✅ Git repository ready!"
echo ""
echo "📤 Next Steps:"
echo "1. Create a new repository on GitHub: https://github.com/new"
echo "2. Name it: odi-win-probability-predictor"
echo "3. Don't initialize with README (you already have one)"
echo "4. Run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/odi-win-probability-predictor.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "5. Then deploy to Streamlit Cloud: https://share.streamlit.io"
echo ""
echo "🎉 Done! Check SETUP_INSTRUCTIONS.md for detailed deployment steps."
