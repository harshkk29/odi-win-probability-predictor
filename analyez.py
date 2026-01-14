import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

def load_predictions(filename='test_predictions.csv'):
    """Load predictions from CSV"""
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found!")
        print("   Please run the training script first to generate predictions.")
        return None
    
    df = pd.read_csv(filename)
    print(f"✓ Loaded {len(df)} predictions from {len(df['match_id'].unique())} matches")
    
    # Check if dataframe has required columns
    required_cols = ['match_id', 'batting_team', 'bowling_team', 'won', 'predicted_winner']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    return df

def load_merged_matches(filename='merged_matches.csv'):
    """Load merged match information"""
    if not os.path.exists(filename):
        print(f"⚠️  Warning: {filename} not found!")
        return None
    
    df = pd.read_csv(filename)
    print(f"✓ Loaded merged information for {len(df)} matches")
    return df

def check_advanced_features(df):
    """Check if dataframe has advanced features"""
    advanced_features = [
        'batting_historical_win_ratio', 'bowling_historical_win_ratio',
        'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
        'elo_difference', 'batting_player_strength', 'bowling_player_strength'
    ]
    
    missing = [f for f in advanced_features if f not in df.columns]
    
    if missing:
        print(f"⚠️  Missing advanced features: {missing}")
        print("   Using basic features only for analysis")
        return False
    return True

def plot_confusion_matrix_heatmap(df):
    """Create detailed confusion matrix heatmap"""
    final_predictions = df.groupby('match_id').last().reset_index()
    
    y_true = final_predictions['won']
    y_pred = final_predictions['predicted_winner']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap 1: Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
                xticklabels=['Lost', 'Won'], yticklabels=['Lost', 'Won'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_title('Confusion Matrix - Absolute Counts', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual Outcome', fontsize=12)
    axes[0].set_xlabel('Predicted Outcome', fontsize=12)
    
    # Heatmap 2: Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', cbar=True, ax=axes[1],
                xticklabels=['Lost', 'Won'], yticklabels=['Lost', 'Won'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[1].set_title('Confusion Matrix - Normalized (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual Outcome', fontsize=12)
    axes[1].set_xlabel('Predicted Outcome', fontsize=12)
    
    fig.suptitle('Confusion Matrix Heatmaps - Match Outcome Predictions', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('01_confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '01_confusion_matrix_heatmap.png'")
    plt.close()
    
    # Classification report
    print("\n" + "="*100)
    print("CLASSIFICATION REPORT")
    print("="*100)
    print(classification_report(y_true, y_pred, target_names=['Lost', 'Won']))
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("="*100)

def plot_roc_curve(df):
    """Plot ROC curve and calculate AUC"""
    final_predictions = df.groupby('match_id').last().reset_index()
    
    y_true = final_predictions['won']
    y_proba = final_predictions['predicted_win_probability'] / 100
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Model Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    
    plt.tight_layout()
    plt.savefig('02_roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '02_roc_curve.png'")
    plt.close()
    
    print(f"\n✓ ROC AUC Score: {roc_auc:.4f}")

def plot_feature_correlation_heatmap(df, has_advanced=True):
    """Create correlation heatmap for all features"""
    
    if has_advanced:
        features_to_plot = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'balls_remaining', 'current_run_rate', 'required_run_rate',
            'batting_historical_win_ratio', 'bowling_historical_win_ratio',
            'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
            'elo_difference', 'batting_player_strength', 'bowling_player_strength', 'won'
        ]
    else:
        features_to_plot = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'balls_remaining', 'current_run_rate', 'required_run_rate', 'won'
        ]
    
    # Get unique rows per match (to avoid data duplication bias)
    match_data = df.groupby('match_id').last().reset_index()
    
    correlation_data = match_data[features_to_plot].copy()
    correlation_matrix = correlation_data.corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax, annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '03_correlation_heatmap.png'")
    plt.close()
    
    return correlation_matrix

def plot_pairplot(df, has_advanced=True):
    """Create pairplot for top features"""
    
    # Get unique rows per match
    match_data = df.groupby('match_id').last().reset_index()
    
    if has_advanced:
        top_features = [
            'current_run_rate', 'required_run_rate', 'current_score',
            'batting_recent_form', 'bowling_recent_form', 'elo_difference', 'won'
        ]
    else:
        top_features = [
            'current_run_rate', 'required_run_rate', 'current_score', 'wickets_fallen', 'won'
        ]
    
    plot_data = match_data[top_features].copy()
    plot_data['Result'] = plot_data['won'].map({0: 'Lost', 1: 'Won'})
    
    fig = sns.pairplot(plot_data, hue='Result', diag_kind='hist', 
                       plot_kws={'alpha': 0.6, 's': 50},
                       diag_kws={'bins': 15, 'edgecolor': 'black'})
    
    fig.fig.suptitle('Pairplot - Feature Relationships by Match Outcome', 
                     fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('04_pairplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '04_pairplot.png'")
    plt.close()

def plot_distribution_analysis(df, has_advanced=True):
    """Plot distributions and normality tests for all features"""
    
    # Get unique rows per match
    match_data = df.groupby('match_id').last().reset_index()
    
    if has_advanced:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'current_run_rate', 'required_run_rate',
            'batting_historical_win_ratio', 'bowling_historical_win_ratio',
            'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
            'elo_difference', 'batting_player_strength', 'bowling_player_strength'
        ]
    else:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'current_run_rate', 'required_run_rate'
        ]
    
    # Create subplots
    n_features = len(features_to_analyze)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    normality_results = []
    
    for idx, feature in enumerate(features_to_analyze):
        ax = axes[idx]
        
        # Plot histogram with normal distribution overlay
        data = match_data[feature].dropna()
        ax.hist(data, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit and plot normal distribution
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')
        
        # Shapiro-Wilk normality test
        stat, p_value = stats.shapiro(data)
        is_normal = "✓ Normal" if p_value > 0.05 else "✗ Not Normal"
        
        normality_results.append({
            'Feature': feature,
            'Statistic': stat,
            'p-value': p_value,
            'Normal': is_normal
        })
        
        ax.set_title(f'{feature}\n{is_normal} (p={p_value:.4f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(len(features_to_analyze), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle('Distribution Analysis - Normality Tests', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('05_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '05_distribution_analysis.png'")
    plt.close()
    
    return pd.DataFrame(normality_results)

def plot_feature_importance_boxplot(df, has_advanced=True):
    """Box plots comparing features by match outcome"""
    
    # Get unique rows per match
    match_data = df.groupby('match_id').last().reset_index()
    
    if has_advanced:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'runs_required',
            'current_run_rate', 'required_run_rate',
            'batting_recent_form', 'bowling_recent_form',
            'elo_difference', 'batting_player_strength', 'bowling_player_strength'
        ]
    else:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'runs_required',
            'current_run_rate', 'required_run_rate'
        ]
    
    n_features = len(features_to_analyze)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_analyze):
        ax = axes[idx]
        
        # Prepare data
        plot_data = match_data[[feature, 'won']].copy()
        plot_data['Outcome'] = plot_data['won'].map({0: 'Lost', 1: 'Won'})
        
        # Create box plot
        sns.boxplot(data=plot_data, x='Outcome', y=feature, ax=ax, 
                   palette=['#E63946', '#06A77D'])
        
        # Add individual points
        sns.stripplot(data=plot_data, x='Outcome', y=feature, ax=ax, 
                     color='black', alpha=0.3, size=4)
        
        # Perform t-test
        won_data = match_data[match_data['won'] == 1][feature]
        lost_data = match_data[match_data['won'] == 0][feature]
        
        t_stat, p_value = stats.ttest_ind(won_data, lost_data)
        sig = "**" if p_value < 0.05 else "ns"
        
        ax.set_title(f'{feature}\np-value: {p_value:.4f} {sig}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove extra subplots
    for idx in range(len(features_to_analyze), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle('Feature Comparison - Won vs Lost Matches (Box Plots)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('06_feature_boxplots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '06_feature_boxplots.png'")
    plt.close()

def plot_feature_statistics_table(df, has_advanced=True):
    """Create comprehensive feature statistics table visualization"""
    
    # Get unique rows per match
    match_data = df.groupby('match_id').last().reset_index()
    
    if has_advanced:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'balls_remaining', 'current_run_rate', 'required_run_rate',
            'batting_historical_win_ratio', 'bowling_historical_win_ratio',
            'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
            'elo_difference', 'batting_player_strength', 'bowling_player_strength'
        ]
    else:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'balls_remaining', 'current_run_rate', 'required_run_rate'
        ]
    
    stats_table = []
    
    for feature in features_to_analyze:
        won_data = match_data[match_data['won'] == 1][feature]
        lost_data = match_data[match_data['won'] == 0][feature]
        
        t_stat, p_value = stats.ttest_ind(won_data, lost_data)
        
        stats_table.append({
            'Feature': feature,
            'Won Mean': won_data.mean(),
            'Lost Mean': lost_data.mean(),
            'Won Std': won_data.std(),
            'Lost Std': lost_data.std(),
            'Difference': won_data.mean() - lost_data.mean(),
            'T-Statistic': t_stat,
            'P-Value': p_value,
            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    
    stats_df = pd.DataFrame(stats_table)
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for _, row in stats_df.iterrows():
        table_data.append([
            row['Feature'],
            f"{row['Won Mean']:.4f}",
            f"{row['Won Std']:.4f}",
            f"{row['Lost Mean']:.4f}",
            f"{row['Lost Std']:.4f}",
            f"{row['Difference']:.4f}",
            f"{row['T-Statistic']:.4f}",
            f"{row['P-Value']:.4f}",
            row['Significant']
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'Won Mean', 'Won Std', 'Lost Mean', 
                              'Lost Std', 'Difference', 'T-Stat', 'P-Value', 'Sig'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color header
    for i in range(9):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(9):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8EDF7')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Feature Statistics - Descriptive Analysis & T-Tests\n(Significance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant)',
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('07_feature_statistics_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '07_feature_statistics_table.png'")
    plt.close()
    
    return stats_df

def plot_probability_calibration(df):
    """Plot predicted probability calibration"""
    
    final_predictions = df.groupby('match_id').last().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calibration plot
    ax = axes[0]
    prob_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_edges = np.linspace(0, 100, 11)
    
    bin_means = []
    actual_means = []
    bin_centers = []
    
    for i in range(len(bin_edges)-1):
        bin_data = final_predictions[
            (final_predictions['predicted_win_probability'] >= bin_edges[i]) &
            (final_predictions['predicted_win_probability'] < bin_edges[i+1])
        ]
        
        if len(bin_data) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_means.append((bin_edges[i] + bin_edges[i+1]) / 2)
            actual_means.append(bin_data['won'].mean() * 100)
    
    ax.plot([0, 100], [0, 100], 'k--', lw=2, label='Perfect Calibration')
    ax.plot(bin_means, actual_means, 'o-', linewidth=2.5, markersize=8, 
           color='#2E86AB', label='Model Calibration')
    ax.fill_between(bin_means, actual_means, bin_means, alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Predicted Probability (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Probability Calibration Plot', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Reliability diagram
    ax = axes[1]
    
    counts, edges = np.histogram(final_predictions['predicted_win_probability'], 
                                 bins=10, range=(0, 100))
    
    ax.bar(range(10), counts, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Probability Bins', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'{i*10}-{(i+1)*10}%' for i in range(10)], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Probability Calibration & Confidence Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('08_probability_calibration.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot as '08_probability_calibration.png'")
    plt.close()

def generate_feature_selection_report(df, has_advanced=True):
    """Generate comprehensive feature selection report"""
    
    # Get unique rows per match
    match_data = df.groupby('match_id').last().reset_index()
    
    if has_advanced:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'balls_remaining', 'current_run_rate', 'required_run_rate',
            'batting_historical_win_ratio', 'bowling_historical_win_ratio',
            'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
            'elo_difference', 'batting_player_strength', 'bowling_player_strength'
        ]
    else:
        features_to_analyze = [
            'current_score', 'wickets_fallen', 'overs_completed',
            'runs_required', 'balls_remaining', 'current_run_rate', 'required_run_rate'
        ]
    
    print("\n" + "="*120)
    print("FEATURE SELECTION REPORT FOR LOGISTIC REGRESSION")
    print("="*120)
    
    feature_report = []
    
    for feature in features_to_analyze:
        won_data = match_data[match_data['won'] == 1][feature]
        lost_data = match_data[match_data['won'] == 0][feature]
        
        # T-test
        t_stat, t_pvalue = stats.ttest_ind(won_data, lost_data)
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pvalue = stats.mannwhitneyu(won_data, lost_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(won_data)-1)*won_data.std()**2 + 
                             (len(lost_data)-1)*lost_data.std()**2) / 
                            (len(won_data) + len(lost_data) - 2))
        cohens_d = (won_data.mean() - lost_data.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Correlation with outcome
        correlation = np.corrcoef(match_data[feature], match_data['won'])[0, 1]
        
        feature_report.append({
            'Feature': feature,
            'Mean(Won)': won_data.mean(),
            'Mean(Lost)': lost_data.mean(),
            'Difference': won_data.mean() - lost_data.mean(),
            'T-Statistic': t_stat,
            'T-P-Value': t_pvalue,
            'U-Statistic': u_stat,
            'U-P-Value': u_pvalue,
            'Cohens_d': cohens_d,
            'Correlation': correlation,
            'Significant': t_pvalue < 0.05
        })
    
    report_df = pd.DataFrame(feature_report)
    report_df = report_df.sort_values('T-Statistic', key=abs, ascending=False)
    
    # Print report
    print(f"\n{'Feature':<35} {'T-Stat':>10} {'P-Value':>12} {'Cohen\'s d':>10} {'Correlation':>12} {'Significant':>12}")
    print("-"*120)
    
    for _, row in report_df.iterrows():
        sig = "✓ YES" if row['Significant'] else "✗ NO"
        print(f"{row['Feature']:<35} {row['T-Statistic']:>10.4f} {row['T-P-Value']:>12.4f} "
              f"{row['Cohens_d']:>10.4f} {row['Correlation']:>12.4f} {sig:>12}")
    
    print("="*120)
    
    significant_features = report_df[report_df['Significant']]['Feature'].tolist()
    print(f"\n✓ Significant Features ({len(significant_features)}):")
    for feat in significant_features:
        print(f"  • {feat}")
    
    print(f"\n✗ Non-Significant Features ({len(report_df) - len(significant_features)}):")
    for feat in report_df[~report_df['Significant']]['Feature'].tolist():
        print(f"  • {feat}")
    
    print("="*120 + "\n")
    
    return report_df

def main():
    print("\n" + "="*120)
    print("ICC WORLD CUP TEST PREDICTIONS - ADVANCED STATISTICAL ANALYSIS")
    print("="*120)
    
    # Load predictions
    df = load_predictions()
    
    if df is None:
        return
    
    # Load merged matches
    merged_df = load_merged_matches()
    
    # Check if advanced features are available
    has_advanced = check_advanced_features(df)
    
    # 1. Confusion Matrix Heatmap
    print("\n📊 1. Creating Confusion Matrix Heatmap...")
    plot_confusion_matrix_heatmap(df)
    
    # 2. ROC Curve
    print("📈 2. Creating ROC Curve...")
    plot_roc_curve(df)
    
    # 3. Feature Correlation Heatmap
    print("🔗 3. Creating Feature Correlation Heatmap...")
    correlation_matrix = plot_feature_correlation_heatmap(df, has_advanced)
    
    # 4. Pairplot
    print("📐 4. Creating Pairplot...")
    plot_pairplot(df, has_advanced)
    
    # 5. Distribution Analysis
    print("📊 5. Creating Distribution Analysis...")
    normality_df = plot_distribution_analysis(df, has_advanced)
    normality_df.to_csv('normality_test_results.csv', index=False)
    print("   ✓ Saved normality test results")
    
    # 6. Feature Boxplots
    print("📦 6. Creating Feature Box Plots...")
    plot_feature_importance_boxplot(df, has_advanced)
    
    # 7. Feature Statistics Table
    print("📋 7. Creating Feature Statistics Table...")
    stats_df = plot_feature_statistics_table(df, has_advanced)
    stats_df.to_csv('feature_statistics.csv', index=False)
    print("   ✓ Saved feature statistics")
    
    # 8. Probability Calibration
    print("🎯 8. Creating Probability Calibration Plot...")
    plot_probability_calibration(df)
    
    # 9. Feature Selection Report
    print("🔍 9. Generating Feature Selection Report...")
    report_df = generate_feature_selection_report(df, has_advanced)
    report_df.to_csv('feature_selection_report.csv', index=False)
    print("   ✓ Saved feature selection report")
    
    # Summary statistics
    print("\n" + "="*120)
    print("ANALYSIS SUMMARY")
    print("="*120)
    
    final_preds = df.groupby('match_id').last()
    final_accuracy = (final_preds['won'] == final_preds['predicted_winner']).mean()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Total Matches: {len(df['match_id'].unique())}")
    print(f"   • Total Predictions: {len(df)}")
    print(f"   • Final Accuracy: {final_accuracy*100:.2f}%")
    
    print(f"\n✓ Generated Visualizations:")
    print(f"   1. 01_confusion_matrix_heatmap.png")
    print(f"   2. 02_roc_curve.png")
    print(f"   3. 03_correlation_heatmap.png")
    print(f"   4. 04_pairplot.png")
    print(f"   5. 05_distribution_analysis.png")
    print(f"   6. 06_feature_boxplots.png")
    print(f"   7. 07_feature_statistics_table.png")
    print(f"   8. 08_probability_calibration.png")
    
    print(f"\n📄 Generated Reports:")
    print(f"   • normality_test_results.csv")
    print(f"   • feature_statistics.csv")
    print(f"   • feature_selection_report.csv")
    
    print("\n" + "="*120 + "\n")

if __name__ == "__main__":
    main()