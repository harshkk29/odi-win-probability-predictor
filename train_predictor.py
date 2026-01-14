import json
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import warnings
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================
# PART 0: TEAM STATISTICS TRACKER (FIXED)
# ============================================

class TeamStatsTracker:
    """Track team statistics across matches, including ELO and player stats."""
    def __init__(self):
        self.match_history = defaultdict(list)
        self.elo_ratings = defaultdict(lambda: 1600)
        # player_stats: {team: {player: {'runs': 0, 'matches': 0}}}
        self.player_stats = defaultdict(lambda: defaultdict(lambda: {'runs': 0, 'matches': 0}))
    
    def update_match_result(self, date, team, opponent, result, match_data):
        """Update team history, ELO, and PLAYER STATS after a match."""
        self.match_history[team].append({
            'date': date,
            'opponent': opponent,
            'result': result,
            'match_data': match_data
        })
        self.update_elo(team, opponent, result)
        self.update_player_stats(team, opponent, match_data) 

    def update_player_stats(self, team1, team2, match_data):
        """Update individual player statistics (runs and matches played) for both teams."""
        if 'innings' not in match_data:
            return

        teams = {team1, team2}
        players_in_match = defaultdict(set)
        
        # 1. Identify all players who participated from the 'info' section
        try:
            for team in teams:
                players_in_match[team].update(match_data['info']['players'].get(team, []))
        except:
            pass

        # 2. Update runs scored
        for inning in match_data['innings']:
            batting_team = inning['team']
            if batting_team not in teams:
                continue
            
            # Update runs scored for batters
            for over_data in inning['overs']:
                for delivery in over_data['deliveries']:
                    batter = delivery.get('batter')
                    runs_scored = delivery['runs'].get('batter', 0)
                    
                    if batter and runs_scored > 0:
                        self.player_stats[batting_team][batter]['runs'] += runs_scored

        # 3. Update 'matches' played for all players listed in the match info (once per match)
        for team, players in players_in_match.items():
            for player in players:
                self.player_stats[team][player]['matches'] += 1

    def update_elo(self, team, opponent, result, k=32):
        """Update ELO ratings (1 = win, 0 = loss)"""
        team_elo = self.elo_ratings[team]
        opponent_elo = self.elo_ratings[opponent]
        
        expected = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))
        
        self.elo_ratings[team] = team_elo + k * (result - expected)
        self.elo_ratings[opponent] = opponent_elo + k * ((1 - result) - (1 - expected))
    
    def get_historical_win_ratio(self, team):
        """Get proportion of matches won"""
        if not self.match_history[team]:
            return 0.5
        
        wins = sum(1 for m in self.match_history[team] if m['result'] == 1)
        total = len(self.match_history[team])
        return wins / total if total > 0 else 0.5
    
    def get_head_to_head(self, team1, team2):
        """Get head-to-head win ratio for team1 against team2"""
        h2h_matches = [m for m in self.match_history[team1] if m['opponent'] == team2]
        
        if not h2h_matches:
            return 0.5
        
        wins = sum(1 for m in h2h_matches if m['result'] == 1)
        return wins / len(h2h_matches)
    
    def get_recent_form(self, team, matches=5):
        """Get win ratio from last N matches"""
        recent = self.match_history[team][-matches:]
        
        if not recent:
            return 0.5
        
        wins = sum(1 for m in recent if m['result'] == 1)
        return wins / len(recent)
    
    def get_elo_difference(self, team1, team2):
        """Get ELO rating difference"""
        return self.elo_ratings[team1] - self.elo_ratings[team2]
    
    def get_elo_ratings(self, team):
        """Get ELO rating for a team"""
        return self.elo_ratings[team]
    
    def get_player_strength(self, team, players_list):
        """Calculate aggregate player strength (Average Runs/Match)"""
        if not players_list:
            return 0.5
        
        valid_players = [
            player for player in players_list 
            if player in self.player_stats[team] and self.player_stats[team][player]['matches'] > 0
        ]
        
        if not valid_players:
            return 0.5

        player_averages = []
        for player in valid_players:
            stats = self.player_stats[team][player]
            if stats['matches'] > 0:
                 # Calculate simple average runs per match
                player_averages.append(stats['runs'] / stats['matches'])
        
        if not player_averages:
            return 0.5
            
        # Calculate the average runs per match for the playing XI
        avg_runs_per_match_xi = sum(player_averages) / len(player_averages)
        
        # Normalize the strength (e.g., max strength at 40 runs/match, capped at 1.0)
        return min(avg_runs_per_match_xi / 40, 1.0)

# ============================================
# PART 1: DATA LOADING FROM FOLDER
# ============================================

def load_match_from_json(file_path):
    """Load a single JSON match file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
        return match_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_all_matches(folder_path):
    """Load all JSON files from folder"""
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return []
    
    print(f"Found {len(json_files)} match files")
    
    matches = []
    for file_path in json_files:
        match_data = load_match_from_json(file_path)
        if match_data:
            matches.append({
                'file_name': os.path.basename(file_path),
                'data': match_data
            })
    
    print(f"Successfully loaded {len(matches)} matches")
    return matches

def sort_matches_by_date(matches):
    """Sort matches by date in ascending order"""
    def get_match_date(match):
        try:
            date_str = match['data']['info'].get('dates', [None])[0]
            if date_str:
                return datetime.strptime(date_str, '%Y-%m-%d')
            return datetime.min
        except:
            return datetime.min
    
    sorted_matches = sorted(matches, key=get_match_date)
    
    print("\n" + "="*60)
    print("MATCHES SORTED BY DATE (ASCENDING)")
    print("="*60)
    for match in sorted_matches:
        date_str = match['data']['info'].get('dates', ['Unknown'])[0]
        print(f"{match['file_name']:<40} | {date_str}")
    print("="*60)
    
    return sorted_matches

def merge_all_matches_info(matches):
    """Merge all match information into a single dataframe"""
    merged_data = []
    
    for match in matches:
        try:
            info = match['data']['info']
            match_record = {
                'file_name': match['file_name'],
                'date': info.get('dates', [None])[0],
                'match_type': info.get('match_type', 'Unknown'),
                'teams': ', '.join(info.get('teams', [])),
                'venue': info.get('venue', 'Unknown'),
                'city': info.get('city', 'Unknown'),
                'winner': info.get('outcome', {}).get('winner', 'No Result'),
                'gender': info.get('gender', 'Unknown'),
                'overs': info.get('overs', 'Unknown')
            }
            merged_data.append(match_record)
        except Exception as e:
            print(f"Error processing {match['file_name']}: {e}")
    
    if merged_data:
        merged_df = pd.DataFrame(merged_data)
        merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
        merged_df = merged_df.sort_values('date', ascending=True).reset_index(drop=True)
        return merged_df
    
    return pd.DataFrame()

# ============================================
# PART 2: FEATURE EXTRACTION WITH ADVANCED STATS
# ============================================

def create_ball_by_ball_dataset(match_data, match_id, stats_tracker, match_date):
    """Create ball-by-ball dataset with advanced features"""
    dataset = []
    
    try:
        info = match_data['info']
        innings = match_data['innings']
        
        if len(innings) < 2 or 'outcome' not in info or 'winner' not in info['outcome']:
            return pd.DataFrame()
        
        winner = info['outcome']['winner']
        teams = info['teams']
        venue = info.get('venue', 'Unknown')
        city = info.get('city', 'Unknown')
        
        # Team batting second (Innings 2) is the focus for ball-by-ball prediction
        batting_team = innings[1]['team']
        bowling_team = innings[0]['team']
        target = innings[1].get('target', {}).get('runs', 0)
        
        if target == 0:
            return pd.DataFrame()
        
        # Get playing XIs
        batting_xi = info.get('players', {}).get(batting_team, [])
        bowling_xi = info.get('players', {}).get(bowling_team, [])
        
        # Get team statistics BEFORE this match
        batting_win_ratio = stats_tracker.get_historical_win_ratio(batting_team)
        bowling_win_ratio = stats_tracker.get_historical_win_ratio(bowling_team)
        h2h_ratio = stats_tracker.get_head_to_head(batting_team, bowling_team)
        batting_recent_form = stats_tracker.get_recent_form(batting_team, matches=5)
        bowling_recent_form = stats_tracker.get_recent_form(bowling_team, matches=5)
        elo_diff = stats_tracker.get_elo_difference(batting_team, bowling_team)
        
        # Player strength should now be calculated correctly due to the fix
        batting_player_strength = stats_tracker.get_player_strength(batting_team, batting_xi)
        bowling_player_strength = stats_tracker.get_player_strength(bowling_team, bowling_xi)
        
        current_score = 0
        current_wickets = 0
        balls_bowled = 0
        
        # Create ball-by-ball records (for second innings only)
        for over_data in innings[1]['overs']:
            
            for delivery in over_data['deliveries']:
                balls_bowled += 1
                current_score += delivery['runs']['total']
                
                if 'wickets' in delivery:
                    current_wickets += len(delivery['wickets'])
                
                balls_remaining = 300 - balls_bowled
                runs_required = target - current_score
                overs_completed = (balls_bowled / 6)
                current_run_rate = (current_score / balls_bowled) * 6 if balls_bowled > 0 else 0
                required_run_rate = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0
                
                won = 1 if batting_team == winner else 0
                
                record = {
                    'match_id': match_id,
                    'match_date': match_date,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'city': city,
                    'venue': venue,
                    'current_score': current_score,
                    'wickets_fallen': current_wickets,
                    'overs_completed': overs_completed,
                    'balls_bowled': balls_bowled,
                    'runs_required': runs_required,
                    'balls_remaining': balls_remaining,
                    'current_run_rate': current_run_rate,
                    'required_run_rate': required_run_rate,
                    'target': target,
                    # Advanced Features
                    'batting_historical_win_ratio': batting_win_ratio,
                    'bowling_historical_win_ratio': bowling_win_ratio,
                    'head_to_head_ratio': h2h_ratio,
                    'batting_recent_form': batting_recent_form,
                    'bowling_recent_form': bowling_recent_form,
                    'elo_difference': elo_diff,
                    'batting_player_strength': batting_player_strength,
                    'bowling_player_strength': bowling_player_strength,
                    'won': won
                }
                dataset.append(record)
        
        # Update stats tracker AFTER ball-by-ball processing is complete
        team1 = innings[0]['team']
        team2 = innings[1]['team']
        
        team1_result = 1 if team1 == winner else 0 # 1=Win, 0=Loss for team1
        
        # Update stats for Team 1 vs Team 2. 
        stats_tracker.update_match_result(match_date, team1, team2, team1_result, match_data)
                
    except Exception as e:
        print(f"Error processing match {match_id}: {e}")
        return pd.DataFrame()
    
    return pd.DataFrame(dataset)

def process_all_matches(matches):
    """Process all matches in chronological order to build historical context"""
    all_data = []
    stats_tracker = TeamStatsTracker()
    
    print("\nProcessing matches in chronological order...")
    for i, match in enumerate(matches):
        match_id = match['file_name'].replace('.json', '')
        
        try:
            match_date = match['data']['info'].get('dates', [None])[0]
            if match_date:
                match_date = datetime.strptime(match_date, '%Y-%m-%d')
            else:
                match_date = datetime.min
        except:
            match_date = datetime.min
        
        df = create_ball_by_ball_dataset(match['data'], match_id, stats_tracker, match_date)
        
        if not df.empty:
            all_data.append(df)
            print(f"✓ Processed {match_id}: {len(df)} ball-by-ball records")
        else:
            print(f"✗ Skipped {match_id}: No valid data (Target 0 or structure issue)")
    
    if not all_data:
        print("No valid data found!")
        return pd.DataFrame(), None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df = combined_df.sort_values(
        by=['match_date', 'batting_team', 'balls_bowled'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    print(f"\n✓ Total ball-by-ball records: {len(combined_df)}")
    print(f"✓ Total matches processed: {len(all_data)}")
    
    return combined_df, stats_tracker

# ============================================
# PART 3: EXTRACT TEAM FEATURES
# ============================================

def extract_team_features(combined_df):
    """Extract unique team features for use in Streamlit"""
    team_features = []
    
    # Drop duplicates to get one row per match-team combination
    for _, row in combined_df.drop_duplicates(subset=['match_id', 'batting_team']).iterrows():
        team_features.append({
            'match_id': row['match_id'],
            'batting_team': row['batting_team'],
            'bowling_team': row['bowling_team'],
            'target': int(row['target']),
            'batting_historical_win_ratio': row['batting_historical_win_ratio'],
            'bowling_historical_win_ratio': row['bowling_historical_win_ratio'],
            'head_to_head_ratio': row['head_to_head_ratio'],
            'batting_recent_form': row['batting_recent_form'],
            'bowling_recent_form': row['bowling_recent_form'],
            'elo_difference': row['elo_difference'],
            'batting_player_strength': row['batting_player_strength'],
            'bowling_player_strength': row['bowling_player_strength']
        })
    
    team_features_df = pd.DataFrame(team_features)
    print(f"\n✓ Extracted features for {len(team_features_df)} team matchups")
    return team_features_df

# ============================================
# PART 4: TRAIN-TEST SPLIT BY MATCHES
# ============================================

def split_by_matches(df, test_size=0.2, random_state=42):
    """Split data by matches (80% train, 20% test)"""
    match_ids = df['match_id'].unique()
    
    train_matches, test_matches = train_test_split(
        match_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_df = df[df['match_id'].isin(train_matches)].copy()
    test_df = df[df['match_id'].isin(test_matches)].copy()
    
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    print(f"Training matches: {len(train_matches)} ({(1-test_size)*100:.0f}%)")
    print(f"Testing matches:  {len(test_matches)} ({test_size*100:.0f}%)")
    print(f"Training records: {len(train_df)}")
    print(f"Testing records:  {len(test_df)}")
    print("="*60)
    
    return train_df, test_df, train_matches, test_matches

# ============================================
# PART 5: MODEL TRAINING
# ============================================

def train_model(train_df):
    """Train the win probability model including current_score and FIXED player strength"""
    feature_columns = [
        'current_score', 
        'wickets_fallen', 'overs_completed',
        'runs_required', 'balls_remaining',
        'current_run_rate', 'required_run_rate',
        'batting_historical_win_ratio', 'bowling_historical_win_ratio',
        'head_to_head_ratio', 'batting_recent_form', 'bowling_recent_form',
        'elo_difference',
        'batting_player_strength', 'bowling_player_strength' 
    ]
    
    X_train = train_df[feature_columns].copy()
    y_train = train_df['won'].copy()
    
    X_train = X_train.fillna(0.5)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("\n" + "="*60)
    print("TRAINING MODEL (Fixed Player Strength Included)")
    print("="*60)
    print(f"Total Features: {len(feature_columns)}")
    print("Features:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"  {i}. {feat}")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = model.score(X_train_scaled, y_train)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print("="*60)
    
    return model, feature_columns, scaler

# ============================================
# PART 6: BALL-BY-BALL PREDICTION ON TEST DATA (FILENAMES ADJUSTED)
# ============================================

def predict_ball_by_ball(model, test_df, feature_columns, scaler):
    """Make predictions for each ball in test matches using the updated feature set"""
    X_test = test_df[feature_columns].copy()
    X_test = X_test.fillna(0.5)
    
    X_test_scaled = scaler.transform(X_test)
    
    probabilities = model.predict_proba(X_test_scaled)[:, 1] * 100
    
    test_df = test_df.copy()
    test_df['predicted_win_probability'] = probabilities
    test_df['predicted_winner'] = (probabilities >= 50).astype(int)
    
    return test_df

def evaluate_test_matches(predictions_df):
    """Evaluate model performance on test matches"""
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    y_true = predictions_df['won']
    y_pred = predictions_df['predicted_winner']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred,
                                target_names=['Batting Team Lost', 'Batting Team Won']))
    print("="*60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def display_match_predictions(predictions_df, match_id, interval=10):
    """Display ball-by-ball predictions for a specific test match"""
    match_data = predictions_df[predictions_df['match_id'] == match_id].copy()
    
    if match_data.empty:
        print(f"No data found for match {match_id}")
        return
    
    print("\n" + "="*60)
    print(f"BALL-BY-BALL PREDICTIONS: {match_id}")
    print("="*60)
    
    batting_team = match_data['batting_team'].iloc[0]
    bowling_team = match_data['bowling_team'].iloc[0]
    target = match_data['target'].iloc[0]
    actual_winner = "WON" if match_data['won'].iloc[0] == 1 else "LOST"
    
    print(f"Batting: {batting_team} | Bowling: {bowling_team}")
    print(f"Target:  {target} | Result: {batting_team} {actual_winner}")
    
    h2h = match_data['head_to_head_ratio'].iloc[0]
    batting_form = match_data['batting_recent_form'].iloc[0]
    bowling_form = match_data['bowling_recent_form'].iloc[0]
    elo_diff = match_data['elo_difference'].iloc[0]
    
    print(f"\nTeam Stats (Pre-Match):")
    print(f"  H2H Ratio: {h2h:.2f} | {batting_team} Recent Form: {batting_form:.2f} | "
          f"{bowling_team} Recent Form: {bowling_form:.2f} | ELO Diff: {elo_diff:.1f}")
    
    batting_strength = match_data['batting_player_strength'].iloc[0]
    bowling_strength = match_data['bowling_player_strength'].iloc[0]
    print(f"  {batting_team} Player Strength: {batting_strength:.2f} | "
          f"{bowling_team} Player Strength: {bowling_strength:.2f}")
    
    print("="*60)
    
    print(f"\n{'Ball':<6} {'Score':<12} {'Need':<15} {'CRR':<8} {'RRR':<8} {'Win%':<8}")
    print("-"*70)
    
    for idx, row in match_data.iterrows():
        # Display every 'interval' balls or the last ball
        if row['balls_bowled'] % interval == 0 or row['balls_bowled'] == len(match_data):
            
            score = f"{int(row['current_score'])}/{int(row['wickets_fallen'])}"
            need = f"{int(row['runs_required'])}({int(row['balls_remaining'])})"
            
            print(f"{row['balls_bowled']:<6} {score:<12} {need:<15} {row['current_run_rate']:>6.2f} "
                  f"{row['required_run_rate']:>6.2f} {row['predicted_win_probability']:>6.1f}%")
    
    print("="*60)

def save_predictions_to_csv(predictions_df, filename='test_predictions.csv'):
    """Save all predictions to CSV (Simplified filename)"""
    predictions_df.to_csv(filename, index=False)
    print(f"\n✓ Predictions saved to '{filename}'")

def save_team_features(team_features_df, filename='team_features.csv'):
    """Save team features for Streamlit use (Simplified filename)"""
    team_features_df.to_csv(filename, index=False)
    print(f"✓ Team features saved to '{filename}'")

def save_merged_matches(merged_df, filename='merged_matches.csv'):
    """Save merged match information to CSV (Simplified filename)"""
    merged_df.to_csv(filename, index=False)
    print(f"✓ Merged matches saved to '{filename}'")

def save_model(model, feature_columns, scaler, filename='worldcup_win_predictor.pkl'):
    """Save trained model with scaler (Simplified filename)"""
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'scaler': scaler
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Model saved to '{filename}'")

# ============================================
# PART 7: MAIN EXECUTION
# ============================================

def main():
    print("\n" + "="*60)
    print("ICC WORLD CUP WIN PROBABILITY PREDICTION SYSTEM (FULL FEATURES)")
    print("PLAYER STRENGTH FEATURE IS NOW CORRECTLY CALCULATED.")
    print("FILENAMES ADJUSTED TO MATCH STREAMLIT APP EXPECTATIONS.")
    print("="*60)
    
    folder_path = input("\nEnter the path to JSON match folder: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return
    
    # Step 1: Load all matches
    print("\nStep 1: Loading matches from folder...")
    matches = load_all_matches(folder_path)
    
    if not matches:
        print("No matches loaded. Exiting.")
        return
    
    # Step 1b: Sort matches by date
    print("\nStep 1b: Sorting matches by date...")
    sorted_matches = sort_matches_by_date(matches)
    
    # Step 1c: Merge all matches info
    print("\nStep 1c: Merging match information...")
    merged_df = merge_all_matches_info(sorted_matches)
    
    if not merged_df.empty:
        print("\n" + "="*60)
        print("MERGED MATCHES SUMMARY (Sorted by Date)")
        print("="*60)
        print(merged_df.to_string(index=False))
        print("="*60)
        save_merged_matches(merged_df)
    
    # Step 2: Process all matches with team statistics
    print("\nStep 2: Processing matches with team statistics...")
    combined_df, stats_tracker = process_all_matches(sorted_matches)
    
    if combined_df.empty:
        print("No valid data processed. Exiting.")
        return
    
    # Step 2b: Extract team features for Streamlit
    print("\nStep 2b: Extracting team features...")
    team_features_df = extract_team_features(combined_df)
    save_team_features(team_features_df)
    
    # Step 3: Split into train and test
    print("\nStep 3: Splitting into train and test sets...")
    train_df, test_df, train_matches, test_matches = split_by_matches(combined_df)
    
    # Step 4: Train model with updated features
    print("\nStep 4: Training model with updated features...")
    model, feature_columns, scaler = train_model(train_df)
    
    # Step 5: Ball-by-ball predictions
    print("\nStep 5: Making ball-by-ball predictions...")
    predictions_df = predict_ball_by_ball(model, test_df, feature_columns, scaler)
    
    # Step 6: Evaluate performance
    print("\nStep 6: Evaluating model performance...")
    metrics = evaluate_test_matches(predictions_df)
    
    # Step 7: Display sample predictions
    print("\nStep 7: Displaying sample match predictions...")
    
    for i, match_id in enumerate(test_matches[:3]):
        display_match_predictions(predictions_df, match_id, interval=12)
        
        if i < 2:
            cont = input("\nPress Enter to see next match (or 'q' to skip): ")
            if cont.lower() == 'q':
                break
    
    # Step 8: Save results
    print("\nStep 8: Saving results...")
    save_predictions_to_csv(predictions_df)
    save_model(model, feature_columns, scaler)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (FULL FEATURES INCLUDED)")
    print("="*60)
    print(f"Total Matches:        {len(sorted_matches)}")
    print(f"Training Matches:     {len(train_matches)}")
    print(f"Testing Matches:      {len(test_matches)}")
    print(f"Test Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Test F1-Score:        {metrics['f1_score']:.4f}")
    print("="*60)
    print("\n✓ Predictions saved to 'test_predictions.csv'")
    print("✓ Model saved to 'worldcup_win_predictor.pkl'")
    print("✓ Team features saved to 'team_features.csv'")
    print("\n**NEXT STEP:** Run your Streamlit app now. The necessary files have been created.")
    print("="*60 + "\n")
    
if __name__ == "__main__":
    main()