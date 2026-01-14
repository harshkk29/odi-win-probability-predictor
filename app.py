import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🏆 One Day International (ODI) matches Win Probability Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B35;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        # NOTE: If you are using the model saved in the previous fixed code block,
        # you should change the filename here to 'worldcup_win_predictor_full_features.pkl'
        with open('worldcup_win_predictor.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['feature_columns'], model_data['scaler']
    except FileNotFoundError:
        st.error("Model file not found! Please run the training script first.")
        return None, None, None

# Load team features
@st.cache_data
def load_team_features():
    try:
        # NOTE: If you are using the team features saved in the previous fixed code block,
        # you should change the filename here to 'team_features_full.csv'
        df = pd.read_csv('team_features.csv')
        return df
    except FileNotFoundError:
        st.error("Team features file not found! Please run the training script first.")
        return None

def predict_win_probability(model, scaler, feature_columns,
                           current_score, wickets_fallen, overs_completed, 
                           target, batting_historical_win_ratio, bowling_historical_win_ratio,
                           head_to_head_ratio, batting_recent_form, bowling_recent_form,
                           elo_difference, batting_player_strength, bowling_player_strength):
    """Predict win probability with all advanced features"""
    
    balls_bowled = int(overs_completed * 6)
    balls_remaining = 300 - balls_bowled
    
    runs_required = target - current_score
    current_run_rate = (current_score / balls_bowled) * 6 if balls_bowled > 0 else 0
    required_run_rate = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0
    
    # Create feature array with all 15 features
    features = np.array([[
        current_score, 
        wickets_fallen, 
        overs_completed,
        runs_required, 
        balls_remaining,
        current_run_rate, 
        required_run_rate,
        batting_historical_win_ratio,
        bowling_historical_win_ratio,
        head_to_head_ratio,
        batting_recent_form,
        bowling_recent_form,
        elo_difference,
        batting_player_strength,
        bowling_player_strength
    ]])
    
    features_scaled = scaler.transform(features)
    win_prob = model.predict_proba(features_scaled)[0][1] * 100
    
    return {
        'batting_team_win_probability': win_prob,
        'bowling_team_win_probability': 100 - win_prob,
        'runs_required': runs_required,
        'balls_remaining': balls_remaining,
        'current_run_rate': current_run_rate,
        'required_run_rate': required_run_rate,
        'elo_difference': elo_difference,
        'h2h_ratio': head_to_head_ratio,
        'batting_form': batting_recent_form,
        'bowling_form': bowling_recent_form
    }

def create_probability_gauge(probability, team_name):
    """Create a gauge chart for win probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{team_name} Win Probability", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_run_rate_comparison(crr, rrr):
    """Create bar chart comparing run rates"""
    fig = go.Figure(data=[
        go.Bar(name='Current Run Rate', x=['Run Rate'], y=[crr], marker_color='#3498db'),
        go.Bar(name='Required Run Rate', x=['Run Rate'], y=[rrr], marker_color='#e74c3c')
    ])
    
    fig.update_layout(
        title='Run Rate Comparison',
        yaxis_title='Runs per Over',
        height=300,
        barmode='group'
    )
    return fig

def create_team_stats_comparison(batting_stats, bowling_stats):
    """Create comparison chart for team stats"""
    categories = ['Historical\nWin %', 'Recent\nForm', 'Player\nStrength']
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=[batting_stats['history'] * 100, batting_stats['form'] * 100, batting_stats['strength'] * 100],
            theta=categories,
            fill='toself',
            name='Batting Team'
        ),
        go.Scatterpolar(
            r=[bowling_stats['history'] * 100, bowling_stats['form'] * 100, bowling_stats['strength'] * 100],
            theta=categories,
            fill='toself',
            name='Bowling Team'
        )
    ])
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400,
        title='Team Statistics Comparison'
    )
    return fig

# Main app (FIXED)
def main():
    # Header
    st.markdown('<h1 class="main-header">🏆 One Day International (ODI) matches Win Probability Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Load model and features
    model, feature_columns, scaler = load_model()
    team_features_df = load_team_features()
    
    # **FIX: Gracefully handle missing files and return**
    if model is None or team_features_df is None:
        st.error("Application setup failed. Please ensure 'worldcup_win_predictor.pkl' and 'team_features.csv' exist.")
        return
    
    # Get unique teams and matchups (Execution is now safe here)
    all_teams = sorted(set(team_features_df['batting_team'].unique().tolist() + 
                          team_features_df['bowling_team'].unique().tolist()))
    
    # Sidebar for inputs
    st.sidebar.title("⚙️ Match Configuration")
    st.sidebar.markdown("---")
    
    # Select teams
    batting_team = st.sidebar.selectbox("🏏 Batting Team", all_teams, index=0)
    bowling_team = st.sidebar.selectbox("⚾ Bowling Team", 
                                        [t for t in all_teams if t != batting_team])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Match Details (Live)")
    
    # Match situation inputs
    target = st.sidebar.number_input("🎯 Target Score", 
                                     min_value=1, max_value=400, value=280, step=1)
    
    current_score = st.sidebar.number_input("📈 Current Score", 
                                            min_value=0, max_value=target, value=100, step=1)
    
    wickets_fallen = st.sidebar.slider("❌ Wickets Fallen", 
                                       min_value=0, max_value=10, value=3)
    
    overs_completed = st.sidebar.slider("⏱️ Overs Completed", 
                                        min_value=0.0, max_value=49.5, value=25.0, step=0.1)
    
    # Get team features from dataframe
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Team Features (from Training Data)")
    
    # Filter team features for selected teams
    team_match = team_features_df[
        (team_features_df['batting_team'] == batting_team) & 
        (team_features_df['bowling_team'] == bowling_team)
    ]
    
    if len(team_match) > 0:
        # Get the first matching record (they should have same features)
        features_row = team_match.iloc[0]
        
        batting_historical_win_ratio = features_row['batting_historical_win_ratio']
        bowling_historical_win_ratio = features_row['bowling_historical_win_ratio']
        head_to_head_ratio = features_row['head_to_head_ratio']
        batting_recent_form = features_row['batting_recent_form']
        bowling_recent_form = features_row['bowling_recent_form']
        elo_difference = features_row['elo_difference']
        batting_player_strength = features_row['batting_player_strength']
        bowling_player_strength = features_row['bowling_player_strength']
        
        # Display features in two columns
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.write(f"**{batting_team}**")
            st.metric("Historical Win %", f"{batting_historical_win_ratio*100:.1f}%")
            st.metric("Recent Form", f"{batting_recent_form*100:.1f}%")
            st.metric("Player Strength", f"{batting_player_strength*100:.1f}%")
        
        with col2:
            st.write(f"**{bowling_team}**")
            st.metric("Historical Win %", f"{bowling_historical_win_ratio*100:.1f}%")
            st.metric("Recent Form", f"{bowling_recent_form*100:.1f}%")
            st.metric("Player Strength", f"{bowling_player_strength*100:.1f}%")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔥 Match Dynamics")
        st.sidebar.metric("Head-to-Head Ratio", f"{head_to_head_ratio:.2f}")
        st.sidebar.metric("ELO Difference", f"{elo_difference:+.1f}")
        
    else:
        st.sidebar.warning(f"⚠️ No training data for {batting_team} vs {bowling_team}")
        batting_historical_win_ratio = 0.5
        bowling_historical_win_ratio = 0.5
        head_to_head_ratio = 0.5
        batting_recent_form = 0.5
        bowling_recent_form = 0.5
        elo_difference = 0
        batting_player_strength = 0.5
        bowling_player_strength = 0.5
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 PREDICT WIN PROBABILITY", use_container_width=True)
    
    # Main content
    if predict_button:
        # Get prediction with all features
        result = predict_win_probability(
            model, scaler, feature_columns,
            current_score, wickets_fallen, overs_completed, target,
            batting_historical_win_ratio, bowling_historical_win_ratio,
            head_to_head_ratio, batting_recent_form, bowling_recent_form,
            elo_difference, batting_player_strength, bowling_player_strength
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_probability_gauge(
                    result['batting_team_win_probability'], 
                    batting_team
                ),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_probability_gauge(
                    result['bowling_team_win_probability'], 
                    bowling_team
                ),
                use_container_width=True
            )
        
        # Match situation
        st.markdown("---")
        st.subheader("📊 Current Match Situation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Runs Required",
                value=f"{result['runs_required']}",
                delta=f"from {result['balls_remaining']} balls"
            )
        
        with col2:
            st.metric(
                label="Current Run Rate",
                value=f"{result['current_run_rate']:.2f}",
                delta="runs/over"
            )
        
        with col3:
            st.metric(
                label="Required Run Rate",
                value=f"{result['required_run_rate']:.2f}",
                delta="runs/over"
            )
        
        with col4:
            st.metric(
                label="Wickets Remaining",
                value=f"{10 - wickets_fallen}",
                delta="out of 10"
            )
        
        # Run rate comparison
        st.markdown("---")
        st.plotly_chart(
            create_run_rate_comparison(
                result['current_run_rate'],
                result['required_run_rate']
            ),
            use_container_width=True
        )
        
        # Team statistics comparison
        st.markdown("---")
        batting_stats = {
            'history': batting_historical_win_ratio,
            'form': batting_recent_form,
            'strength': batting_player_strength
        }
        bowling_stats = {
            'history': bowling_historical_win_ratio,
            'form': bowling_recent_form,
            'strength': bowling_player_strength
        }
        
        st.plotly_chart(
            create_team_stats_comparison(batting_stats, bowling_stats),
            use_container_width=True
        )
        
        # Match insights
        st.markdown("---")
        st.subheader("🔍 Match Insights")
        
        if result['batting_team_win_probability'] > 75:
            insight = f"🟢 **{batting_team}** is in a commanding position with a {result['batting_team_win_probability']:.1f}% chance of winning!"
        elif result['batting_team_win_probability'] > 60:
            insight = f"🟡 **{batting_team}** has a good chance with {result['batting_team_win_probability']:.1f}% probability of winning."
        elif result['batting_team_win_probability'] > 40:
            insight = f"🟠 This match is evenly contested! Slight edge to **{batting_team}** with {result['batting_team_win_probability']:.1f}% chance."
        else:
            insight = f"🔴 **{bowling_team}** is in control with a {result['bowling_team_win_probability']:.1f}% chance of winning!"
        
        st.info(insight)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Batting Team Analysis")
            st.write(f"**Historical Win Ratio:** {batting_historical_win_ratio:.2%}")
            st.write(f"**Recent Form:** {batting_recent_form:.2%}")
            st.write(f"**Player Strength:** {batting_player_strength:.2%}")
            
            if result['required_run_rate'] > 12:
                st.warning("⚠️ High RRR - Needs aggressive batting!")
            elif result['required_run_rate'] < 6:
                st.success("✅ Manageable RRR - Cruising pace is fine.")
        
        with col2:
            st.subheader("📋 Bowling Team Analysis")
            st.write(f"**Historical Win Ratio:** {bowling_historical_win_ratio:.2%}")
            st.write(f"**Recent Form:** {bowling_recent_form:.2%}")
            st.write(f"**Player Strength:** {bowling_player_strength:.2%}")
            
            if wickets_fallen > 6:
                st.warning("⚠️ Batting side weak - Few resources left!")
            elif wickets_fallen < 3:
                st.success("✅ Batting side strong - Many batters available.")
        
        # Head-to-head insight
        st.markdown("---")
        st.subheader("⚔️ Head-to-Head Dynamics")
        col1, col2 = st.columns(2)
        
        with col1:
            if head_to_head_ratio > 0.6:
                st.success(f"**{batting_team}** leads in H2H record ({head_to_head_ratio:.1%})")
            elif head_to_head_ratio < 0.4:
                st.error(f"**{bowling_team}** leads in H2H record ({(1-head_to_head_ratio):.1%})")
            else:
                st.info(f"Teams are evenly matched in H2H ({head_to_head_ratio:.1%})")
        
        with col2:
            if elo_difference > 50:
                st.success(f"**{batting_team}** has ELO advantage (+{elo_difference:.1f})")
            elif elo_difference < -50:
                st.error(f"**{bowling_team}** has ELO advantage (+{abs(elo_difference):.1f})")
            else:
                st.info(f"ELO ratings nearly equal (Diff: {elo_difference:+.1f})")
    
    else:
        # Welcome screen
        st.markdown("---")
        st.info("👈 Configure match details in the sidebar and click 'PREDICT WIN PROBABILITY' to get started!")
        
        st.subheader("📈 How It Works")
        st.markdown("""
        This predictor uses **15 advanced features** including:
        
        **Live Match Features:**
        - Current Score & Wickets Fallen
        - Overs Completed
        - Run Rates (Current vs Required)
        
        **Team Statistics Features (from Training Data):**
        - Historical Win Ratio
        - Recent Form (Last 5 Matches)
        - Player Strength Index
        - Head-to-Head Record
        - ELO Rating Difference
        
        All advanced features are **automatically loaded from training data** - no manual configuration needed!
        """)
        
        st.markdown("---")
        st.subheader("📊 Sample Predictions")
        st.info(f"**Total team matchups available:** {len(team_features_df)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Made with ❤️ using Streamlit | "
        "🏆 Win Probability Predictor © 2025 | 15 Advanced ML Features</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()