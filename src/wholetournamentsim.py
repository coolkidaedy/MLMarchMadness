#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import random

def load_and_prepare_data():
    team_stats = pd.read_csv('mad.csv')
    print("Available columns in team_stats:")
    print(team_stats.columns.tolist())
    
    tournament_results = pd.read_csv('post.csv')
    
    march_madness = tournament_results[tournament_results['Post-Season Tournament'] == 'March Madness']
    recent_years = march_madness[march_madness['Season'] >= 2010].copy()
    
    recent_years = recent_years[['Season', 'Team Name', 'Seed', 'Region', 
                                'Tournament Winner?', 'Tournament Championship?', 'Final Four?']]
    
    key_columns = ['Season', 'Mapped ESPN Team Name']
    stat_columns = []
    
    potential_stat_columns = [
        'Adjusted Offensive Efficiency', 'Adjusted Defensive Efficiency',
        'AdjEM', 'FG2Pct', 'FG3Pct', 'FTPct', 'Experience',
        'eFGPct', 'TOPct', 'ORPct', 'FTRate', 'Tempo',
        'BlockPct', 'StlRate'
    ]
    
    for col in potential_stat_columns:
        if col in team_stats.columns:
            stat_columns.append(col)
    
    team_stats_slim = team_stats[key_columns + stat_columns]
    
    print("\nUsing these statistical columns:")
    print(stat_columns)
    
    tournament_teams = pd.merge(
        recent_years,
        team_stats_slim,
        left_on=['Season', 'Team Name'],
        right_on=['Season', 'Mapped ESPN Team Name'],
        how='inner'
    )
    
    print(f"Number of tournament teams with stats: {len(tournament_teams)}")
    
    champions = tournament_teams[tournament_teams['Tournament Winner?'] == 'Yes'].copy()
    if len(champions) > 0:
        print(f"Found {len(champions)} champions in the dataset")
        champion_profile_cols = [col for col in stat_columns if col in champions.columns]
        champion_profile = champions[champion_profile_cols].mean()
        
        for feature in champion_profile_cols:
            std = tournament_teams[feature].std()
            if std > 0:
                tournament_teams[f'{feature}_ChampDiff'] = (tournament_teams[feature] - champion_profile[feature]) / std
        
        champ_diff_cols = [f'{feature}_ChampDiff' for feature in champion_profile_cols]
        tournament_teams['Champion_Similarity'] = -np.sqrt((tournament_teams[champ_diff_cols]**2).sum(axis=1))
        print("Added champion similarity features")
    else:
        tournament_teams['Champion_Similarity'] = 0
        print("No champions found in data, added placeholder similarity feature")
    
    tournament_teams['Late_Season_Momentum'] = np.random.uniform(-5, 5, size=len(tournament_teams))
    
    tournament_teams['Tournament_Experience'] = np.random.uniform(0, 3, size=len(tournament_teams))
    
    if 'Adjusted Offensive Efficiency' in tournament_teams.columns and 'Adjusted Defensive Efficiency' in tournament_teams.columns:
        tournament_teams['Off_Def_Ratio'] = tournament_teams['Adjusted Offensive Efficiency'] / tournament_teams['Adjusted Defensive Efficiency']
    
    if 'Adjusted Offensive Efficiency' in tournament_teams.columns and 'Adjusted Defensive Efficiency' in tournament_teams.columns:
        tournament_teams['Balance_Score'] = -abs(tournament_teams['Adjusted Offensive Efficiency'] - 
                                          (200 - tournament_teams['Adjusted Defensive Efficiency']))
    
    upset_probabilities = {
        (16, 1): 0.01,
        (15, 2): 0.05,
        (14, 3): 0.15,
        (13, 4): 0.20,
        (12, 5): 0.35,
        (11, 6): 0.32,
        (10, 7): 0.40,
        (9, 8): 0.55,
    }
    
    matchups = []
    
    for (season, region), group in tournament_teams.groupby(['Season', 'Region']):
        print(f"\nProcessing Season: {season}, Region: {region}")
        
        if 'Seed' not in group.columns:
            print("WARNING: 'Seed' column not found!")
            if 'Seed_x' in group.columns:
                print("Found 'Seed_x', using it instead")
                group = group.rename(columns={'Seed_x': 'Seed'})
            else:
                print("No suitable Seed column found, skipping this group")
                continue
                
        if not pd.api.types.is_numeric_dtype(group['Seed']):
            print(f"Converting Seed column from {group['Seed'].dtype} to numeric")
            group['Seed'] = pd.to_numeric(group['Seed'], errors='coerce')
        
        group_sorted = group.sort_values('Seed')
        seeds = group_sorted['Seed'].unique()
        
        print(f"Found {len(seeds)} unique seeds: {seeds}")
        
        for i in range(len(seeds)//2):
            seed1 = seeds[i]
            seed2 = seeds[-(i+1)]
            
            team1_df = group_sorted[group_sorted['Seed'] == seed1]
            team2_df = group_sorted[group_sorted['Seed'] == seed2]
            
            if len(team1_df) == 0 or len(team2_df) == 0:
                print(f"Skipping matchup seed {seed1} vs {seed2} due to missing teams")
                continue
                
            team1 = team1_df.iloc[0]
            team2 = team2_df.iloc[0]
            
            team1_winner = False
            team2_winner = False
            
            if pd.notna(team1['Tournament Winner?']) and team1['Tournament Winner?'] == 'Yes':
                team1_winner = True
            elif pd.notna(team2['Tournament Winner?']) and team2['Tournament Winner?'] == 'Yes':
                team2_winner = True
            elif pd.notna(team1['Tournament Championship?']) and team1['Tournament Championship?'] == 'Yes':
                team1_winner = True
            elif pd.notna(team2['Tournament Championship?']) and team2['Tournament Championship?'] == 'Yes':
                team2_winner = True
            elif pd.notna(team1['Final Four?']) and team1['Final Four?'] == 'Yes':
                team1_winner = True
            elif pd.notna(team2['Final Four?']) and team2['Final Four?'] == 'Yes':
                team2_winner = True
            else:
                if seed1 > seed2:
                    team1, team2 = team2, team1
                    seed1, seed2 = seed2, seed1
                    
                upset_prob = upset_probabilities.get((seed2, seed1), 0.5)
                
                if 'AdjEM' in team1 and 'AdjEM' in team2:
                    team1_quality = team1['AdjEM']
                    team2_quality = team2['AdjEM']
                    quality_diff = team1_quality - team2_quality
                    
                    quality_adjustment = 0.01 * quality_diff
                    upset_prob = max(0.01, min(0.99, upset_prob - quality_adjustment))
                
                upset_prob += random.uniform(-0.1, 0.1)
                upset_prob = max(0.01, min(0.99, upset_prob))
                
                if random.random() < upset_prob:
                    team1_winner = False
                else:
                    team1_winner = True
            
            matchup = {
                'Season': season,
                'Region': region,
                'Team1': team1['Mapped ESPN Team Name'],
                'Team2': team2['Mapped ESPN Team Name'],
                'Seed1': seed1,
                'Seed2': seed2,
                'Seed_Diff': seed1 - seed2,
                'Team1_Won': 1 if team1_winner else 0
            }
            
            for stat in stat_columns:
                if stat in team1 and stat in team2:
                    matchup[f'{stat}_Diff'] = team1[stat] - team2[stat]
            
            if 'Champion_Similarity' in team1 and 'Champion_Similarity' in team2:
                matchup['Champion_Similarity_Diff'] = team1['Champion_Similarity'] - team2['Champion_Similarity']
            
            if 'Late_Season_Momentum' in team1 and 'Late_Season_Momentum' in team2:
                matchup['Momentum_Diff'] = team1['Late_Season_Momentum'] - team2['Late_Season_Momentum']
            
            if 'Tournament_Experience' in team1 and 'Tournament_Experience' in team2:
                matchup['Experience_Diff'] = team1['Tournament_Experience'] - team2['Tournament_Experience']
            
            if 'Off_Def_Ratio' in team1 and 'Off_Def_Ratio' in team2:
                matchup['Efficiency_Ratio'] = team1['Off_Def_Ratio'] / team2['Off_Def_Ratio']
            
            if 'Balance_Score' in team1 and 'Balance_Score' in team2:
                matchup['Balance_Diff'] = team1['Balance_Score'] - team2['Balance_Score']
            
            if 'AdjEM' in team1 and 'AdjEM' in team2:
                seed_weight = 0.1
                stats_weight = 0.9
                
                normalized_seed_diff = (seed1 - seed2) * 2
                
                matchup['Composite_Score'] = (
                    seed_weight * normalized_seed_diff + 
                    stats_weight * (team1['AdjEM'] - team2['AdjEM'])
                )
            
            matchups.append(matchup)
    
    matchups_df = pd.DataFrame(matchups)
    print(f"Number of matchups created: {len(matchups_df)}")
    
    seed_matchups = []
    for i in range(1, 9):
        for j in range(9, 17):
            seed_matchups.append((i, j))
    
    for seed1, seed2 in seed_matchups:
        matchups_df[f'Seed_{seed1}_vs_{seed2}'] = ((matchups_df['Seed1'] == seed1) & 
                                                (matchups_df['Seed2'] == seed2)).astype(int)
    
    if len(matchups_df) == 0:
        raise ValueError("No matchups were created! Check your data structure.")
    
    return matchups_df

def train_random_forest(matchups_df):
    exclude_cols = ['Season', 'Region', 'Team1', 'Team2', 'Team1_Won']
    feature_cols = [col for col in matchups_df.columns if col not in exclude_cols and not pd.isna(matchups_df[col]).all()]
    
    print(f"Using {len(feature_cols)} features for the model:")
    print(feature_cols)
    
    X = matchups_df[feature_cols]
    y = matchups_df['Team1_Won']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'class_weight': [None, 'balanced']
    }
    
    model = RandomForestClassifier(random_state=42)
    
    print("\nFinding optimal model parameters...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()
    
    print(f"Model Accuracy: {accuracy:.4f}")
    
    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Support: {len(y_test)}")
    
    upset_indices = (y_test == 0) & (matchups_df.loc[y_test.index, 'Seed1'] < matchups_df.loc[y_test.index, 'Seed2'])
    if np.sum(upset_indices) > 0:
        upset_accuracy = np.mean(y_pred[upset_indices] == y_test[upset_indices])
        print(f"\nUpset prediction accuracy: {upset_accuracy:.4f} on {np.sum(upset_indices)} upsets")
    
    feature_importances = best_model.feature_importances_
    
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    })
    importances = importances.sort_values('Importance', ascending=False)
    
    print("\nTop 15 Feature Importances:")
    print(importances.head(15))
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importances.head(15))
    plt.title('Top 15 Feature Importances for March Madness Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return best_model, scaler, feature_cols

def predict_tournament(model, scaler, features, current_matchups):
    for feature in features:
        if feature not in current_matchups.columns:
            if feature.startswith('Seed_') and '_vs_' in feature:
                seed1, seed2 = map(int, feature.replace('Seed_', '').split('_vs_'))
                current_matchups[feature] = ((current_matchups['Seed1'] == seed1) & 
                                         (current_matchups['Seed2'] == seed2)).astype(int)
            else:
                print(f"Warning: Feature {feature} not found in current matchups, setting to 0")
                current_matchups[feature] = 0
    
    X_pred = current_matchups[features].values
    X_pred_scaled = scaler.transform(X_pred)
    
    probabilities = model.predict_proba(X_pred_scaled)[:, 1]
    predictions = model.predict(X_pred_scaled)
    
    results = current_matchups.copy()
    results['Team1_Win_Probability'] = probabilities
    results['Predicted_Winner'] = np.where(predictions == 1, 
                                          results['Team1'], 
                                          results['Team2'])
    results['Predicted_Winner_Seed'] = np.where(predictions == 1, 
                                             results['Seed1'], 
                                             results['Seed2'])
    results['Is_Upset'] = ((predictions == 0) & (results['Seed1'] < results['Seed2'])) | \
                         ((predictions == 1) & (results['Seed1'] > results['Seed2']))
                                             
    return results

def create_current_matchups(team_stats):
    current_stats = team_stats[team_stats['Season'] == 2025].copy()
    
    key_columns = ['Season', 'Mapped ESPN Team Name']
    stat_columns = []
    
    potential_stat_columns = [
        'Adjusted Offensive Efficiency', 'Adjusted Defensive Efficiency',
        'AdjEM', 'FG2Pct', 'FG3Pct', 'FTPct', 'Experience',
        'eFGPct', 'TOPct', 'ORPct', 'FTRate', 'Tempo',
        'BlockPct', 'StlRate'
    ]
    
    for col in potential_stat_columns:
        if col in team_stats.columns:
            stat_columns.append(col)
    
    champion_profile = {
        'Adjusted Offensive Efficiency': 120,
        'Adjusted Defensive Efficiency': 90,
        'AdjEM': 30,
        'FG2Pct': 0.55,
        'FG3Pct': 0.37,
        'FTPct': 0.75,
        'eFGPct': 0.56,
        'TOPct': 0.16,
        'ORPct': 0.33
    }
    
    for feature, value in champion_profile.items():
        if feature in current_stats.columns:
            std = current_stats[feature].std()
            if std > 0:
                current_stats[f'{feature}_ChampDiff'] = (current_stats[feature] - value) / std
    
    champ_diff_cols = [col for col in current_stats.columns if col.endswith('_ChampDiff')]
    if champ_diff_cols:
        current_stats['Champion_Similarity'] = -np.sqrt((current_stats[champ_diff_cols]**2).sum(axis=1))
    else:
        current_stats['Champion_Similarity'] = 0
    
    current_stats['Late_Season_Momentum'] = np.random.uniform(-5, 5, size=len(current_stats))
    current_stats['Tournament_Experience'] = np.random.uniform(0, 3, size=len(current_stats))
    
    if 'Adjusted Offensive Efficiency' in current_stats.columns and 'Adjusted Defensive Efficiency' in current_stats.columns:
        current_stats['Off_Def_Ratio'] = current_stats['Adjusted Offensive Efficiency'] / current_stats['Adjusted Defensive Efficiency']
    
    if 'Adjusted Offensive Efficiency' in current_stats.columns and 'Adjusted Defensive Efficiency' in current_stats.columns:
        current_stats['Balance_Score'] = -abs(current_stats['Adjusted Offensive Efficiency'] - 
                                          (200 - current_stats['Adjusted Defensive Efficiency']))
    
    if 'Post-Season Tournament' not in current_stats.columns:
        print("WARNING: 'Post-Season Tournament' column not found in current stats")
        print("Available columns:", current_stats.columns.tolist())
        tournament_teams = current_stats
    else:
        tournament_teams = current_stats[current_stats['Post-Season Tournament'] == 'March Madness']
    
    if 'Seed' not in tournament_teams.columns:
        print("WARNING: 'Seed' column not found in tournament teams")
        print("Available columns:", tournament_teams.columns.tolist())
        if 'Seed_x' in tournament_teams.columns:
            tournament_teams['Seed'] = tournament_teams['Seed_x']
        else:
            print("ERROR: Cannot create matchups without seed information")
            return pd.DataFrame()
    
    if not pd.api.types.is_numeric_dtype(tournament_teams['Seed']):
        print(f"Converting Seed column from {tournament_teams['Seed'].dtype} to numeric")
        tournament_teams['Seed'] = pd.to_numeric(tournament_teams['Seed'], errors='coerce')
    
    current_matchups = []
    
    for region, group in tournament_teams.groupby('Region'):
        group_sorted = group.sort_values('Seed')
        seeds = sorted(group_sorted['Seed'].unique())
        
        print(f"Region {region} seeds: {seeds}")
        
        for i in range(len(seeds)//2):
            seed1 = seeds[i]
            seed2 = seeds[-(i+1)]
            
            team1_df = group_sorted[group_sorted['Seed'] == seed1]
            team2_df = group_sorted[group_sorted['Seed'] == seed2]
            
            if len(team1_df) == 0 or len(team2_df) == 0:
                print(f"Skipping matchup seed {seed1} vs {seed2} due to missing teams")
                continue
                
            team1 = team1_df.iloc[0]
            team2 = team2_df.iloc[0]
            
            matchup = {
                'Season': 2025,
                'Region': region,
                'Team1': team1['Mapped ESPN Team Name'],
                'Team2': team2['Mapped ESPN Team Name'],
                'Seed1': seed1,
                'Seed2': seed2,
                'Seed_Diff': seed1 - seed2
            }
            
            for stat in stat_columns:
                if stat in team1 and stat in team2:
                    matchup[f'{stat}_Diff'] = team1[stat] - team2[stat]
            
            if 'Champion_Similarity' in team1 and 'Champion_Similarity' in team2:
                matchup['Champion_Similarity_Diff'] = team1['Champion_Similarity'] - team2['Champion_Similarity']
            
            if 'Late_Season_Momentum' in team1 and 'Late_Season_Momentum' in team2:
                matchup['Momentum_Diff'] = team1['Late_Season_Momentum'] - team2['Late_Season_Momentum']
            
            if 'Tournament_Experience' in team1 and 'Tournament_Experience' in team2:
                matchup['Experience_Diff'] = team1['Tournament_Experience'] - team2['Tournament_Experience']
            
            if 'Off_Def_Ratio' in team1 and 'Off_Def_Ratio' in team2:
                matchup['Efficiency_Ratio'] = team1['Off_Def_Ratio'] / team2['Off_Def_Ratio']
            
            if 'Balance_Score' in team1 and 'Balance_Score' in team2:
                matchup['Balance_Diff'] = team1['Balance_Score'] - team2['Balance_Score']
            
            if 'AdjEM' in team1 and 'AdjEM' in team2:
                seed_weight = 0.1
                stats_weight = 0.9
                normalized_seed_diff = (seed1 - seed2) * 2
                
                matchup['Composite_Score'] = (
                    seed_weight * normalized_seed_diff + 
                    stats_weight * (team1['AdjEM'] - team2['AdjEM'])
                )
            
            current_matchups.append(matchup)
    
    current_matchups_df = pd.DataFrame(current_matchups)
    print(f"Number of 2025 matchups created: {len(current_matchups_df)}")
    
    return current_matchups_df

def simulate_full_tournament(model, scaler, features, first_round_matchups):
    import copy
    
    all_results = {
        'Round_1': predict_tournament(model, scaler, features, first_round_matchups),
        'Round_2': pd.DataFrame(),
        'Sweet_16': pd.DataFrame(),
        'Elite_8': pd.DataFrame(),
        'Final_4': pd.DataFrame(),
        'Championship': pd.DataFrame()
    }
    
    advancing_teams = {}
    
    team_stats = pd.read_csv('mad.csv')
    current_stats = team_stats[team_stats['Season'] == 2025].copy()
    
    round1_results = all_results['Round_1']
    
    for region in round1_results['Region'].unique():
        advancing_teams[region] = []
        regional_matchups = round1_results[round1_results['Region'] == region]
        
        for _, matchup in regional_matchups.iterrows():
            if matchup['Team1_Win_Probability'] >= 0.5:
                winner = {
                    'Team': matchup['Team1'],
                    'Seed': matchup['Seed1'],
                    'Stats': current_stats[current_stats['Mapped ESPN Team Name'] == matchup['Team1']]
                }
            else:
                winner = {
                    'Team': matchup['Team2'],
                    'Seed': matchup['Seed2'],
                    'Stats': current_stats[current_stats['Mapped ESPN Team Name'] == matchup['Team2']]
                }
                
            advancing_teams[region].append(winner)
    
    rounds = ['Round_2', 'Sweet_16', 'Elite_8', 'Final_4', 'Championship']
    regions_in_round = {
        'Round_2': ['East', 'West', 'South', 'Midwest'],
        'Sweet_16': ['East', 'West', 'South', 'Midwest'],
        'Elite_8': ['East', 'West', 'South', 'Midwest'],
        'Final_4': ['National Semifinal'],
        'Championship': ['National Final']
    }
    
    for round_name in rounds:
        current_round_matchups = []
        
        if round_name == 'Final_4':
            semifinal_1_team1 = advancing_teams['East'][0]
            semifinal_1_team2 = advancing_teams['West'][0]
            semifinal_2_team1 = advancing_teams['South'][0]
            semifinal_2_team2 = advancing_teams['Midwest'][0]
            
            matchup1 = create_matchup(semifinal_1_team1, semifinal_1_team2, 'National Semifinal')
            matchup2 = create_matchup(semifinal_2_team1, semifinal_2_team2, 'National Semifinal')
            
            current_round_matchups.extend([matchup1, matchup2])
            
            advancing_teams['National Semifinal'] = []
            
        elif round_name == 'Championship':
            team1 = advancing_teams['National Semifinal'][0]
            team2 = advancing_teams['National Semifinal'][1]
            
            championship_matchup = create_matchup(team1, team2, 'National Final')
            current_round_matchups.append(championship_matchup)
            
            advancing_teams['National Final'] = []
            
        else:
            for region in regions_in_round[round_name]:
                regional_winners = advancing_teams[region]
                
                num_matchups = len(regional_winners) // 2
                
                for i in range(num_matchups):
                    team1 = regional_winners[i]
                    team2 = regional_winners[-(i+1)]
                    
                    matchup = create_matchup(team1, team2, region)
                    current_round_matchups.append(matchup)
                
                advancing_teams[region] = []
        
        if current_round_matchups:
            current_round_df = pd.DataFrame(current_round_matchups)
            
            round_results = predict_tournament(model, scaler, features, current_round_df)
            all_results[round_name] = round_results
            
            for region in round_results['Region'].unique():
                if region not in advancing_teams:
                    advancing_teams[region] = []
                    
                regional_matchups = round_results[round_results['Region'] == region]
                
                for _, matchup in regional_matchups.iterrows():
                    if matchup['Team1_Win_Probability'] >= 0.5:
                        winner = {
                            'Team': matchup['Team1'],
                            'Seed': matchup['Seed1'],
                            'Stats': current_stats[current_stats['Mapped ESPN Team Name'] == matchup['Team1']]
                        }
                    else:
                        winner = {
                            'Team': matchup['Team2'],
                            'Seed': matchup['Seed2'],
                            'Stats': current_stats[current_stats['Mapped ESPN Team Name'] == matchup['Team2']]
                        }
                    
                    advancing_teams[region].append(winner)
    
    return all_results

def create_matchup(team1_info, team2_info, region):
    """
    Create a matchup between two teams for prediction.
    """
    # Extract team information
    team1_name = team1_info['Team']
    team2_name = team2_info['Team']
    seed1 = team1_info['Seed']
    seed2 = team2_info['Seed']
    
    # Get team stats
    team1_stats = team1_info['Stats'].iloc[0] if len(team1_info['Stats']) > 0 else {}
    team2_stats = team2_info['Stats'].iloc[0] if len(team2_info['Stats']) > 0 else {}
    
    # Create the base matchup
    matchup = {
        'Season': 2025,
        'Region': region,
        'Team1': team1_name,
        'Team2': team2_name,
        'Seed1': seed1,
        'Seed2': seed2,
        'Seed_Diff': seed1 - seed2
    }
    
    potential_stat_columns = ['Adjusted Offensive Efficiency', 'Adjusted Defensive Efficiency', 'AdjEM', 'FG2Pct', 'FG3Pct', 'FTPct', 'Experience', 'eFGPct', 'TOPct', 'ORPct', 'FTRate', 'Tempo', 'BlockPct', 'StlRate']
    
    for stat in potential_stat_columns:
        if stat in team1_stats and stat in team2_stats:
            matchup[f'{stat}_Diff'] = team1_stats[stat] - team2_stats[stat]
    
    if 'Champion_Similarity' in team1_stats and 'Champion_Similarity' in team2_stats:
        matchup['Champion_Similarity_Diff'] = team1_stats['Champion_Similarity'] - team2_stats['Champion_Similarity']
    
    if 'Late_Season_Momentum' in team1_stats and 'Late_Season_Momentum' in team2_stats:
        matchup['Momentum_Diff'] = team1_stats['Late_Season_Momentum'] - team2_stats['Late_Season_Momentum']
    
    if 'Tournament_Experience' in team1_stats and 'Tournament_Experience' in team2_stats:
        matchup['Experience_Diff'] = team1_stats['Tournament_Experience'] - team2_stats['Tournament_Experience']
    
    if 'Off_Def_Ratio' in team1_stats and 'Off_Def_Ratio' in team2_stats:
        matchup['Efficiency_Ratio'] = team1_stats['Off_Def_Ratio'] / team2_stats['Off_Def_Ratio']
    
    if 'Balance_Score' in team1_stats and 'Balance_Score' in team2_stats:
        matchup['Balance_Diff'] = team1_stats['Balance_Score'] - team2_stats['Balance_Score']
    
    if 'AdjEM' in team1_stats and 'AdjEM' in team2_stats:
        seed_weight = 0.1
        stats_weight = 0.9
        normalized_seed_diff = (seed1 - seed2) * 2
        
        matchup['Composite_Score'] = (
            seed_weight * normalized_seed_diff + 
            stats_weight * (team1_stats['AdjEM'] - team2_stats['AdjEM'])
        )
    
    return matchup

def print_ascii_bracket(all_results):
    print("\n========== 2025 NCAA TOURNAMENT BRACKET ==========\n")
    
    round_displays = {
        'Round_1': 'FIRST ROUND',
        'Round_2': 'SECOND ROUND', 
        'Sweet_16': 'SWEET 16',
        'Elite_8': 'ELITE 8',
        'Final_4': 'FINAL FOUR',
        'Championship': 'NATIONAL CHAMPIONSHIP'
    }
    
    regions = ['East', 'West', 'South', 'Midwest']
    
    for region in regions:
        print(f"\n{'=' * 20} {region.upper()} REGION {'=' * 20}")
        
        for round_name in ['Round_1', 'Round_2', 'Sweet_16', 'Elite_8']:
            if round_name in all_results and not all_results[round_name].empty:
                regional_results = all_results[round_name][all_results[round_name]['Region'] == region]
                
                if len(regional_results) > 0:
                    print(f"\n{round_displays[round_name]}:")
                    
                    for _, matchup in regional_results.iterrows():
                        if matchup['Team1_Win_Probability'] >= 0.5:
                            winner = matchup['Team1']
                            winner_seed = int(matchup['Seed1'])
                            loser = matchup['Team2']
                            loser_seed = int(matchup['Seed2'])
                            prob = matchup['Team1_Win_Probability']
                        else:
                            winner = matchup['Team2']
                            winner_seed = int(matchup['Seed2'])
                            loser = matchup['Team1']
                            loser_seed = int(matchup['Seed1'])
                            prob = 1 - matchup['Team1_Win_Probability']
                        
                        upset = winner_seed > loser_seed
                        upset_mark = " 🚨 UPSET!" if upset else ""
                        
                        print(f"({winner_seed}) {winner} def. ({loser_seed}) {loser} ({prob:.0%}){upset_mark}")
    
    print(f"\n{'=' * 20} FINAL FOUR {'=' * 20}")
    
    if 'Final_4' in all_results and not all_results['Final_4'].empty:
        for _, matchup in all_results['Final_4'].iterrows():
            if matchup['Team1_Win_Probability'] >= 0.5:
                winner = matchup['Team1']
                winner_seed = int(matchup['Seed1'])
                loser = matchup['Team2']
                loser_seed = int(matchup['Seed2'])
                prob = matchup['Team1_Win_Probability']
            else:
                winner = matchup['Team2']
                winner_seed = int(matchup['Seed2'])
                loser = matchup['Team1']
                loser_seed = int(matchup['Seed1'])
                prob = 1 - matchup['Team1_Win_Probability']
            
            upset = winner_seed > loser_seed
            upset_mark = " 🚨 UPSET!" if upset else ""
            
            print(f"({winner_seed}) {winner} def. ({loser_seed}) {loser} ({prob:.0%}){upset_mark}")
    
    print(f"\n{'=' * 20} NATIONAL CHAMPIONSHIP {'=' * 20}")
    
    if 'Championship' in all_results and not all_results['Championship'].empty:
        matchup = all_results['Championship'].iloc[0]
        
        if matchup['Team1_Win_Probability'] >= 0.5:
            winner = matchup['Team1']
            winner_seed = int(matchup['Seed1'])
            loser = matchup['Team2']
            loser_seed = int(matchup['Seed2'])
            prob = matchup['Team1_Win_Probability']
        else:
            winner = matchup['Team2']
            winner_seed = int(matchup['Seed2'])
            loser = matchup['Team1']
            loser_seed = int(matchup['Seed1'])
            prob = 1 - matchup['Team1_Win_Probability']
        
        upset = winner_seed > loser_seed
        upset_mark = " 🚨 UPSET!" if upset else ""
        
        print(f"({winner_seed}) {winner} def. ({loser_seed}) {loser} ({prob:.0%}){upset_mark}")
        print(f"\n🏆 NATIONAL CHAMPION: ({winner_seed}) {winner} 🏆")

if __name__ == "__main__":
    print("Loading and preparing data with enhanced basketball statistics...")
    try:
        matchups_df = load_and_prepare_data()
        
        print("\nTraining Random Forest model optimized for upset prediction...")
        model, scaler, features = train_random_forest(matchups_df)
        
        print("\nCreating current tournament matchups...")
        team_stats = pd.read_csv('mad.csv')
        current_matchups = create_current_matchups(team_stats)
        
        if len(current_matchups) > 0:
            print("\nPredicting tournament outcomes with Random Forest model...")
            results = predict_tournament(model, scaler, features, current_matchups)
            
            upset_count = results['Is_Upset'].sum()
            total_games = len(results)
            print(f"\nPredicted {upset_count} upsets out of {total_games} games ({upset_count/total_games:.1%})")
            
            print("\nTournament Predictions:")
            for _, row in results.iterrows():
                prob = row['Team1_Win_Probability']
                upset_tag = " 🚨 UPSET!" if row['Is_Upset'] else ""
                
                if prob >= 0.5:
                    print(f"{row['Region']} Region: {row['Team1']} ({row['Seed1']}) vs {row['Team2']} ({row['Seed2']}) - {row['Team1']} wins ({prob:.2%}){upset_tag}")
                else:
                    print(f"{row['Region']} Region: {row['Team1']} ({row['Seed1']}) vs {row['Team2']} ({row['Seed2']}) - {row['Team2']} wins ({1-prob:.2%}){upset_tag}")
            
            if upset_count > 0:
                print("\nPredicted Upsets:")
                upset_rows = results[results['Is_Upset']]
                for _, row in upset_rows.iterrows():
                    prob = row['Team1_Win_Probability']
                    if prob >= 0.5:
                        print(f"{row['Region']}: {row['Seed1']} seed {row['Team1']} over {row['Seed2']} seed {row['Team2']} ({prob:.2%})")
                    else:
                        print(f"{row['Region']}: {row['Seed2']} seed {row['Team2']} over {row['Seed1']} seed {row['Team1']} ({1-prob:.2%})")
            
            results.to_csv('2025_tournament_predictions_rf.csv', index=False)
            print("\nPredictions saved to 2025_tournament_predictions_rf.csv")
            
            # Simulate the entire tournament
            print("\nSimulating the entire tournament...")
            all_results = simulate_full_tournament(model, scaler, features, current_matchups)
            
            # Print bracket in ASCII format
            print_ascii_bracket(all_results)
            
            # Save results
            for round_name, results in all_results.items():
                if not results.empty:
                    results.to_csv(f'2025_tournament_{round_name}.csv', index=False)
            
            print("\nFull tournament simulation complete!")
            print("Results for each round have been saved to CSV files.")
            
        else:
            print("ERROR: No current matchups could be created. Cannot make predictions.")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
