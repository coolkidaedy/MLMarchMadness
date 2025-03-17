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
            
            plt.figure(figsize=(14, 20))
            
            regions = results['Region'].unique()
            region_positions = {region: i for i, region in enumerate(regions)}
            
            for region, group in results.groupby('Region'):
                pos = region_positions[region]
                plt.subplot(2, 2, pos+1)
                
                group = group.sort_values('Seed1')
                
                labels = []
                colors = []
                
                for _, row in group.iterrows():
                    if row['Team1_Win_Probability'] >= 0.5:
                        winner = f"{row['Team1']} ({row['Seed1']})"
                        prob = row['Team1_Win_Probability']
                        is_upset = row['Seed1'] > row['Seed2']
                    else:
                        winner = f"{row['Team2']} ({row['Seed2']})"
                        prob = 1 - row['Team1_Win_Probability']
                        is_upset = row['Seed2'] > row['Seed1']
                    
                    labels.append(f"{winner} {prob:.0%}")
                    
                    if is_upset:
                        colors.append((1.0, 0.4 - 0.4 * prob, 0.4 - 0.4 * prob))
                    else:
                        colors.append((0.4 - 0.4 * prob, 0.4 - 0.4 * prob, 1.0))
                
                y_pos = range(len(labels))
                plt.barh(y_pos, [1] * len(labels), color=colors)
                
                plt.yticks(y_pos, labels)
                plt.title(f"{region} Region")
                plt.xlim(0, 1.1)
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                
                if any(row['Is_Upset'] for _, row in group.iterrows()):
                    plt.plot([], [], color='red', label='Upset')
                    plt.plot([], [], color='blue', label='Expected')
                    plt.legend(loc='lower right')
            
            plt.tight_layout()
            plt.savefig('2025_bracket_visualization.png')
            print("Bracket visualization saved to 2025_bracket_visualization.png")
            
            favorite_win_rate = len(results[~results['Is_Upset']]) / len(results)
            print(f"\nFavorite win rate: {favorite_win_rate:.1%}")
            
            if 'AdjEM_Diff' in results.columns:
                avg_predicted_margin = results['AdjEM_Diff'].mean()
                print(f"Average predicted margin (AdjEM): {avg_predicted_margin:.1f}")
            
            if upset_count > 0:
                print("\nUpset analysis by seed matchup:")
                upset_by_seed = {}
                
                for _, row in results[results['Is_Upset']].iterrows():
                    seed1, seed2 = row['Seed1'], row['Seed2']
                    if row['Team1_Win_Probability'] >= 0.5:
                        key = f"{int(seed1)} over {int(seed2)}"
                    else:
                        key = f"{int(seed2)} over {int(seed1)}"
                    
                    upset_by_seed[key] = upset_by_seed.get(key, 0) + 1
                
                for matchup, count in sorted(upset_by_seed.items()):
                    print(f"  {matchup}: {count} occurrence(s)")
            
            print("\nThis model can be extended to simulate the entire tournament!")
            print("For a complete bracket prediction, you would:")
            print("1. Use these first round results to create second round matchups")
            print("2. Apply the same prediction process for each subsequent round")
            print("3. Continue until you have a predicted champion")
            
        else:
            print("ERROR: No current matchups could be created. Cannot make predictions.")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
