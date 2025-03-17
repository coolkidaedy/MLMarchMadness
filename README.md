# March Madness Prediction System - Results Summary

## Model Performance
- Overall Accuracy: 82.5% on test data
- Precision: 87.0%
- Recall: 93.1%
- F1-score: 89.9%
- Upset Prediction Accuracy: 26.3% on historical upsets

## Top Predictive Features
- Efficiency Ratio (7.5%)
- Adjusted Efficiency Margin Difference (7.2%)
- Composite Score (6.8%)
- Turnover Percentage Difference (6.3%)
- Offensive Efficiency Difference (6.1%)

## 2025 Tournament Predictions
- Total Games Analyzed: 32 first-round matchups
- Predicted Upsets: 4 games (12.5% of matchups)
- Favorite Win Rate: 87.5%
- Average Predicted Margin: 15.6 AdjEM points

## Notable Upset Predictions
- Midwest Region: 10-seed Utah State over 7-seed UCLA (51%)
- South Region: 12-seed UC San Diego over 5-seed Michigan (68%)
- West Region: 10-seed Arkansas over 7-seed Kansas (59%)
- West Region: 9-seed Oklahoma over 8-seed UConn (80%)

## Methodology
- Dataset includes statistics from 2010-2025 NCAA tournaments
- Random Forest classifier with optimized parameters
- Champion similarity metrics incorporated for team evaluation
- Reduced seed influence with 90% weight on statistical differences
- Basket-specific features tailored for tournament prediction
- Comprehensive statistical profile analysis beyond just seeding

The model successfully identifies potential upset candidates while maintaining reasonable accuracy, balancing the risk of picking too many upsets against being too conservative.
