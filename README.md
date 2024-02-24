# Football Match Result Prediction

## Overview
This project utilizes data analytics and logistic regression classification techniques to predict the outcomes of football matches based on various parameters such as team performance, opponent, match location, time of day, and team tactics. The aim is to develop a predictive model that can assist in understanding the potential results of football matches given specific conditions.The ouput of the code gives the result of team.

## Dataset
The dataset used in this project contains historical data of football matches including details such as team names, opponents, match locations (home/away), match times (day/night), team tactics, and match results (win/loss/draw).
[football.csv](https://github.com/cmshebeeb/LR_Football/blob/main/football.csv)

## Methodology
- **Data Preprocessing:** The dataset is preprocessed to handle missing values, encode categorical variables, and prepare the data for analysis.
- **Model Training:** A logistic regression classifier is trained using the preprocessed data to predict the outcomes of football matches.
- **Prediction:** Given input parameters including the team, opponent, match location, time of day, and tactics, the trained model predicts the most likely outcome of the match along with a confidence level.
- **Tactical Analysis:** Additionally, the project includes an analysis of past matches with similar conditions to suggest the best tactics for maximizing the chances of winning against a specific opponent under similar circumstances.

## Usage
To utilize the prediction functionality:
1. Select the team and opponent for the upcoming match, choosing from the available options.
2. Specify whether the match is at home or away, and whether it is during the day or night.
3. Select the preferred team tactics for the match.
4. The model will provide the predicted result of the match along with the confidence level, and suggest the best tactics based on past matches with similar conditions.

## Files
- **football_model.joblib:** Pre-trained logistic regression model for match result prediction.
- **football.csv:** Dataset containing historical football match data.
- **main.py:** Python script for predicting match results and suggesting tactics based on user input.
- **README.md:** This file, providing an overview of the project and instructions for usage.

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- joblib
- matplotlib

## Author
- [Muhammed Shebeeb C](https://github.com/cmshebeeb)
- [Alfred Shyjo](https://github.com/Alfredshyjo)
- [Sooraj Sajeev](https://github.com/SoorajSajeev2156)
## Sample Input
- Select a team (Real Madrid, Barcelona, Manchester United, Liverpool, Bayern Munich, Borussia Dortmund, Paris Saint-Germain, Marseille, Juventus, AC Milan): AC Milan
- Select an opponent (Barcelona, Real Madrid, Liverpool, Manchester United, Borussia Dortmund, Bayern Munich, Marseille, Paris Saint-Germain, AC Milan, Juventus): Juventus
- Select 'Home' or 'Away' (Home, Away): Away
- Select 'Day' or 'Night' (Night, Day): Day
- Select 'Offensive', 'Defensive', or 'Balanced' (Balanced, Offensive, Defensive): Balanced

## Sample Output
- The predicted result for the match is: Loss
- Confidence: 59.92% for Loss


![AC Milan VS Juventus](https://github.com/cmshebeeb/LR_Football/assets/96789111/d9deffc4-50ab-4d83-a3f5-e85ac6f0746f)

