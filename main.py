#this is a data science project for predicting the probability of a football match result by analysing 
# Team,Opponent, Home-Away, Time, team's Tactics based on the previous perfomnace of each team

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# Dictionary mapping full team names to short forms
team_short_forms = {
    'Arsenal': 'ARS',
    'Chelsea': 'CHE',
    'Liverpool': 'LIV',
    'Manchester United': 'MUN',
    'Real Madrid': 'RM',
    'Barcelona': 'BAR',
    'Bayern Munich': 'BM',
    'Borussia Dortmund': 'BVB',
    'Paris Saint-Germain': 'PSG',
    'Marseille': 'MAR',
    'Juventus': 'JUV',
    'AC Milan': 'ACM'
}

model = joblib.load('football_model.joblib')

df = pd.read_csv('football.csv')
df.columns = df.columns.str.strip().str.lower()

def label_encode_input(input_data, label_encoder):
    encoded_data = []
    for value in input_data:
        encoded_value = label_encoder.transform([value])[0]
        encoded_data.append(encoded_value)
    return encoded_data

teams = df['team'].unique()
opponents = df['opponent'].unique()
home_away_options = df['home_away'].unique()
day_night_options = df['day_night'].unique()
tactics_options = df['tactics'].unique()

team = input(f"Select a team ({', '.join(teams)}): ")
opponent = input(f"Select an opponent ({', '.join(opponents)}): ")
home_away = input(f"Select 'Home' or 'Away' ({', '.join(home_away_options)}): ")
day_night = input(f"Select 'Day' or 'Night' ({', '.join(day_night_options)}): ")
tactics = input(f"Select 'Offensive', 'Defensive', or 'Balanced' ({', '.join(tactics_options)}): ")

# Replace full team name with short form
team_short_form = team_short_forms.get(team, team)

label_encoder_team = LabelEncoder()
label_encoder_opponent = LabelEncoder()
label_encoder_home_away = LabelEncoder()
label_encoder_day_night = LabelEncoder()
label_encoder_tactics = LabelEncoder()
label_encoder_team.fit(df['team'])
label_encoder_opponent.fit(df['opponent'])
label_encoder_home_away.fit(df['home_away'])
label_encoder_day_night.fit(df['day_night'])
label_encoder_tactics.fit(df['tactics'])

team_encoded = label_encoder_team.transform([team])[0]
opponent_encoded = label_encoder_opponent.transform([opponent])[0]
home_away_encoded = label_encoder_home_away.transform([home_away])[0]
day_night_encoded = label_encoder_day_night.transform([day_night])[0]
tactics_encoded = label_encoder_tactics.transform([tactics])[0]

input_data = [[team_encoded, opponent_encoded, home_away_encoded, day_night_encoded, tactics_encoded]]
result_probabilities = model.predict_proba(input_data)[0]

label_encoder_result = LabelEncoder()
label_encoder_result.fit(df['result'])
result_classes = label_encoder_result.classes_
result_prediction = model.predict(input_data)[0]
result_prediction_decoded = label_encoder_result.inverse_transform([result_prediction])[0]

confidence_percentage = max(result_probabilities) * 100
confidence_class = result_classes[result_probabilities.argmax()]

print(f"The predicted result for the match is: {result_prediction_decoded}")
print(f"Confidence: {confidence_percentage:.2f}% for {confidence_class}")

# Analyze and suggest best tactics for winning
filtered_matches = df[(df['opponent'] == opponent) & (df['home_away'] == home_away) & (df['day_night'] == day_night)]
won_matches = filtered_matches[filtered_matches['result'] == 'Win']

if not won_matches.empty:
    best_tactics = won_matches['tactics'].mode().iloc[0]
    print(f"\nAnalyzing past matches with the same conditions, the best tactics for winning against {opponent} at {'home' if home_away == 'Home' else 'away'} during {day_night} are: {best_tactics}")
else:
    print("\nNo past matches found with the same conditions.")

# Create a string with input parameters as bullet points
input_bullet_points = f"Input Parameters:\n\
- Team: {team}\n\
- Opponent: {opponent}\n\
- Home/Away: {home_away}\n\
- Day/Night: {day_night}\n\
- Tactics: {tactics}"

# Plotting Pie Chart for result probabilities
plt.figure(figsize=(8, 6))
plt.pie(result_probabilities, labels=result_classes, autopct='%1.1f%%', startangle=140)
plt.title('Result Probabilities')
plt.suptitle('AC Milan(Away) VS Juventus(Home)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
