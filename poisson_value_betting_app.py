import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Poisson Value Betting", layout="centered")
st.title("Football Value Betting (Poisson Model)")

# Mock upcoming matches
matches = pd.DataFrame({
    "Home Team": ["Man City", "Liverpool"],
    "Away Team": ["Arsenal", "Man United"],
    "Home Odds": [1.80, 2.10],
    "Draw Odds": [3.70, 3.50],
    "Away Odds": [4.20, 3.40]
})

# Minimal mock training data for demo
historical = pd.DataFrame({
    "Home Team": ["Man City", "Liverpool", "Arsenal"],
    "Away Team": ["Chelsea", "Arsenal", "Liverpool"],
    "Home Goals": [2, 2, 1],
    "Away Goals": [1, 1, 2]
})

# Calculate league average
avg_home_goals = historical["Home Goals"].mean()
avg_away_goals = historical["Away Goals"].mean()

# Team strength factors
teams = pd.unique(historical[["Home Team", "Away Team"]].values.ravel())
stats = []

for team in teams:
    home = historical[historical["Home Team"] == team]
    away = historical[historical["Away Team"] == team]
    stats.append({
        "Team": team,
        "Home Attack": home["Home Goals"].mean() / avg_home_goals if not home.empty else 1,
        "Home Defense": home["Away Goals"].mean() / avg_away_goals if not home.empty else 1,
        "Away Attack": away["Away Goals"].mean() / avg_away_goals if not away.empty else 1,
        "Away Defense": away["Home Goals"].mean() / avg_home_goals if not away.empty else 1,
    })

team_stats = pd.DataFrame(stats)

def predict_poisson(home_team, away_team):
    ht = team_stats[team_stats["Team"] == home_team]
    at = team_stats[team_stats["Team"] == away_team]
    if ht.empty or at.empty:
        return 0.33, 0.33, 0.33

    mu_home = ht["Home Attack"].values[0] * at["Away Defense"].values[0] * avg_home_goals
    mu_away = at["Away Attack"].values[0] * ht["Home Defense"].values[0] * avg_away_goals

    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            matrix[i][j] = poisson.pmf(i, mu_home) * poisson.pmf(j, mu_away)

    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    return round(home_win, 3), round(draw, 3), round(away_win, 3)

results = []
for _, row in matches.iterrows():
    h, d, a = predict_poisson(row["Home Team"], row["Away Team"])
    ev_home = (h * row["Home Odds"]) - 1
    ev_draw = (d * row["Draw Odds"]) - 1
    ev_away = (a * row["Away Odds"]) - 1
    best = max(ev_home, ev_draw, ev_away)
    best_bet = ["Home", "Draw", "Away"][np.argmax([ev_home, ev_draw, ev_away])] if best > 0.05 else "No Value"
    results.append({
        "Match": f"{row['Home Team']} vs {row['Away Team']}",
        "Home Win Prob": h,
        "Draw Prob": d,
        "Away Win Prob": a,
        "EV Home": round(ev_home, 3),
        "EV Draw": round(ev_draw, 3),
        "EV Away": round(ev_away, 3),
        "Best Bet": best_bet
    })

df = pd.DataFrame(results)
st.dataframe(df)
