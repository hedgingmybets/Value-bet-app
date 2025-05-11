import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Value Bets with Real Odds", layout="centered")
st.title("Football Value Betting (Live Odds + Poisson Model)")

# Odds API Key
API_KEY = "8ce16d805de3ae1a3bb23670a86ea37f"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

# Get live odds from The Odds API
def fetch_odds():
    params = {
        "apiKey": API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    response = requests.get(ODDS_URL, params=params)
    if response.status_code != 200:
        st.error("Failed to fetch odds data.")
        return pd.DataFrame()

    data = response.json()
    matches = []
    for match in data:
        if "bookmakers" not in match or not match["bookmakers"]:
            continue
        bookmaker = match["bookmakers"][0]
        odds = bookmaker["markets"][0]["outcomes"]
        home_team = odds[0]["name"]
        away_team = odds[1]["name"]
        home_odds = odds[0]["price"]
        away_odds = odds[1]["price"]
        draw_odds = 3.2 + np.random.rand()  # placeholder for draw
        matches.append({
            "Home Team": home_team,
            "Away Team": away_team,
            "Home Odds": home_odds,
            "Draw Odds": draw_odds,
            "Away Odds": away_odds
        })
    return pd.DataFrame(matches)

matches = fetch_odds()

# Minimal historical data
historical = pd.DataFrame({
    "Home Team": ["Man City", "Liverpool", "Arsenal"],
    "Away Team": ["Chelsea", "Arsenal", "Liverpool"],
    "Home Goals": [2, 2, 1],
    "Away Goals": [1, 1, 2]
})

avg_home_goals = historical["Home Goals"].mean()
avg_away_goals = historical["Away Goals"].mean()

teams = pd.unique(historical[["Home Team", "Away Team"]].values.ravel())
team_stats = []

for team in teams:
    home = historical[historical["Home Team"] == team]
    away = historical[historical["Away Team"] == team]
    team_stats.append({
        "Team": team,
        "Home Attack": home["Home Goals"].mean() / avg_home_goals if not home.empty else 1,
        "Home Defense": home["Away Goals"].mean() / avg_away_goals if not home.empty else 1,
        "Away Attack": away["Away Goals"].mean() / avg_away_goals if not away.empty else 1,
        "Away Defense": away["Home Goals"].mean() / avg_home_goals if not away.empty else 1,
    })

team_stats = pd.DataFrame(team_stats)

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

# Display odds and value bets
st.write("### Upcoming Premier League Matches")
if matches.empty:
    st.warning("No matches available or API limit reached.")
else:
    for _, row in matches.iterrows():
        h, d, a = predict_poisson(row["Home Team"], row["Away Team"])
        ev_home = (h * row["Home Odds"]) - 1
        ev_draw = (d * row["Draw Odds"]) - 1
        ev_away = (a * row["Away Odds"]) - 1
        best_ev = max(ev_home, ev_draw, ev_away)
        best_bet = ["Home", "Draw", "Away"][np.argmax([ev_home, ev_draw, ev_away])] if best_ev > 0.05 else "No Value"

        with st.container():
            st.markdown(f"**{row['Home Team']} vs {row['Away Team']}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Home Win", f"{h:.2f}", f"EV: {ev_home:.2f}")
            col2.metric("Draw", f"{d:.2f}", f"EV: {ev_draw:.2f}")
            col3.metric("Away Win", f"{a:.2f}", f"EV: {ev_away:.2f}")
            st.markdown(f"**Best Bet:** `{best_bet}`")
            st.markdown("---")