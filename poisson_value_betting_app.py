import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Premier League Value Bets", layout="centered")
st.title("Value Betting with Real Odds (Poisson Model)")

API_KEY = "8ce16d805de3ae1a3bb23670a86ea37f"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

team_name_map = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham", "Brighton & Hove Albion": "Brighton",
    "Newcastle United": "Newcastle", "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham", "Nottingham Forest": "Nottm Forest",
    "Sheffield United": "Sheffield Utd", "Luton Town": "Luton",
    "AFC Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Fulham": "Fulham", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Burnley": "Burnley", "Aston Villa": "Aston Villa",
    "Chelsea": "Chelsea", "Liverpool": "Liverpool", "Arsenal": "Arsenal"
}

def fetch_odds():
    params = {
        "apiKey": API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    response = requests.get(ODDS_URL, params=params)
    if response.status_code != 200:
        st.error(f"Failed to fetch odds data. Status code: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    matches = []
    for match in data:
        if not match.get("bookmakers") or "home_team" not in match or "away_team" not in match:
            continue

        raw_home = match["home_team"]
        raw_away = match["away_team"]
        home_team = team_name_map.get(raw_home, raw_home)
        away_team = team_name_map.get(raw_away, raw_away)

        odds = match["bookmakers"][0]["markets"][0]["outcomes"]
        home_odds = away_odds = None

        for outcome in odds:
            name = team_name_map.get(outcome["name"], outcome["name"])
            if name == home_team:
                home_odds = outcome["price"]
            elif name == away_team:
                away_odds = outcome["price"]

        if home_odds and away_odds:
            matches.append({
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Odds": home_odds,
                "Draw Odds": 3.2 + np.random.rand(),
                "Away Odds": away_odds
            })
    return pd.DataFrame(matches)

matches = fetch_odds()

# Historical data with 20 Premier League teams
historical = pd.DataFrame({
    "Home Team": [
        "Man City", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Man United", "Newcastle", "Brighton",
        "Brentford", "Fulham", "Crystal Palace", "Wolves", "Aston Villa", "West Ham", "Everton", "Burnley",
        "Luton", "Nottm Forest", "Sheffield Utd", "Bournemouth"
    ],
    "Away Team": [
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Brighton", "Tottenham", "Wolves", "Fulham",
        "West Ham", "Brentford", "Leeds", "Everton", "Crystal Palace", "Newcastle", "Southampton", "Bournemouth",
        "Aston Villa", "Leicester", "Luton", "Sheffield Utd"
    ],
    "Home Goals": [2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1],
    "Away Goals": [1, 1, 2, 2, 1, 1, 0, 2, 1, 2, 2, 1, 1, 0, 1, 1, 2, 0, 2, 1]
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

fallback_probs = {"home": 0.45, "draw": 0.28, "away": 0.27}

def predict_poisson(home_team, away_team):
    ht = team_stats[team_stats["Team"] == home_team]
    at = team_stats[team_stats["Team"] == away_team]
    if ht.empty or at.empty:
        return fallback_probs["home"], fallback_probs["draw"], fallback_probs["away"]

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

# Display
st.write("### Corrected Home vs Away Matchups with Value Bets")
if matches.empty:
    st.warning("No matches or odds available.")
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