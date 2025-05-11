
import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Auto-Updating Poisson Betting Model", layout="centered")
st.title("Premier League Value Betting — Auto Updating")

# API keys
ODDS_API_KEY = "8ce16d805de3ae1a3bb23670a86ea37f"
RESULTS_API_KEY = "015bbaf510cb464fb3accc3309783ccb"
RESULTS_URL = "https://api.football-data.org/v4/competitions/PL/matches?season=2024&status=FINISHED"
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

@st.cache_data(ttl=86400)
def load_results():
    headers = {"X-Auth-Token": RESULTS_API_KEY}
    r = requests.get(RESULTS_URL, headers=headers)
    data = r.json().get("matches", [])
    rows = []
    for match in data:
        if match["score"]["fullTime"]["home"] is not None:
            home = team_name_map.get(match["homeTeam"]["name"], match["homeTeam"]["name"])
            away = team_name_map.get(match["awayTeam"]["name"], match["awayTeam"]["name"])
            rows.append({
                "Home Team": home,
                "Away Team": away,
                "Home Goals": match["score"]["fullTime"]["home"],
                "Away Goals": match["score"]["fullTime"]["away"]
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=900)
def load_odds():
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    r = requests.get(ODDS_URL, params=params)
    data = r.json()
    matches = []
    for match in data:
        if not match.get("bookmakers") or "home_team" not in match or "away_team" not in match:
            continue
        home_team = team_name_map.get(match["home_team"], match["home_team"])
        away_team = team_name_map.get(match["away_team"], match["away_team"])
        outcomes = match["bookmakers"][0]["markets"][0]["outcomes"]
        home_odds = away_odds = None
        for o in outcomes:
            name = team_name_map.get(o["name"], o["name"])
            if name == home_team:
                home_odds = o["price"]
            elif name == away_team:
                away_odds = o["price"]
        if home_odds and away_odds:
            matches.append({
                "Home Team": home_team,
                "Away Team": away_team,
                "Home Odds": home_odds,
                "Draw Odds": 3.2 + np.random.rand(),
                "Away Odds": away_odds
            })
    return pd.DataFrame(matches)

results = load_results()
odds = load_odds()

avg_home = results["Home Goals"].mean()
avg_away = results["Away Goals"].mean()

teams = pd.unique(results[["Home Team", "Away Team"]].values.ravel())
stats = []

for team in teams:
    home = results[results["Home Team"] == team]
    away = results[results["Away Team"] == team]
    stats.append({
        "Team": team,
        "Home Attack": home["Home Goals"].mean() / avg_home if not home.empty else 1,
        "Home Defense": home["Away Goals"].mean() / avg_away if not home.empty else 1,
        "Away Attack": away["Away Goals"].mean() / avg_away if not away.empty else 1,
        "Away Defense": away["Home Goals"].mean() / avg_home if not away.empty else 1,
    })

team_stats = pd.DataFrame(stats)

fallback = {"home": 0.45, "draw": 0.28, "away": 0.27}
def predict_poisson(ht, at):
    h_row = team_stats[team_stats["Team"] == ht]
    a_row = team_stats[team_stats["Team"] == at]
    if h_row.empty or a_row.empty:
        return fallback["home"], fallback["draw"], fallback["away"]
    mu_h = h_row["Home Attack"].values[0] * a_row["Away Defense"].values[0] * avg_home
    mu_a = a_row["Away Attack"].values[0] * h_row["Home Defense"].values[0] * avg_away
    matrix = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            matrix[i][j] = poisson.pmf(i, mu_h) * poisson.pmf(j, mu_a)
    return round(np.sum(np.tril(matrix, -1)),3), round(np.sum(np.diag(matrix)),3), round(np.sum(np.triu(matrix,1)),3)

st.write("### Auto-Updating Match Predictions and Value Bets")
if odds.empty:
    st.warning("No upcoming matches found.")
else:
    for _, row in odds.iterrows():
        h, d, a = predict_poisson(row["Home Team"], row["Away Team"])
        ev_h = (h * row["Home Odds"]) - 1
        ev_d = (d * row["Draw Odds"]) - 1
        ev_a = (a * row["Away Odds"]) - 1
        best = max(ev_h, ev_d, ev_a)
        best_bet = ["Home", "Draw", "Away"][np.argmax([ev_h, ev_d, ev_a])] if best > 0.05 else "No Value"
        st.markdown(f"**{row['Home Team']} vs {row['Away Team']}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{h:.2f}", f"EV: {ev_h:.2f}")
        col2.metric("Draw", f"{d:.2f}", f"EV: {ev_d:.2f}")
        col3.metric("Away Win", f"{a:.2f}", f"EV: {ev_a:.2f}")
        st.markdown(f"**Best Bet:** `{best_bet}`")
        st.markdown("---")