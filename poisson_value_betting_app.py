import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Multi-League Value Betting", layout="centered")
st.title("Auto-Updating Value Betting Model — Europe + England")

# League map
leagues = {
    "Premier League": {"code": "PL", "odds_key": "soccer_epl"},
    "Championship": {"code": "ELC", "odds_key": "soccer_efl_championship"},
    "League One": {"code": "EL1", "odds_key": "soccer_england_league1"},
    "League Two": {"code": "EL2", "odds_key": "soccer_england_league2"},
    "La Liga": {"code": "PD", "odds_key": "soccer_spain_la_liga"},
    "Serie A": {"code": "SA", "odds_key": "soccer_italy_serie_a"},
    "Bundesliga": {"code": "BL1", "odds_key": "soccer_germany_bundesliga"},
    "Ligue 1": {"code": "FL1", "odds_key": "soccer_france_ligue_one"}
}

ODDS_API_KEY = "8ce16d805de3ae1a3bb23670a86ea37f"
RESULTS_API_KEY = "015bbaf510cb464fb3accc3309783ccb"

team_name_map = {}  # Will build automatically as names are encountered

# User selects league
selected = st.selectbox("Choose a League", list(leagues.keys()))
league_code = leagues[selected]["code"]
odds_key = leagues[selected]["odds_key"]

@st.cache_data(ttl=86400)
def load_results(code: str):
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?season=2024&status=FINISHED"
    headers = {"X-Auth-Token": RESULTS_API_KEY}
    r = requests.get(url, headers=headers)
    data = r.json().get("matches", [])
    results = []
    for match in data:
        h_name = match["homeTeam"]["name"]
        a_name = match["awayTeam"]["name"]
        team_name_map[h_name] = h_name
        team_name_map[a_name] = a_name
        if match["score"]["fullTime"]["home"] is not None:
            results.append({
                "Home Team": h_name,
                "Away Team": a_name,
                "Home Goals": match["score"]["fullTime"]["home"],
                "Away Goals": match["score"]["fullTime"]["away"]
            })
    return pd.DataFrame(results)

@st.cache_data(ttl=600)
def load_odds():
    url = f"https://api.the-odds-api.com/v4/sports/{odds_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    r = requests.get(url, params=params)
    data = r.json()
    matches = []
    for match in data:
        if not match.get("bookmakers") or "home_team" not in match or "away_team" not in match:
            continue
        home_team = match["home_team"]
        away_team = match["away_team"]
        team_name_map[home_team] = home_team
        team_name_map[away_team] = away_team
        outcomes = match["bookmakers"][0]["markets"][0]["outcomes"]
        home_odds = away_odds = None
        for o in outcomes:
            name = o["name"]
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

# Load data
results = load_results(league_code)
odds = load_odds()

# Build team stats
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
    h = team_stats[team_stats["Team"] == ht]
    a = team_stats[team_stats["Team"] == at]
    if h.empty or a.empty:
        return fallback["home"], fallback["draw"], fallback["away"]
    mu_h = h["Home Attack"].values[0] * a["Away Defense"].values[0] * avg_home
    mu_a = a["Away Attack"].values[0] * h["Home Defense"].values[0] * avg_away
    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            matrix[i][j] = poisson.pmf(i, mu_h) * poisson.pmf(j, mu_a)
    return round(np.sum(np.tril(matrix, -1)), 3), round(np.sum(np.diag(matrix)), 3), round(np.sum(np.triu(matrix, 1)), 3)

# Show output
st.write(f"### Upcoming {selected} Matches")
if odds.empty:
    st.warning("No matches or odds available.")
else:
    for _, row in odds.iterrows():
        h, d, a = predict_poisson(row["Home Team"], row["Away Team"])
        ev_h = (h * row["Home Odds"]) - 1
        ev_d = (d * row["Draw Odds"]) - 1
        ev_a = (a * row["Away Odds"]) - 1
        best_ev = max(ev_h, ev_d, ev_a)
        best_bet = ["Home", "Draw", "Away"][np.argmax([ev_h, ev_d, ev_a])] if best_ev > 0.05 else "No Value"
        st.markdown(f"**{row['Home Team']} vs {row['Away Team']}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{h:.2f}", f"EV: {ev_h:.2f}")
        col2.metric("Draw", f"{d:.2f}", f"EV: {ev_d:.2f}")
        col3.metric("Away Win", f"{a:.2f}", f"EV: {ev_a:.2f}")
        st.markdown(f"**Best Bet:** `{best_bet}`")
        st.markdown("---")