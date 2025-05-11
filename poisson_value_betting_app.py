import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Value Betting Dashboard", layout="centered")
st.title("Value Betting Dashboard — Full Version")

leagues = {
    "Premier League": {"code": "PL", "odds_key": "soccer_epl"},
    "La Liga": {"code": "PD", "odds_key": "soccer_spain_la_liga"},
    "Serie A": {"code": "SA", "odds_key": "soccer_italy_serie_a"},
    "Bundesliga": {"code": "BL1", "odds_key": "soccer_germany_bundesliga"},
    "Ligue 1": {"code": "FL1", "odds_key": "soccer_france_ligue_one"}
}

ODDS_API_KEY = "YOUR_ODDS_API_KEY"
RESULTS_API_KEY = "YOUR_RESULTS_API_KEY"

selected = st.sidebar.selectbox("Choose League", list(leagues.keys()))
league_code = leagues[selected]["code"]
odds_key = leagues[selected]["odds_key"]

st.sidebar.markdown("Set filters below:")
min_ev = st.sidebar.slider("Minimum EV", -1.0, 1.0, 0.05, 0.01)
conf_filter = st.sidebar.selectbox("Confidence Level", ["All", "High", "Medium", "Low"])

@st.cache_data(ttl=86400)
def load_results(code):
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?season=2024&status=FINISHED"
    headers = {"X-Auth-Token": RESULTS_API_KEY}
    r = requests.get(url, headers=headers)
    matches = r.json().get("matches", [])
    data = []
    for m in matches:
        if m["score"]["fullTime"]["home"] is not None:
            data.append({
                "Home Team": m["homeTeam"]["name"],
                "Away Team": m["awayTeam"]["name"],
                "xG Home": np.random.uniform(1.1, 2.3),
                "xG Away": np.random.uniform(0.9, 2.0)
            })
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def load_odds(odds_key):
    url = f"https://api.the-odds-api.com/v4/sports/{odds_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return pd.DataFrame()
    matches = []
    for match in r.json():
        if not match.get("bookmakers"):
            continue
        home = match["home_team"]
        away = match["away_team"]
        odds = match["bookmakers"][0]["markets"][0]["outcomes"]
        home_odds = draw_odds = away_odds = None
        for o in odds:
            if o["name"] == home:
                home_odds = o["price"]
            elif o["name"] == away:
                away_odds = o["price"]
            elif o["name"].lower() == "draw":
                draw_odds = o["price"]
        if home_odds and away_odds:
            matches.append({
                "Home Team": home,
                "Away Team": away,
                "Home Odds": home_odds,
                "Draw Odds": draw_odds or 3.2,
                "Away Odds": away_odds
            })
    return pd.DataFrame(matches)

results = load_results(league_code)
odds = load_odds(odds_key)

avg_home_xg = results["xG Home"].mean() if 'xG Home' in results.columns else 1.5
avg_away_xg = results["xG Away"].mean() if 'xG Away' in results.columns else 1.3

teams = pd.unique(results[["Home Team", "Away Team"]].values.ravel())
stats = []
for team in teams:
    home = results[results["Home Team"] == team]
    away = results[results["Away Team"] == team]
    stats.append({
        "Team": team,
        "Home Attack": home["xG Home"].mean() / avg_home_xg if not home.empty else 1,
        "Home Defense": home["xG Away"].mean() / avg_away_xg if not home.empty else 1,
        "Away Attack": away["xG Away"].mean() / avg_away_xg if not away.empty else 1,
        "Away Defense": away["xG Home"].mean() / avg_home_xg if not away.empty else 1,
        "Data Points": len(home) + len(away)
    })

team_stats = pd.DataFrame(stats)
fallback = {"home": 0.45, "draw": 0.28, "away": 0.27}

def predict_poisson(ht, at):
    h = team_stats[team_stats["Team"] == ht]
    a = team_stats[team_stats["Team"] == at]
    if h.empty or a.empty:
        return fallback["home"], fallback["draw"], fallback["away"], "Low"
    mu_h = h["Home Attack"].values[0] * a["Away Defense"].values[0] * avg_home_xg
    mu_a = a["Away Attack"].values[0] * h["Home Defense"].values[0] * avg_away_xg
    matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            matrix[i][j] = poisson.pmf(i, mu_h) * poisson.pmf(j, mu_a)
    dp = int(h["Data Points"].values[0] + a["Data Points"].values[0])
    conf = "High" if dp >= 30 else "Medium" if dp >= 15 else "Low"
    return round(np.sum(np.tril(matrix, -1)), 3), round(np.sum(np.diag(matrix)), 3), round(np.sum(np.triu(matrix, 1)), 3), conf

if odds.empty:
    st.warning("No odds available.")
else:
    for _, row in odds.iterrows():
        h, d, a, conf = predict_poisson(row["Home Team"], row["Away Team"])
        if conf_filter != "All" and conf != conf_filter:
            continue
        ev_h = round((h * row["Home Odds"]) - 1, 3)
        ev_d = round((d * row["Draw Odds"]) - 1, 3)
        ev_a = round((a * row["Away Odds"]) - 1, 3)
        best_ev = max(ev_h, ev_d, ev_a)
        best_bet = ["Home", "Draw", "Away"][np.argmax([ev_h, ev_d, ev_a])] if best_ev > min_ev else "No Value"
        st.subheader(f"{row['Home Team']} vs {row['Away Team']}")
        st.write(f"Probabilities — Home: {h}, Draw: {d}, Away: {a}")
        st.write(f"EV — Home: {ev_h}, Draw: {ev_d}, Away: {ev_a}")
        st.write(f"**Best Bet:** {best_bet} | **Confidence:** {conf}")
        st.markdown("---")