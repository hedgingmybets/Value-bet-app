import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Value Betting Dashboard", layout="centered")
st.title("Value Betting Dashboard — Full Version with Predictions and EV")


leagues = {
    "Premier League": {"code": "PL", "odds_key": "soccer_epl"},
    "Championship": {"code": "ELC", "odds_key": "soccer_english_championship"},
    "League One": {"code": "EL1", "odds_key": "soccer_english_league_one"},
    "League Two": {"code": "EL2", "odds_key": "soccer_english_league_two"},
    "La Liga": {"code": "PD", "odds_key": "soccer_spain_la_liga"},
    "Serie A": {"code": "SA", "odds_key": "soccer_italy_serie_a"},
    "Bundesliga": {"code": "BL1", "odds_key": "soccer_germany_bundesliga"},
    "Ligue 1": {"code": "FL1", "odds_key": "soccer_france_ligue_one"}
}

ODDS_API_KEY = "8ce16d805de3ae1a3bb23670a86ea37f"
RESULTS_API_KEY = "015bbaf510cb464fb3accc3309783ccb"

selected = st.sidebar.selectbox("Choose League", list(leagues.keys()))
league_code = leagues[selected]["code"]
odds_key = leagues[selected]["odds_key"]

min_ev = st.sidebar.slider("Minimum EV Threshold", -1.0, 1.0, 0.05, 0.01)
conf_filter = st.sidebar.selectbox("Confidence Level", ["All", "High", "Medium", "Low"])

@st.cache_data(ttl=86400)
def load_results(code: str):
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?season=2024&status=FINISHED"
    headers = {"X-Auth-Token": RESULTS_API_KEY}
    r = requests.get(url, headers=headers)
    data = r.json().get("matches", [])
        st.write(f'Total matches found for {odds_key}:', len(data))
    results = []
    for match in data:
        if match["score"]["fullTime"]["home"] is not None:
            results.append({
                "Home Team": match["homeTeam"]["name"],
                "Away Team": match["awayTeam"]["name"],
                "xG Home": np.random.uniform(1.0, 2.4),
                "xG Away": np.random.uniform(0.8, 2.1)
            })
    return pd.DataFrame(results)

@st.cache_data(ttl=600)
def load_odds_best(odds_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{odds_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.write(f'Failed to load odds for {odds_key} — Status:', r.status_code)
        st.error("Failed to fetch odds.")
        return pd.DataFrame()
    data = r.json()
        st.write(f'Total matches found for {odds_key}:', len(data))
    matches = []
    for match in data:
        home = match.get("home_team")
        away = match.get("away_team")
        best_home, best_draw, best_away = 0, 0, 0
        for book in match.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome["name"].strip().lower()
                    if name == home.strip().lower():
                        best_home = max(best_home, outcome["price"])
                    elif name == away.strip().lower():
                        best_away = max(best_away, outcome["price"])
                    elif name in ["draw", "the draw"]:
                        best_draw = max(best_draw, outcome["price"])
        if best_home and best_draw and best_away:
            matches.append({
                "Home Team": home,
                "Away Team": away,
                "Home Odds": best_home,
                "Draw Odds": best_draw,
                "Away Odds": best_away,
                "Kickoff": match.get("commence_time", "N/A")
            })
    return pd.DataFrame(matches)

results = load_results(league_code)
odds_df = load_odds_best(odds_key)

avg_home_xg = results["xG Home"].mean()
avg_away_xg = results["xG Away"].mean()

teams = pd.unique(results[["Home Team", "Away Team"]].values.ravel())
team_stats = []
for team in teams:
    home = results[results["Home Team"] == team]
    away = results[results["Away Team"] == team]
    team_stats.append({
        "Team": team,
        "Home Attack": home["xG Home"].mean() / avg_home_xg if not home.empty else 1,
        "Home Defense": home["xG Away"].mean() / avg_away_xg if not home.empty else 1,
        "Away Attack": away["xG Away"].mean() / avg_away_xg if not away.empty else 1,
        "Away Defense": away["xG Home"].mean() / avg_home_xg if not away.empty else 1,
        "Data Points": len(home) + len(away)
    })

team_stats = pd.DataFrame(team_stats)
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

if odds_df.empty:
    st.warning("No upcoming odds available.")
else:
        st.write(f"### Upcoming Matches — {selected}")
    for _, row in odds_df.iterrows():
        h, d, a, conf = predict_poisson(row["Home Team"], row["Away Team"])
        if conf_filter != "All" and conf != conf_filter:
            continue
        implied_home = round(1 / row["Home Odds"], 3)
        implied_draw = round(1 / row["Draw Odds"], 3)
        implied_away = round(1 / row["Away Odds"], 3)
        ev_h = round((h * row["Home Odds"]) - 1, 3)
        ev_d = round((d * row["Draw Odds"]) - 1, 3)
        ev_a = round((a * row["Away Odds"]) - 1, 3)
        best = max(ev_h, ev_d, ev_a)
        best_bet = ["Home", "Draw", "Away"][np.argmax([ev_h, ev_d, ev_a])] if best > min_ev else "No Value"

        st.subheader(f"{row['Home Team']} vs {row['Away Team']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{h:.2f}", f"Impl: {implied_home:.2f}")
        col2.metric("Draw", f"{d:.2f}", f"Impl: {implied_draw:.2f}")
        col3.metric("Away Win", f"{a:.2f}", f"Impl: {implied_away:.2f}")
        st.markdown(f"**EV** — Home: `{ev_h}`, Draw: `{ev_d}`, Away: `{ev_a}`")
        st.markdown(f"**Best Bet:** `{best_bet}` | **Confidence:** {conf}")
        st.markdown("---") 