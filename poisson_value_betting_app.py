import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Value Betting Dashboard", layout="centered")
st.title("Value Betting Dashboard — Best Odds Fixed")

leagues = {
    "Premier League": {"code": "PL", "odds_key": "soccer_epl"},
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
        st.error("Failed to fetch odds.")
        return pd.DataFrame()
    data = r.json()
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

odds_df = load_odds_best(odds_key)
if odds_df.empty:
    st.warning("No upcoming odds available.")
else:
    st.dataframe(odds_df)