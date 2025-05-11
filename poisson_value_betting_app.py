import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Value Betting Dashboard", layout="centered")
st.title("Value Betting Dashboard — Fixed Odds Handling")

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
def load_odds(odds_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{odds_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    r = requests.get(url, params=params)
    st.write("ODDS API Status Code:", r.status_code)
    try:
        data = r.json()
        st.json(data)
    except:
        st.write("Invalid or empty response.")
        return pd.DataFrame()
    matches = []
    for match in data:
        if not match.get("bookmakers") or "home_team" not in match or "away_team" not in match:
            continue
        home = match["home_team"]
        away = match["away_team"]
        odds = match["bookmakers"][0]["markets"][0]["outcomes"]
        home_odds = away_odds = None
        draw_odds = 3.2 + np.random.rand()  # default
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
                "Draw Odds": draw_odds,
                "Away Odds": away_odds,
                "Kickoff": match.get("commence_time", "N/A")
            })
    return pd.DataFrame(matches)

# Just show odds table for demo
odds = load_odds(odds_key)
if odds.empty:
    st.warning("No odds available.")
else:
    st.dataframe(odds)