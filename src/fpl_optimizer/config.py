"""FPL scoring rules and model parameters."""

BASE_URL = "https://fantasy.premierleague.com/api/"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT = 30

DEFAULT_HORIZON = 5
DEFAULT_N_SIMS = 10_000
DEFAULT_SEED = 42

# Bayesian shrinkage: per-90 rates are blended with a positional prior worth this many minutes.
PRIOR_MINS = 300.0
POS_BASELINE_XG = {"GKP": 0.00, "DEF": 0.03, "MID": 0.12, "FWD": 0.25}
POS_BASELINE_XA = {"GKP": 0.00, "DEF": 0.05, "MID": 0.12, "FWD": 0.10}
POS_BASELINE_DC = {"GKP": 0.0, "DEF": 8.0, "MID": 6.0, "FWD": 3.0}  # defensive actions per 90

HOME_ADV, AWAY_ADV = 1.10, 0.90
SUB_APPEAR_PROB = 0.25  # chance of coming off the bench when not starting
SUB_MINUTES = 20.0
CARD_PENALTY_90 = 0.15  # expected points lost to cards per 90 minutes

# FPL scoring (2025/26 rules)
GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # defensive contributions needed for points
DC_PTS = 2
MAX_BONUS = 3.0

# Squad rules
SQUAD_SIZE = 15
SQUAD_SLOTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
FORMATION_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
POS_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
