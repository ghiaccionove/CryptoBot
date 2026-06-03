"""
cryptobot.py
ML-powered crypto trading bot for Binance Testnet.
Fetches OHLCV data, engineers features, trains an XGBoost classifier,
backtests on historical data, and runs live paper trading on Binance Testnet.
"""

import os
import csv
import json
import time
import traceback
from datetime import datetime, timedelta, timezone
import warnings
import urllib.request
import ccxt
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from backtesting import Backtest, Strategy
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Carica le variabili dal file .env (deve stare nella stessa cartella del bot)
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG -- edit these before running
# ─────────────────────────────────────────────

TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
TESTNET_SECRET  = os.getenv("BINANCE_TESTNET_SECRET", "")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

if not TESTNET_API_KEY or not TESTNET_SECRET:
    print("[WARN] BINANCE_TESTNET_API_KEY o BINANCE_TESTNET_SECRET non trovati nel .env.")
    print("       Il bot funzionerà in modalità backtest, ma non potrà piazzare ordini live.")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
    print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID non trovati nel .env.")
    print("       Le notifiche Telegram saranno disabilitate.")

SYMBOL          = "BTC/USDT"
TIMEFRAME       = "5m"        # candele 5 minuti (scalping)
FETCH_LIMIT     = 15000       # quante candele storiche scaricare (paginato, ~52 giorni a 5m)
N_TRAIN         = 400         # candele usate per il training
FUTURE_BARS     = 5           # quante candele in avanti per calcolare il label (25 min)
SIGNAL_THRESH   = 0.004       # 0.4% di movimento minimo per generare segnale
MIN_PROBA       = 0.58        # confidenza minima per BUY
TRADE_SIZE      = 0.80        # % del capitale usata per ogni trade
STOP_LOSS       = 0.007       # 0.7% stop loss (scalping — più largo per filtrare rumore 5m)
TRAIL_ATR_MULT  = 2.0         # trailing stop = ATR * 2.0 (più largo, evita shake-out)
TRAIL_ACTIVATE  = 0.005       # trailing si attiva dopo +0.5% di profitto
TAKE_PROFIT     = 0.012       # 1.2% take profit (R:R 1.71:1 con SL 0.7%)
INITIAL_CASH    = 5000         # capitale iniziale per il backtest (in USD)
SLEEP_SECONDS   = 60          # secondi tra ogni ciclo del bot (1 min, segnali ML ogni 5m)
MAX_RETRIES     = 3           # tentativi per errori di rete transitori
RETRY_BACKOFF   = [5, 15, 30] # secondi di attesa tra retry (piu' veloci per scalping)
MIN_HOLD_BARS   = 6           # hold minimo 6 barre = 30 min
LOG_FILE        = "trades_log.csv"
RETRAIN_HOURS   = 12          # riaddestra il modello ogni 12 ore (piu' frequente per 5m)
MODEL_FILE      = "model.joblib"
STATE_FILE      = "bot_state.json"
DASHBOARD_FILE  = "dashboard_data.json"
PRICE_HISTORY_FILE = "price_history.json"

# SHORT parameters (scalping)
SHORT_STOP_LOSS    = 0.007   # 0.7% stop loss per SHORT (scalping — più largo per filtrare rumore 5m)
SHORT_MIN_PROBA    = 0.60    # soglia SHORT
SHORT_TRADE_SIZE   = 0.80    # 80% capitale
SHORT_MIN_HOLD     = 5       # hold minimo 5 barre = 25 min
ENABLE_SHORT       = True    # flag per abilitare/disabilitare SHORT
SHORT_ADX_MIN      = 12      # ADX minimo per SHORT

# Futures-only mode (entrambi LONG e SHORT su Futures per fee 0.02% maker)
USE_FUTURES_FOR_BOTH = True

# Model files (dual model, rinominati per evitare conflitto col branch main)
MODEL_BUY_FILE     = "model_buy_scalp.joblib"
MODEL_SHORT_FILE   = "model_short_scalp.joblib"

# Optuna cache files (scalping)
OPTUNA_BUY_FILE    = "best_params_buy_scalp.json"
OPTUNA_SHORT_FILE  = "best_params_short_scalp.json"
OPTUNA_TRIALS      = 100     # trial per ottimizzazione bayesiana (per modello, +67% vs v1)

TRANSIENT_ERRORS = (
    ccxt.NetworkError,      # copre ExchangeNotAvailable, RequestTimeout, DDoSProtection
    ccxt.InvalidNonce,      # clock drift temporaneo VPS vs Binance
)

FEATURES = [
    # 5m base - oscillatori/momentum (11)
    "rsi_fast", "rsi_slope",
    "macd_fast", "macd_signal_fast", "macd_hist_fast",
    "stoch_k", "stoch_d",
    "bb_width", "bb_pct",
    "willr", "cci",
    # 5m dinamica prezzo/volume (7)
    "price_change", "price_change_3",
    "vol_change", "vol_spike",
    "vwap_dist", "atr", "atr_ratio",
    # 5m microstruttura (3)
    "spread_proxy", "body_ratio", "ema_cross_fast",
    # 15m contesto (resample da 5m) (3)
    "rsi_15m", "macd_15m", "trend_15m",
    # 1h contesto (resample da 5m) (3)
    "rsi_1h", "trend_1h", "adx_1h",
    # Tempo (1)
    "minute_of_day",
]

# Walk-forward validation (scalping)
WF_TRAIN_BARS   = 3000        # candele per ogni finestra di training (+50%, ~10 giorni a 5m)
WF_TEST_BARS    = 500         # candele per ogni finestra di test (~1.7 giorni)

# ─────────────────────────────────────────────
# 1. FETCH DATI
# ─────────────────────────────────────────────

_ohlcv_exchange = None  # cache globale per evitare load_markets() ogni ciclo

def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=FETCH_LIMIT):
    """
    Scarica i dati OHLCV da Binance (endpoint pubblico, niente API key).
    Supporta paginazione per ottenere piu' di 1000 candele.
    Usa istanza cached per evitare chiamate ripetute a exchangeInfo.
    Fallback: se l'API pubblica fallisce, usa il demo endpoint.
    """
    global _ohlcv_exchange
    if _ohlcv_exchange is None:
        try:
            _ohlcv_exchange = ccxt.binance({
                "options": {"fetchMarkets": {"types": ["spot"]}},
            })
            _ohlcv_exchange.load_markets()
        except Exception:
            # Fallback: usa demo endpoint (stessi dati di mercato)
            print("[WARN] API pubblica Binance non raggiungibile, uso demo endpoint")
            _ohlcv_exchange = ccxt.binance({
                "apiKey": TESTNET_API_KEY,
                "secret": TESTNET_SECRET,
                "options": {"defaultType": "spot", "fetchMarkets": {"types": ["spot"]}},
            })
            _ohlcv_exchange.enable_demo_trading(True)
            _ohlcv_exchange.load_markets()
    exchange = _ohlcv_exchange
    max_per_request = 1000

    if limit <= max_per_request:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    else:
        # Paginazione: scarica a blocchi da 1000 andando indietro nel tempo
        all_ohlcv = []
        remaining = limit
        since = None

        # Prima richiesta: ultime 1000 candele
        batch = exchange.fetch_ohlcv(symbol, timeframe, limit=max_per_request)
        all_ohlcv = batch
        remaining -= len(batch)

        # Richieste successive: va indietro nel tempo
        while remaining > 0 and len(batch) == max_per_request:
            earliest_ts = all_ohlcv[0][0]
            # Calcola il timestamp da cui partire (prima della candela piu' vecchia)
            tf_ms = exchange.parse_timeframe(timeframe) * 1000
            since = earliest_ts - max_per_request * tf_ms
            batch_size = min(remaining, max_per_request)
            batch = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size
            )
            if not batch:
                break
            # Filtra candele gia' presenti
            batch = [c for c in batch if c[0] < earliest_ts]
            all_ohlcv = batch + all_ohlcv
            remaining -= len(batch)

        ohlcv = all_ohlcv[-limit:]  # tronca al numero richiesto

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING + LABEL
# ─────────────────────────────────────────────

def build_features(df):
    """
    Aggiunge indicatori tecnici scalping e calcola il label:
      +1  = BUY  (prezzo sale > soglia dinamica ATR nelle prossime FUTURE_BARS candele)
      -1  = SELL (prezzo scende)
       0  = HOLD
    Timeframe primario: 5m. Contesto: 15m e 1h (resampleati dal 5m).
    """
    df = df.copy()

    # === Feature 5m (base) - oscillatori/momentum ===
    df["rsi_fast"]    = ta.rsi(df["close"], length=7)
    df["rsi_slope"]   = df["rsi_fast"].diff(3)
    macd_df           = ta.macd(df["close"], fast=8, slow=21, signal=5)
    df["macd_fast"]        = macd_df.iloc[:, 0]  # MACD line
    df["macd_signal_fast"] = macd_df.iloc[:, 2]  # Signal line
    df["macd_hist_fast"]   = macd_df.iloc[:, 1]  # Histogram
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=5, d=3, smooth_k=3)
    df["stoch_k"]     = stoch.iloc[:, 0]
    df["stoch_d"]     = stoch.iloc[:, 1]
    bb = ta.bbands(df["close"], length=10)
    bb_upper = bb.iloc[:, 2]  # BBU
    bb_lower = bb.iloc[:, 0]  # BBL
    df["bb_width"]    = (bb_upper - bb_lower) / df["close"]
    df["bb_pct"]      = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df["willr"]       = ta.willr(df["high"], df["low"], df["close"], length=7)
    df["cci"]         = ta.cci(df["high"], df["low"], df["close"], length=14)

    # === Feature 5m - dinamica prezzo/volume ===
    df["price_change"]   = df["close"].pct_change()
    df["price_change_3"] = df["close"].pct_change(3)
    df["vol_change"]     = df["volume"].pct_change()
    vol_ma_20            = df["volume"].rolling(20).mean()
    df["vol_spike"]      = df["volume"] / (vol_ma_20 + 1e-10)
    cum_vol              = df["volume"].rolling(20).sum()
    cum_vp               = (df["close"] * df["volume"]).rolling(20).sum()
    vwap_20              = cum_vp / (cum_vol + 1e-10)
    df["vwap_dist"]      = (df["close"] - vwap_20) / (vwap_20 + 1e-10)
    df["atr"]            = ta.atr(df["high"], df["low"], df["close"], length=10)
    atr_fast             = ta.atr(df["high"], df["low"], df["close"], length=5)
    atr_slow             = ta.atr(df["high"], df["low"], df["close"], length=20)
    df["atr_ratio"]      = atr_fast / (atr_slow + 1e-10)

    # === Feature 5m - microstruttura ===
    bar_range            = df["high"] - df["low"]
    df["spread_proxy"]   = bar_range / df["close"]
    df["body_ratio"]     = abs(df["close"] - df["open"]) / (bar_range + 1e-10)
    ema5                 = ta.ema(df["close"], length=5)
    ema20                = ta.ema(df["close"], length=20)
    df["ema_cross_fast"] = ema5 - ema20

    # trend_up interno (usato per filtri, non e' una feature ML)
    df["ema_20"]         = ema20
    df["trend_up"]       = (df["close"] > ema20).astype(int)

    # === Feature 15m contesto (resample da 5m) ===
    df_15m = df[["open", "high", "low", "close", "volume"]].resample(
        "15min", closed="right", label="right"
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    df_15m["rsi_15m"]  = ta.rsi(df_15m["close"], length=14)
    macd_15m           = ta.macd(df_15m["close"])
    df_15m["macd_15m"] = macd_15m.iloc[:, 0] if macd_15m is not None else 0.0
    ema20_15m          = ta.ema(df_15m["close"], length=20)
    df_15m["trend_15m"] = (df_15m["close"] > ema20_15m).astype(int)
    for col in ["rsi_15m", "macd_15m", "trend_15m"]:
        df[col] = df_15m[col].reindex(df.index, method="ffill")

    # === Feature 1h contesto (resample da 5m) ===
    df_1h = df[["open", "high", "low", "close", "volume"]].resample(
        "1h", closed="right", label="right"
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    df_1h["rsi_1h"]   = ta.rsi(df_1h["close"], length=14)
    ema20_1h           = ta.ema(df_1h["close"], length=20)
    df_1h["trend_1h"]  = (df_1h["close"] > ema20_1h).astype(int)
    adx_1h_df          = ta.adx(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
    df_1h["adx_1h"]    = adx_1h_df["ADX_14"]
    for col in ["rsi_1h", "trend_1h", "adx_1h"]:
        df[col] = df_1h[col].reindex(df.index, method="ffill")

    # === Tempo ===
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute

    # === Target dinamico (soglia basata su ATR, scalping) ===
    atr_pct = df["atr"] / df["close"]
    # Moltiplicatore 0.4 — bilancia tra cattura segnali e riduzione rumore
    dynamic_thresh = np.maximum(SIGNAL_THRESH, atr_pct * 0.3)

    future_return     = df["close"].shift(-FUTURE_BARS) / df["close"] - 1
    df["label"]       = 0
    df.loc[future_return >  dynamic_thresh, "label"] =  1
    df.loc[future_return < -dynamic_thresh, "label"] = -1

    return df.dropna()


# ─────────────────────────────────────────────
# 3. TRAINING MODELLO
# ─────────────────────────────────────────────

def train_model(df, target_label=1):
    """
    Allena un XGBoost binary classifier.
    target_label=1 per BUY vs NO-BUY, target_label=-1 per SHORT vs NO-SHORT.
    Usa shuffle=False perche' i dati sono time-series.
    """
    label_name = "BUY" if target_label == 1 else "SHORT"
    neg_name = "NO-BUY" if target_label == 1 else "NO-SHORT"

    X = df[FEATURES]
    y = (df["label"] == target_label).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Diagnostica distribuzione classi
    print(f"Distribuzione classi {label_name} (train):")
    for cls in [0, 1]:
        count = (y_train == cls).sum()
        name = label_name if cls == 1 else neg_name
        print(f"  {name}: {count} ({count/len(y_train):.1%})")

    # scale_pos_weight: bilancia automaticamente la classe rara
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=1.5,
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
        early_stopping_rounds=80
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"\nMiglior iterazione ({label_name}): {model.best_iteration} / 800")
    print(f"\n=== Valutazione modello {label_name} sul test set ===")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=[neg_name, label_name]))

    return model


# ─────────────────────────────────────────────
# 3b. OTTIMIZZAZIONE IPERPARAMETRI (OPTUNA)
# ─────────────────────────────────────────────

def optimize_hyperparams(df, target_label=1, cache_file=None, n_trials=OPTUNA_TRIALS):
    """
    Usa Optuna per trovare i migliori iperparametri XGBoost.
    Valida su un split temporale (ultimi 20% del dataset).
    target_label: 1 per BUY, -1 per SHORT.
    cache_file: file JSON per caching (default: OPTUNA_BUY_FILE o OPTUNA_SHORT_FILE).
    """
    if cache_file is None:
        cache_file = OPTUNA_BUY_FILE if target_label == 1 else OPTUNA_SHORT_FILE

    label_name = "BUY" if target_label == 1 else "SHORT"

    # Se esiste un file recente (< 48h), riusa i parametri
    if os.path.isfile(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 48:
            with open(cache_file) as f:
                params = json.load(f)
            print(f"Iperparametri {label_name} caricati da {cache_file} (eta': {age_h:.1f}h)")
            return params

    print(f"\n=== Ottimizzazione iperparametri {label_name} ({n_trials} trial) ===")
    X = df[FEATURES]
    y = (df["label"] == target_label).astype(int)

    # Split temporale: train 70%, val 15%, test 15% (test non usato qui)
    split_1 = int(len(X) * 0.70)
    split_2 = int(len(X) * 0.85)
    X_train, y_train = X.iloc[:split_1], y.iloc[:split_1]
    X_val, y_val = X.iloc[split_1:split_2], y.iloc[split_1:split_2]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = min(neg / pos, 30.0) if pos > 0 else 1.0
    print(f"  Samples positivi ({label_name}): {pos}, negativi: {neg}, scale_pos_weight: {spw:.1f}")

    def objective(trial):
        params = {
            "n_estimators": 800,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "verbosity": 0,
            "early_stopping_rounds": 50,
        }
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = m.predict(X_val)
        from sklearn.metrics import f1_score
        return f1_score(y_val, preds, pos_label=1, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"Migliori parametri {label_name} (F1: {study.best_value:.3f}):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Salva su disco
    with open(cache_file, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Salvati in {cache_file}")

    return best


# ─────────────────────────────────────────────
# 3c. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def technical_sell_signal(df):
    """
    Genera segnali SELL basati su regole tecniche (scalping):
    - RSI fast > 75 (ipercomprato estremo) + slope fortemente negativo
    - MACD cross ribassista confermato su 2 barre consecutive
    - Prezzo crolla sotto EMA20 con momentum forte (-0.5%)
    """
    sell = pd.Series(False, index=df.index)

    # RSI ipercomprato estremo + reversal forte
    sell |= (df["rsi_fast"] > 75) & (df["rsi_slope"] < -4)

    # MACD cross ribassista confermato (2 barre consecutive negative e in peggioramento)
    macd_gap = df["macd_fast"] - df["macd_signal_fast"]
    sell |= (macd_gap < 0) & (macd_gap.diff() < 0) & (macd_gap.shift(1) < 0)

    # Prezzo crolla sotto EMA20 con momentum forte (-0.5%)
    sell |= (df["trend_up"] == 0) & (df["price_change"] < -0.005)

    return sell


def technical_cover_signal(df):
    """
    Genera segnali di COPERTURA SHORT basati su regole tecniche (scalping):
    - RSI fast < 28 (ipervenduto estremo) + rimbalzo forte
    - MACD cross rialzista confermato su 2 barre consecutive
    - Prezzo risale sopra EMA20 con momentum forte (+0.5%)
    """
    cover = pd.Series(False, index=df.index)

    # RSI ipervenduto estremo + rimbalzo forte
    cover |= (df["rsi_fast"] < 28) & (df["rsi_slope"] > 4)

    # MACD cross rialzista confermato (2 barre consecutive positive e in miglioramento)
    macd_gap = df["macd_fast"] - df["macd_signal_fast"]
    cover |= (macd_gap > 0) & (macd_gap.diff() > 0) & (macd_gap.shift(1) > 0)

    # Prezzo risale sopra EMA20 con momentum forte (+0.5%)
    cover |= (df["trend_up"] == 1) & (df["price_change"] > 0.005)

    return cover


class EnsembleClassifier:
    """
    Ensemble di piu' modelli XGBoost con strategia max-con-agreement.
    Compatibile con l'interfaccia sklearn (predict_proba, predict, feature_importances_).
    Usato per il live bot per replicare piu' fedelmente il comportamento walk-forward.

    Strategia predict_proba:
    - Se almeno min_agree modelli su N concordano (proba classe 1 >= agree_thresh),
      ritorna il MASSIMO tra le probabilita' dei modelli concordanti.
    - Altrimenti ritorna la MEDIA (naturalmente bassa, nessun segnale).
    Questo elimina la compressione del range causata dalla media semplice
    (che porta le proba da ~55-64% a ~5-43%, sotto le soglie di ingresso).
    """
    def __init__(self, models, keep_last_n=5, min_agree=2, agree_thresh=0.50):
        # Tieni solo gli ultimi N modelli non-degeneri
        self.models = models[-keep_last_n:] if len(models) > keep_last_n else models
        self.min_agree = min_agree
        self.agree_thresh = agree_thresh
        print(f"  Ensemble creato: {len(self.models)} modelli (da {len(models)} fold validi), "
              f"agreement >= {min_agree}/{len(self.models)} @ thresh {agree_thresh:.0%}")

    def predict_proba(self, X):
        """
        Max-con-agreement: se >= min_agree modelli concordano (proba >= agree_thresh),
        ritorna il max. Altrimenti ritorna la media.
        """
        probas_class1 = np.array([m.predict_proba(X)[:, 1] for m in self.models])
        # probas_class1 shape: (n_models, n_samples)
        agreement_count = (probas_class1 >= self.agree_thresh).sum(axis=0)
        # Per ogni campione: usa max se agreement, media altrimenti
        mean_p1 = probas_class1.mean(axis=0)
        max_p1  = probas_class1.max(axis=0)
        p1 = np.where(agreement_count >= self.min_agree, max_p1, mean_p1)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """Predizione basata sulla media delle probabilita' (soglia 0.5)."""
        avg_proba = self.predict_proba(X)
        return (avg_proba[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """Media delle feature importance di tutti i modelli."""
        importances = [m.feature_importances_ for m in self.models]
        return np.mean(importances, axis=0)


def _train_single_model(X_train, y_train, X_test, y_test, hp, prev_model, min_samples=20):
    """
    Allena un singolo XGBClassifier con guard per fold degeneri.
    Ritorna (model_or_fallback, preds, proba, is_degenerate, prev_model).
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = min(neg_count / pos_count, 30.0) if pos_count > 0 else 1.0

    mdl = XGBClassifier(
        n_estimators=800,
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        min_child_weight=hp["min_child_weight"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
        early_stopping_rounds=50
    )
    mdl.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    is_degenerate = mdl.best_iteration < 10 or pos_count < min_samples
    if is_degenerate and prev_model is not None:
        preds = prev_model.predict(X_test)
        proba = prev_model.predict_proba(X_test)[:, 1]
    elif is_degenerate:
        preds = np.zeros(len(X_test), dtype=int)
        proba = np.full(len(X_test), 0.0)
    else:
        preds = mdl.predict(X_test)
        proba = mdl.predict_proba(X_test)[:, 1]
        prev_model = mdl

    return mdl, preds, proba, is_degenerate, prev_model


def walk_forward_train(df, best_params_buy=None, best_params_short=None):
    """
    Walk-forward validation con dual binary classification:
    - model_buy:   BUY (1) vs NO-BUY (0)
    - model_short: SHORT (1) vs NO-SHORT (0)
    Il segnale CLOSE LONG viene da regole tecniche (technical_sell_signal).
    Il segnale CLOSE SHORT viene da regole tecniche (technical_cover_signal).
    Per ogni finestra, genera predictions out-of-sample.
    Signal encoding: +1=OPEN LONG, -1=OPEN SHORT, -2=CLOSE LONG, +2=CLOSE SHORT, 0=HOLD.
    Se ENABLE_SHORT=False, il model_short non viene trainato.
    """
    X = df[FEATURES]
    y_buy   = (df["label"] == 1).astype(int)
    y_short = (df["label"] == -1).astype(int)

    # Risultati per fold (BUY)
    all_buy_preds  = []
    all_buy_proba  = []
    all_buy_actual = []
    # Risultati per fold (SHORT)
    all_short_preds  = []
    all_short_proba  = []
    all_short_actual = []
    all_idx    = []
    n = len(df)
    fold = 0
    prev_buy_model = None
    prev_short_model = None
    MIN_SAMPLES = 20
    # Raccolta modelli validi per ensemble
    valid_buy_models = []
    valid_short_models = []

    # Iperparametri BUY: Optuna o default
    hp_buy = {
        "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
    }
    if best_params_buy:
        hp_buy.update(best_params_buy)
        print(f"Usando iperparametri BUY ottimizzati (Optuna)")

    # Iperparametri SHORT: Optuna o default
    hp_short = {
        "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10,
        "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.3, "reg_lambda": 1.5,
    }
    if best_params_short:
        hp_short.update(best_params_short)
        print(f"Usando iperparametri SHORT ottimizzati (Optuna)")

    short_enabled = ENABLE_SHORT
    mode_str = "LONG+SHORT" if short_enabled else "LONG-only"
    print(f"Walk-forward ({mode_str}): train={WF_TRAIN_BARS}, test={WF_TEST_BARS}, "
          f"dataset={n} righe")

    i = 0
    while i + WF_TRAIN_BARS + WF_TEST_BARS <= n:
        fold += 1
        train_end = i + WF_TRAIN_BARS
        test_end  = min(train_end + WF_TEST_BARS, n)

        X_train = X.iloc[i:train_end]
        X_test  = X.iloc[train_end:test_end]

        # --- BUY model ---
        y_buy_train = y_buy.iloc[i:train_end]
        y_buy_test  = y_buy.iloc[train_end:test_end]
        buy_mdl, buy_preds, buy_proba, buy_degen, prev_buy_model = _train_single_model(
            X_train, y_buy_train, X_test, y_buy_test, hp_buy, prev_buy_model, MIN_SAMPLES
        )

        # --- SHORT model ---
        if short_enabled:
            y_short_train = y_short.iloc[i:train_end]
            y_short_test  = y_short.iloc[train_end:test_end]
            short_mdl, short_preds, short_proba, short_degen, prev_short_model = _train_single_model(
                X_train, y_short_train, X_test, y_short_test, hp_short, prev_short_model, MIN_SAMPLES
            )
        else:
            short_preds = np.zeros(len(X_test), dtype=int)
            short_proba = np.full(len(X_test), 0.0)
            short_degen = True

        # Raccogli modelli validi per ensemble
        if not buy_degen:
            valid_buy_models.append(buy_mdl)
        if short_enabled and not short_degen:
            valid_short_models.append(short_mdl)

        all_buy_preds.extend(buy_preds)
        all_buy_proba.extend(buy_proba)
        all_buy_actual.extend(y_buy_test.values)
        all_short_preds.extend(short_preds)
        all_short_proba.extend(short_proba)
        all_short_actual.extend(y_short.iloc[train_end:test_end].values)
        all_idx.extend(y_buy_test.index)

        buy_cnt = sum(buy_preds)
        short_cnt = sum(short_preds) if short_enabled else 0
        buy_status = " [DEGEN]" if buy_degen else ""
        short_status = " [DEGEN]" if short_degen else ""
        print(f"  Fold {fold}: train [{i}:{train_end}] | "
              f"test [{train_end}:{test_end}] | "
              f"BUY: {buy_cnt}/{len(buy_preds)}{buy_status} | "
              f"SHORT: {short_cnt}/{len(short_preds)}{short_status}")

        i += WF_TEST_BARS

    print(f"\n=== Walk-forward: {fold} fold completati ({mode_str}) ===")
    print(f"Predictions out-of-sample totali: {len(all_buy_preds)}")

    # Classification report BUY
    print("\n=== Classification report BUY (aggregato) ===")
    print(classification_report(
        all_buy_actual, all_buy_preds,
        target_names=["NO-BUY", "BUY"]
    ))

    # Classification report SHORT
    if short_enabled:
        print("=== Classification report SHORT (aggregato) ===")
        print(classification_report(
            all_short_actual, all_short_preds,
            target_names=["NO-SHORT", "SHORT"]
        ))

    # Analisi qualita' segnali BUY
    buy_preds_arr = np.array(all_buy_preds)
    buy_actual_arr = np.array(all_buy_actual)
    if buy_preds_arr.sum() > 0:
        buy_precision = buy_actual_arr[buy_preds_arr == 1].mean()
        print(f"BUY precision effettiva: {buy_precision:.1%} "
              f"({buy_preds_arr.sum()} segnali BUY predetti)")

    # Analisi qualita' segnali SHORT
    if short_enabled:
        short_preds_arr = np.array(all_short_preds)
        short_actual_arr = np.array(all_short_actual)
        if short_preds_arr.sum() > 0:
            short_precision = short_actual_arr[short_preds_arr == 1].mean()
            print(f"SHORT precision effettiva: {short_precision:.1%} "
                  f"({short_preds_arr.sum()} segnali SHORT predetti)")

    # === Genera segnali combinati per il backtest ===
    # Signal encoding:
    #   +1 = OPEN LONG   (ML BUY con alta confidenza)
    #   -1 = OPEN SHORT  (ML SHORT con alta confidenza)
    #   -2 = CLOSE LONG  (technical sell signal)
    #   +2 = CLOSE SHORT (technical cover signal)
    #    0 = HOLD
    sell_signals  = technical_sell_signal(df)
    cover_signals = technical_cover_signal(df)

    pred_values = np.zeros(len(all_buy_preds))
    n_conflicts = 0
    for j in range(len(all_buy_preds)):
        idx = all_idx[j]
        buy_ok   = all_buy_preds[j] == 1 and all_buy_proba[j] >= MIN_PROBA
        short_ok = short_enabled and all_short_preds[j] == 1 and all_short_proba[j] >= SHORT_MIN_PROBA

        if buy_ok and short_ok:
            pred_values[j] = 0  # conflitto: entrambi i modelli "sparano" -> HOLD
            n_conflicts += 1
        elif buy_ok:
            pred_values[j] = 1   # OPEN LONG
        elif short_ok:
            pred_values[j] = -1  # OPEN SHORT
        elif sell_signals.loc[idx]:
            pred_values[j] = -2  # CLOSE LONG (technical)
        elif short_enabled and cover_signals.loc[idx]:
            pred_values[j] = 2   # CLOSE SHORT (technical)
        # else: 0 (HOLD)

    pred_series = pd.Series(pred_values.astype(int), index=all_idx)

    # Stats dei segnali
    n_open_long  = (pred_values == 1).sum()
    n_open_short = (pred_values == -1).sum()
    n_close_long = (pred_values == -2).sum()
    n_close_short = (pred_values == 2).sum()
    print(f"\nSegnali generati: OPEN_LONG={n_open_long}, OPEN_SHORT={n_open_short}, "
          f"CLOSE_LONG={n_close_long}, CLOSE_SHORT={n_close_short}, "
          f"HOLD={int((pred_values == 0).sum())}, CONFLITTI={n_conflicts}")

    # Crea ensemble dagli ultimi N fold validi (piu' rappresentativo del backtest)
    ENSEMBLE_SIZE = 5
    if len(valid_buy_models) >= 2:
        final_buy_model = EnsembleClassifier(valid_buy_models, keep_last_n=ENSEMBLE_SIZE, agree_thresh=0.42)
    else:
        final_buy_model = prev_buy_model if prev_buy_model else buy_mdl
        print("  [WARN] Solo 1 modello BUY valido, no ensemble")

    if short_enabled and len(valid_short_models) >= 2:
        final_short_model = EnsembleClassifier(valid_short_models, keep_last_n=ENSEMBLE_SIZE, agree_thresh=0.40)
    elif short_enabled:
        final_short_model = prev_short_model if prev_short_model else (short_mdl if short_enabled else None)
        print("  [WARN] Solo 1 modello SHORT valido, no ensemble")
    else:
        final_short_model = None

    # === Auto-calibrazione soglie ensemble ===
    # L'ensemble media le probabilita' di piu' modelli, comprimendo il range.
    # Soglie calibrate per singolo modello (58%/60%) non funzionano per l'ensemble.
    # Usiamo SOLO i dati OOS degli ultimi ENSEMBLE_SIZE fold per evitare look-ahead bias:
    # l'ensemble contiene gli ultimi 5 modelli, calibrare su fold precedenti gonfia le probabilita'.
    n_recent_oos = ENSEMBLE_SIZE * WF_TEST_BARS
    recent_idx = all_idx[-n_recent_oos:] if len(all_idx) > n_recent_oos else all_idx
    X_oos = X.loc[recent_idx]

    # Conta segnali per-fold solo nella finestra recente (stessa finestra dell'ensemble)
    recent_buy_preds = np.array(all_buy_preds[-n_recent_oos:]) if len(all_buy_preds) > n_recent_oos else np.array(all_buy_preds)
    recent_buy_proba_arr = np.array(all_buy_proba[-n_recent_oos:]) if len(all_buy_proba) > n_recent_oos else np.array(all_buy_proba)
    n_buy_signals_pf = int(((recent_buy_preds == 1) & (recent_buy_proba_arr >= MIN_PROBA)).sum())

    ens_buy_proba = final_buy_model.predict_proba(X_oos)[:, 1]
    sorted_buy = np.sort(ens_buy_proba)[::-1]
    p10, p25, p50, p75, p90 = np.percentile(ens_buy_proba, [10, 25, 50, 75, 90])
    if n_buy_signals_pf > 0 and n_buy_signals_pf < len(sorted_buy):
        cal_buy_thresh = float(sorted_buy[n_buy_signals_pf - 1])
        cal_buy_thresh = max(0.30, cal_buy_thresh)
        # Cap: mai oltre P75 della distribuzione ensemble, con hard ceiling +10pp sopra il default
        hard_cap_buy = MIN_PROBA + 0.10  # max 0.68
        cal_buy_thresh = min(cal_buy_thresh, max(MIN_PROBA, p75), hard_cap_buy)
    else:
        cal_buy_thresh = MIN_PROBA

    print(f"\n  [CALIBRAZIONE] BUY ensemble proba: "
          f"P10={p10:.2%} P25={p25:.2%} P50={p50:.2%} P75={p75:.2%} P90={p90:.2%}")
    print(f"  [CALIBRAZIONE] BUY: {n_buy_signals_pf} segnali per-fold (ultimi {ENSEMBLE_SIZE} fold) -> "
          f"soglia ensemble = {cal_buy_thresh:.2%} (default {MIN_PROBA:.0%})")

    cal_short_thresh = SHORT_MIN_PROBA
    if short_enabled and final_short_model is not None:
        recent_short_preds = np.array(all_short_preds[-n_recent_oos:]) if len(all_short_preds) > n_recent_oos else np.array(all_short_preds)
        recent_short_proba_arr = np.array(all_short_proba[-n_recent_oos:]) if len(all_short_proba) > n_recent_oos else np.array(all_short_proba)
        n_short_signals_pf = int(((recent_short_preds == 1) & (recent_short_proba_arr >= SHORT_MIN_PROBA)).sum())

        ens_short_proba = final_short_model.predict_proba(X_oos)[:, 1]
        sorted_short = np.sort(ens_short_proba)[::-1]
        p10s, p25s, p50s, p75s, p90s = np.percentile(ens_short_proba, [10, 25, 50, 75, 90])
        if n_short_signals_pf > 0 and n_short_signals_pf < len(sorted_short):
            cal_short_thresh = float(sorted_short[n_short_signals_pf - 1])
            cal_short_thresh = max(0.30, cal_short_thresh)
            # Cap: mai oltre P75 della distribuzione ensemble, con hard ceiling +10pp sopra il default
            hard_cap_short = SHORT_MIN_PROBA + 0.10  # max 0.70
            cal_short_thresh = min(cal_short_thresh, max(SHORT_MIN_PROBA, p75s), hard_cap_short)
        else:
            cal_short_thresh = SHORT_MIN_PROBA
        print(f"  [CALIBRAZIONE] SHORT ensemble proba: "
              f"P10={p10s:.2%} P25={p25s:.2%} P50={p50s:.2%} P75={p75s:.2%} P90={p90s:.2%}")
        print(f"  [CALIBRAZIONE] SHORT: {n_short_signals_pf} segnali per-fold (ultimi {ENSEMBLE_SIZE} fold) -> "
              f"soglia ensemble = {cal_short_thresh:.2%} (default {SHORT_MIN_PROBA:.0%})")

    # Salva soglie calibrate nel modello ensemble per uso in run_bot
    if hasattr(final_buy_model, 'models'):
        final_buy_model.calibrated_threshold = cal_buy_thresh
    if final_short_model is not None and hasattr(final_short_model, 'models'):
        final_short_model.calibrated_threshold = cal_short_thresh

    return final_buy_model, final_short_model, pred_series


# ─────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────

def backtest(df_raw, df_feat, model_buy, model_short=None, pred_series=None, test_size=0.2):
    """
    Backtesta la strategia ML solo sui dati out-of-sample (test set)
    per evitare data leakage. Include stop loss e report P&L in dollari.
    Se pred_series e' fornita (da walk-forward), usa quelle predictions.
    Supporta LONG + SHORT con signal encoding:
      +1=OPEN LONG, -1=OPEN SHORT, -2=CLOSE LONG, +2=CLOSE SHORT, 0=HOLD.
    """

    if pred_series is not None:
        # Walk-forward: usa le predictions aggregate out-of-sample
        feat_index  = pred_series.index
        raw_aligned = df_raw.loc[feat_index].copy()
    else:
        # Singolo split: genera predictions dal modello (LONG-only fallback)
        split_idx   = int(len(df_feat) * (1 - test_size))
        df_test     = df_feat.iloc[split_idx:]
        feat_index  = df_test.index
        raw_aligned = df_raw.loc[feat_index].copy()
        predictions = model_buy.predict(df_test[FEATURES]) - 1
        pred_series = pd.Series(predictions, index=feat_index)

    # Filtro trend: usa trend 1h come direzionale (scalping con contesto)
    trend_1h_series = df_feat.loc[feat_index, "trend_1h"]
    # ATR per trailing stop (in unita' prezzo)
    atr_series = df_feat.loc[feat_index, "atr"]
    # ADX 1h per filtro SHORT
    adx_1h_series = df_feat.loc[feat_index, "adx_1h"]

    class MLStrategy(Strategy):
        def init(self):
            self.signal = self.I(
                lambda: pred_series.reindex(raw_aligned.index).fillna(0).values,
                name="ML Signal"
            )
            self.trend_1h = self.I(
                lambda: trend_1h_series.reindex(raw_aligned.index).fillna(0).values,
                name="Trend1h"
            )
            self.atr = self.I(
                lambda: atr_series.reindex(raw_aligned.index).fillna(0).values,
                name="ATR"
            )
            self.adx_1h = self.I(
                lambda: adx_1h_series.reindex(raw_aligned.index).fillna(0).values,
                name="ADX1h"
            )
            self.entry_price = None
            self.trail_stop = None
            self.bars_held = 0

        def next(self):
            sig = self.signal[-1]
            price = self.data.Close[-1]
            trend_ok = self.trend_1h[-1] == 1
            cur_atr = self.atr[-1] / price_divisor if price_divisor > 1 else self.atr[-1]

            # --- TAKE PROFIT LONG (scalping: 0.6%) ---
            if self.position.is_long and self.entry_price:
                pnl_pct = (price - self.entry_price) / self.entry_price
                if pnl_pct >= TAKE_PROFIT:
                    self.position.close()
                    self._reset()
                    return

            # --- TAKE PROFIT SHORT (scalping: 0.6%) ---
            if self.position.is_short and self.entry_price:
                pnl_pct = (self.entry_price - price) / self.entry_price
                if pnl_pct >= TAKE_PROFIT:
                    self.position.close()
                    self._reset()
                    return

            # --- Trailing stop / Stop loss LONG ---
            if self.position.is_long and self.entry_price:
                self.bars_held += 1
                pnl_pct = (price - self.entry_price) / self.entry_price
                if pnl_pct >= TRAIL_ACTIVATE:
                    new_trail = price - cur_atr * TRAIL_ATR_MULT
                    if self.trail_stop is None:
                        self.trail_stop = max(new_trail, self.entry_price)
                    else:
                        self.trail_stop = max(self.trail_stop, new_trail)
                    if price <= self.trail_stop:
                        self.position.close()
                        self._reset()
                        return
                else:
                    if pnl_pct <= -STOP_LOSS:
                        self.position.close()
                        self._reset()
                        return

            # --- Trailing stop / Stop loss SHORT ---
            if self.position.is_short and self.entry_price:
                self.bars_held += 1
                pnl_pct = (self.entry_price - price) / self.entry_price
                if pnl_pct >= TRAIL_ACTIVATE:
                    new_trail = price + cur_atr * TRAIL_ATR_MULT
                    if self.trail_stop is None:
                        self.trail_stop = min(new_trail, self.entry_price)
                    else:
                        self.trail_stop = min(self.trail_stop, new_trail)
                    if price >= self.trail_stop:
                        self.position.close()
                        self._reset()
                        return
                else:
                    if pnl_pct <= -SHORT_STOP_LOSS:
                        self.position.close()
                        self._reset()
                        return

            # OPEN LONG: sig == +1, flat, trend 1h up
            if sig == 1 and not self.position and trend_ok:
                self.buy(size=TRADE_SIZE)
                self.entry_price = price
                self.trail_stop = None
                self.bars_held = 0

            # OPEN SHORT: sig == -1, flat, trend 1h down, ADX 1h forte
            elif sig == -1 and not self.position and not trend_ok:
                if self.adx_1h[-1] < SHORT_ADX_MIN:
                    return
                self.sell(size=SHORT_TRADE_SIZE)
                self.entry_price = price
                self.trail_stop = None
                self.bars_held = 0

            # CLOSE LONG: sig == -2, hold minimo rispettato
            elif sig == -2 and self.position.is_long:
                if self.bars_held < MIN_HOLD_BARS:
                    return
                self.position.close()
                self._reset()

            # CLOSE SHORT: sig == +2, hold minimo rispettato
            elif sig == 2 and self.position.is_short:
                if self.bars_held < SHORT_MIN_HOLD:
                    return
                self.position.close()
                self._reset()

        def _reset(self):
            self.entry_price = None
            self.trail_stop = None
            self.bars_held = 0

    # backtesting.py richiede colonne con la prima lettera maiuscola
    bt_df = raw_aligned.rename(columns={
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume"
    })

    # backtesting.py arrotonda le posizioni a interi.
    # Con capitali piccoli e BTC a prezzi alti (es. $500 vs $80K),
    # l'ordine risulterebbe 0 unita'. Scaliamo i prezzi in modo che
    # il capitale possa comprare almeno ~10 unita'.
    # Il P&L in $ e i rendimenti % restano identici.
    last_price = bt_df["Close"].iloc[-1]
    price_divisor = max(1, int(last_price / (INITIAL_CASH * 0.1)))
    if price_divisor > 1:
        print(f"  [INFO] Prezzi scalati /{price_divisor} per backtest "
              f"(capitale piccolo vs prezzo asset)")
        for col in ["Open", "High", "Low", "Close"]:
            bt_df[col] = bt_df[col] / price_divisor

    bt = Backtest(
        bt_df,
        MLStrategy,
        cash=INITIAL_CASH,
        commission=0.0004,     # 0.04% round-trip Futures (0.02% per lato)
        exclusive_orders=True
    )
    stats = bt.run()

    print("\n=== Risultati Backtest ===")
    print(stats)

    # Report P&L in dollari
    n_trades     = stats["# Trades"]
    equity_final = stats["Equity Final [$]"]
    pnl_dollar   = equity_final - INITIAL_CASH

    print(f"\n{'=' * 45}")
    print(f"  RIEPILOGO P&L IN DOLLARI")
    print(f"{'=' * 45}")
    print(f"  Capitale iniziale:   ${INITIAL_CASH:>12,.2f}")
    print(f"  Capitale finale:     ${equity_final:>12,.2f}")

    if n_trades > 0:
        commissions = stats.get("Commissions [$]", 0)
        pnl_gross   = pnl_dollar + commissions
        best_pct    = stats["Best Trade [%]"]
        worst_pct   = stats["Worst Trade [%]"]
        print(f"  P&L lordo:           ${pnl_gross:>12,.2f}")
        print(f"  Commissioni:        -${commissions:>12,.2f}")
        print(f"  P&L netto:           ${pnl_dollar:>+12,.2f}")
        print(f"  Trade totali:         {n_trades:>11}")
        print(f"  Miglior trade:       ~${INITIAL_CASH * best_pct / 100:>+11,.2f} ({best_pct:+.2f}%)")
        print(f"  Peggior trade:       ~${INITIAL_CASH * worst_pct / 100:>+11,.2f} ({worst_pct:+.2f}%)")
    else:
        print(f"  P&L netto:           ${pnl_dollar:>+12,.2f}")
        print(f"  Trade totali:         {n_trades:>11}")
        print(f"  [WARN] Nessun trade eseguito.")

    print(f"{'=' * 45}")

    bt.plot()

    return stats


# ─────────────────────────────────────────────
# 5. PAPER TRADING LIVE (Binance Testnet)
# ─────────────────────────────────────────────

def get_testnet_exchange(exchange_type="spot"):
    """
    Ritorna un'istanza ccxt connessa al Binance Demo Trading.
    Scalping: se USE_FUTURES_FOR_BOTH=True, ritorna sempre Futures
    (fee 0.02% maker vs 0.1% spot — critico per scalping).
    exchange_type: "spot" per LONG, "future" per SHORT.
    """
    # In modalita' Futures-only, anche i LONG passano per Futures
    if USE_FUTURES_FOR_BOTH or exchange_type == "future":
        exchange = ccxt.binance({
            "apiKey": TESTNET_API_KEY,
            "secret": TESTNET_SECRET,
            "options": {
                "defaultType": "future",
                "recvWindow": 10000,
                "adjustForTimeDifference": True,
            },
        })
        exchange.enable_demo_trading(True)
        exchange.load_markets()
        # Leva 1x: nessun leverage, puro trading senza margine amplificato
        try:
            exchange.set_leverage(1, SYMBOL)
            print(f"Futures: leverage impostato a 1x per {SYMBOL}")
        except Exception as e:
            raise RuntimeError(
                f"Impossibile impostare leverage 1x su Futures: {e}. "
                f"Bot non avviato per sicurezza (rischio leverage non controllato)."
            )
        return exchange
    else:
        exchange = ccxt.binance({
            "apiKey": TESTNET_API_KEY,
            "secret": TESTNET_SECRET,
            "options": {
                "defaultType": "spot",
                "fetchMarkets": {"types": ["spot"]},
                "recvWindow": 10000,
                "adjustForTimeDifference": True,
            },
        })
        exchange.enable_demo_trading(True)
        exchange.load_markets()
        return exchange


def log_trade(side, qty, price, signal, confidence, pnl_usd=None):
    """
    Appende un trade al file CSV di log.
    """
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "side", "qty",
                "price", "signal", "confidence", "pnl_usd"
            ])
        writer.writerow([
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            SYMBOL, side, qty, f"{price:.2f}",
            signal, f"{confidence:.4f}",
            f"{pnl_usd:.2f}" if pnl_usd is not None else ""
        ])


# ─────────────────────────────────────────────
# 5b. NOTIFICHE TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(message):
    """
    Invia un messaggio tramite Telegram Bot API.
    Non solleva eccezioni se il messaggio non viene inviato.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = (
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            f"?chat_id={TELEGRAM_CHAT}"
            f"&parse_mode=HTML"
            f"&text={urllib.request.quote(message)}"
        )
        urllib.request.urlopen(url, timeout=10)
    except Exception:
        pass  # non bloccare il bot per errori Telegram


# ─────────────────────────────────────────────
# 5c. PERSISTENZA STATO E MODELLO
# ─────────────────────────────────────────────

def save_state(entry_price, entry_qty, entry_time=None, position_type=None, trail_stop=None,
               consecutive_sl=0, daily_pnl=0.0, daily_reset_date=None, pause_until=None,
               cooldown_until=None):
    """Salva lo stato della posizione, circuit breaker e cooldown su file JSON."""
    with open(STATE_FILE, "w") as f:
        json.dump({
            "entry_price": entry_price,
            "entry_qty": entry_qty,
            "entry_time": entry_time.isoformat() if entry_time else None,
            "position_type": position_type,  # "long", "short", or None
            "trail_stop": trail_stop,         # trailing stop price
            # Circuit breaker state
            "consecutive_sl": consecutive_sl,
            "daily_pnl": daily_pnl,
            "daily_reset_date": daily_reset_date,
            "pause_until": pause_until,
            # Cooldown post-uscita tecnica
            "cooldown_until": cooldown_until,
        }, f)


def load_state():
    """
    Carica lo stato della posizione, circuit breaker e cooldown da file JSON.
    Ritorna (entry_price, entry_qty, entry_time, position_type, trail_stop,
             consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until).
    Backward compat: vecchi file senza i nuovi campi usano valori di default.
    """
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            et = s.get("entry_time")
            entry_time = datetime.fromisoformat(et) if et else None
            ep = s.get("entry_price")
            # Backward compat: se entry_price esiste ma position_type manca, assume "long"
            pt = s.get("position_type")
            if ep and not pt:
                pt = "long"
            ts = s.get("trail_stop")
            return (ep, s.get("entry_qty"), entry_time, pt, ts,
                    s.get("consecutive_sl", 0), s.get("daily_pnl", 0.0),
                    s.get("daily_reset_date"), s.get("pause_until"),
                    s.get("cooldown_until"))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return None, None, None, None, None, 0, 0.0, None, None, None


def save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                        entry_price, entry_qty, features_row,
                        short_proba=0.0, position_type=None,
                        action_taken="", reason="",
                        eff_buy_thresh=None, eff_short_thresh=None):
    """Salva snapshot del ciclo corrente per la dashboard web."""
    # Accumula ultimi 20 prezzi per sparkline e ultimi 50 signal log
    sparkline = []
    signal_log = []
    if os.path.isfile(DASHBOARD_FILE):
        try:
            with open(DASHBOARD_FILE) as f:
                old = json.load(f)
            sparkline = old.get("sparkline", [])
            signal_log = old.get("signal_log", [])
        except (json.JSONDecodeError, KeyError):
            pass
    sparkline.append(round(price, 2))
    sparkline = sparkline[-20:]

    # Calcola P&L correttamente per LONG e SHORT
    pnl_pct = None
    pnl_usd = None
    if entry_price and position_type == "short":
        pnl_pct = round((entry_price - price) / entry_price, 4)
        pnl_usd = round((entry_price - price) * (entry_qty or 0), 2)
    elif entry_price:
        pnl_pct = round((price - entry_price) / entry_price, 4)
        pnl_usd = round((price - entry_price) * (entry_qty or 0), 2)

    # Determina trend
    trend_up = features_row.get("trend_up") if features_row else None

    # Aggiungi entry al signal log
    signal_log.append({
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "price": round(price, 2),
        "buy_p": round(buy_proba * 100, 1),
        "short_p": round(short_proba * 100, 1),
        "signal": signal_str,
        "action": action_taken or "HOLD",
        "reason": reason,
        "trend": "UP" if trend_up == 1 else ("DOWN" if trend_up == 0 else "?"),
        "pos": position_type.upper() if position_type else "FLAT",
    })
    signal_log = signal_log[-50:]

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": price,
        "buy_proba": round(buy_proba, 4),
        "short_proba": round(short_proba, 4),
        "signal": signal_str,
        "position_type": position_type,
        "usdt": round(usdt, 2),
        "btc": round(btc, 8),
        "entry_price": entry_price,
        "entry_qty": entry_qty,
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "features": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in features_row.items()},
        "sparkline": sparkline,
        "signal_log": signal_log,
        "sleep_seconds": SLEEP_SECONDS,
        "stop_loss": STOP_LOSS,
        "min_proba": eff_buy_thresh if eff_buy_thresh is not None else MIN_PROBA,
        "short_min_proba": (eff_short_thresh if eff_short_thresh is not None else SHORT_MIN_PROBA) if ENABLE_SHORT else None,
        "trend_up": trend_up,
    }
    with open(DASHBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_price_history(df):
    """Salva le ultime 100 candele OHLCV per il chart candlestick della dashboard."""
    try:
        recent = df.tail(100)[["open", "high", "low", "close", "volume"]].copy()
        recent["timestamp"] = recent.index.strftime("%Y-%m-%dT%H:%M:%S")
        with open(PRICE_HISTORY_FILE, "w") as f:
            json.dump(recent.to_dict(orient="records"), f)
    except Exception as e:
        print(f"[WARN] save_price_history fallito: {e}")


def save_model(model_buy, model_short=None):
    """Salva i modelli su disco con joblib (dual model: BUY + SHORT)."""
    joblib.dump(model_buy, MODEL_BUY_FILE)
    print(f"Modello BUY salvato in {MODEL_BUY_FILE}")
    if model_short is not None:
        joblib.dump(model_short, MODEL_SHORT_FILE)
        print(f"Modello SHORT salvato in {MODEL_SHORT_FILE}")


def load_model():
    """
    Carica i modelli da disco se esistono e hanno meno di RETRAIN_HOURS ore.
    Ritorna (model_buy, model_short) o (None, None).
    Backward compat: se esiste il vecchio model.joblib, lo usa come model_buy.
    """
    model_buy = None
    model_short = None

    # Prova prima i nuovi file dual
    if os.path.isfile(MODEL_BUY_FILE):
        age_hours = (time.time() - os.path.getmtime(MODEL_BUY_FILE)) / 3600
        if age_hours > RETRAIN_HOURS:
            print(f"Modello BUY troppo vecchio ({age_hours:.1f}h). Ri-training necessario.")
            return None, None
        model_buy = joblib.load(MODEL_BUY_FILE)
        print(f"Modello BUY caricato da {MODEL_BUY_FILE} (eta': {age_hours:.1f}h)")
    elif os.path.isfile(MODEL_FILE):
        # Backward compat: vecchio file singolo
        age_hours = (time.time() - os.path.getmtime(MODEL_FILE)) / 3600
        if age_hours > RETRAIN_HOURS:
            print(f"Modello trovato ma troppo vecchio ({age_hours:.1f}h). Ri-training necessario.")
            return None, None
        model_buy = joblib.load(MODEL_FILE)
        print(f"Modello BUY caricato da {MODEL_FILE} (backward compat, eta': {age_hours:.1f}h)")
    else:
        return None, None

    # Carica SHORT se disponibile
    if os.path.isfile(MODEL_SHORT_FILE):
        age_hours_s = (time.time() - os.path.getmtime(MODEL_SHORT_FILE)) / 3600
        if age_hours_s <= RETRAIN_HOURS:
            model_short = joblib.load(MODEL_SHORT_FILE)
            print(f"Modello SHORT caricato da {MODEL_SHORT_FILE} (eta': {age_hours_s:.1f}h)")
        else:
            print(f"Modello SHORT troppo vecchio ({age_hours_s:.1f}h), SHORT disabilitato.")
    else:
        print("Modello SHORT non trovato, SHORT disabilitato nel bot live.")

    return model_buy, model_short


def retrain_model():
    """
    Scarica dati freschi, rigenera feature, allena nuovi modelli (BUY + SHORT)
    usando walk-forward + iperparametri Optuna (come il training iniziale).
    Usato per il retraining periodico nel bot live.
    """
    print("\n=== RETRAINING: scarico dati freschi ===")
    df_raw = fetch_ohlcv()
    save_price_history(df_raw)
    df_feat = build_features(df_raw)
    print(f"Retraining su {len(df_feat)} righe")

    # Carica iperparametri Optuna (o ri-ottimizza se cache scaduta)
    best_params_buy = optimize_hyperparams(df_feat, target_label=1, cache_file=OPTUNA_BUY_FILE)
    best_params_short = None
    if ENABLE_SHORT:
        best_params_short = optimize_hyperparams(df_feat, target_label=-1, cache_file=OPTUNA_SHORT_FILE)

    # Walk-forward training (come il training iniziale)
    model_buy, model_short, _ = walk_forward_train(
        df_feat, best_params_buy=best_params_buy, best_params_short=best_params_short
    )

    save_model(model_buy, model_short)
    send_telegram(
        f"🔄 <b>Retraining completato (walk-forward + Optuna)</b>\n"
        f"📊 Righe: {len(df_feat)} | BUY + {'SHORT' if model_short else 'LONG-only'}"
    )
    return model_buy, model_short


def _calc_pnl(price, entry_price, qty, position_type):
    """Calcola P&L correttamente per LONG e SHORT."""
    if position_type == "short":
        pnl_usd = (entry_price - price) * qty
        pnl_pct = (entry_price - price) / entry_price
    else:
        pnl_usd = (price - entry_price) * qty
        pnl_pct = (price - entry_price) / entry_price
    return pnl_usd, pnl_pct


def run_bot(model_buy, model_short=None):
    """
    Loop principale del bot scalping:
    - Futures-only: LONG e SHORT entrambi su Binance Futures Demo (fee 0.02%)
    - Ciclo ogni 60s: check stop/TP/trailing ogni minuto, segnali ML ogni 5m
    - Take profit 0.6%, stop loss 0.4%, trailing ATR×1.5
    - Hold minimo 15 min (3 barre da 5m)
    """
    # === Exchange setup (Futures-only per scalping) ===
    exchange_futures = get_testnet_exchange("future")
    short_enabled = ENABLE_SHORT and model_short is not None
    mode_str = "LONG+SHORT" if short_enabled else "LONG-only"
    print(f"\nBot SCALPING avviato su Binance Futures Demo | {SYMBOL} | {TIMEFRAME} | {mode_str}")
    print("=" * 55)

    # Carica stato precedente (sopravvive a restart)
    (entry_price, entry_qty, entry_time, position_type, trail_stop,
     consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until) = load_state()
    if entry_price:
        print(f"Stato caricato: {position_type.upper()} @ {entry_price:.2f} "
              f"({entry_qty} BTC, trail_stop: {trail_stop})")
    if consecutive_sl > 0 or pause_until:
        print(f"  [CB] SL consecutivi: {consecutive_sl} | daily P&L: ${daily_pnl:.2f} | pausa: {pause_until}")
    if cooldown_until:
        print(f"  [COOLDOWN] Cooldown post-uscita attivo fino: {cooldown_until}")

    last_retrain = time.time()
    last_status  = time.time()
    STATUS_INTERVAL = 86400

    # Soglie: usa quelle calibrate dall'ensemble se disponibili, altrimenti default
    eff_buy_thresh = getattr(model_buy, 'calibrated_threshold', MIN_PROBA)
    eff_short_thresh = getattr(model_short, 'calibrated_threshold', SHORT_MIN_PROBA) if model_short else SHORT_MIN_PROBA
    print(f"Soglie effettive: BUY >= {eff_buy_thresh:.2%} | SHORT >= {eff_short_thresh:.2%}")
    if eff_buy_thresh < MIN_PROBA:
        print(f"  [INFO] Soglia BUY calibrata ({eff_buy_thresh:.2%}) sotto default ({MIN_PROBA:.0%}) - compressione ensemble compensata")
    if eff_short_thresh < SHORT_MIN_PROBA:
        print(f"  [INFO] Soglia SHORT calibrata ({eff_short_thresh:.2%}) sotto default ({SHORT_MIN_PROBA:.0%}) - compressione ensemble compensata")

    send_telegram(
        f"🤖 <b>Bot SCALPING avviato ({mode_str})</b>\n"
        f"📈 {SYMBOL} | {TIMEFRAME} | Futures-only\n"
        f"🔄 Retraining ogni {RETRAIN_HOURS}h\n"
        f"🛡 SL: {STOP_LOSS:.1%} | TP: {TAKE_PROFIT:.1%} | BUY min: {eff_buy_thresh:.0%}"
        + (f" | SHORT min: {eff_short_thresh:.0%}" if short_enabled else "")
    )

    consecutive_net_errors = 0
    initial_capital = None  # catturato al primo ciclo dopo fetch balance

    # Cache dati e segnali (aggiornati ogni 5m)
    cached_df_feat = None
    cached_buy_proba = 0.0
    cached_short_proba = 0.0
    cached_buy_signal = False
    cached_short_signal = False
    cached_sell_signal = False
    cached_cover_signal = False
    cached_price = 0.0

    while True:
        try:
            # --- Retraining periodico ---
            hours_since = (time.time() - last_retrain) / 3600
            if hours_since >= RETRAIN_HOURS:
                print(f"\n[RETRAIN] {hours_since:.1f}h dall'ultimo training")
                try:
                    model_buy, model_short_new = retrain_model()
                    if model_short_new and short_enabled:
                        model_short = model_short_new
                    # Aggiorna soglie calibrate dopo retrain
                    eff_buy_thresh = getattr(model_buy, 'calibrated_threshold', MIN_PROBA)
                    eff_short_thresh = getattr(model_short, 'calibrated_threshold', SHORT_MIN_PROBA) if model_short else SHORT_MIN_PROBA
                    print(f"  Soglie aggiornate: BUY >= {eff_buy_thresh:.2%} | SHORT >= {eff_short_thresh:.2%}")
                except Exception as e:
                    print(f"[RETRAIN FALLITO] {e}")
                    send_telegram(
                        f"⚠️ <b>Retraining fallito</b>\n"
                        f"<code>{str(e)[:500]}</code>"
                    )
                last_retrain = time.time()

            # --- Gate 5m: genera segnali ML solo su chiusura candela 5m ---
            now = datetime.now(timezone.utc)
            is_5m_close = (now.minute % 5 == 0)

            if is_5m_close or cached_df_feat is None:
                df = fetch_ohlcv()
                save_price_history(df)
                cached_df_feat = build_features(df)

                last_row = cached_df_feat[FEATURES].iloc[-1:]
                cached_buy_proba = model_buy.predict_proba(last_row)[0][1]
                cached_buy_signal = cached_buy_proba >= eff_buy_thresh

                cached_short_proba = 0.0
                cached_short_signal = False
                if short_enabled and model_short is not None:
                    cached_short_proba = model_short.predict_proba(last_row)[0][1]
                    cached_short_signal = cached_short_proba >= eff_short_thresh

                # Diagnostica distanza dalla soglia
                print(f"  [LIVE-DIAG] BUY={cached_buy_proba:.2%} (soglia {eff_buy_thresh:.2%}, gap {cached_buy_proba - eff_buy_thresh:+.2%}) | "
                      f"SHORT={cached_short_proba:.2%} (soglia {eff_short_thresh:.2%}, gap {cached_short_proba - eff_short_thresh:+.2%})")

                # Segnali tecnici di chiusura
                cached_sell_signal = technical_sell_signal(cached_df_feat).iloc[-1]
                cached_cover_signal = technical_cover_signal(cached_df_feat).iloc[-1]

                # Conflict resolution
                if cached_buy_signal and cached_short_signal:
                    cached_buy_signal = False
                    cached_short_signal = False
                    print(f"  [CONFLICT] BUY ({cached_buy_proba:.0%}) + SHORT ({cached_short_proba:.0%}) -> HOLD")

                # Mutual exclusion
                if cached_buy_signal and position_type == "short":
                    cached_buy_signal = False
                if cached_short_signal and position_type == "long":
                    cached_short_signal = False

            # Usa segnali cached
            buy_signal = cached_buy_signal
            short_signal = cached_short_signal
            sell_signal = cached_sell_signal
            cover_signal = cached_cover_signal
            buy_proba = cached_buy_proba
            short_proba = cached_short_proba
            df_feat = cached_df_feat

            # --- Balance (Futures-only) ---
            price = df_feat["close"].iloc[-1]
            cached_price = price
            balance = exchange_futures.fetch_balance()
            usdt = balance.get("USDT", {}).get("free", 0)
            btc = 0.0  # Futures: posizione tracciata via entry_qty

            # --- Capitale iniziale (catturato una volta sola) ---
            if initial_capital is None and usdt > 0:
                initial_capital = usdt

            # --- Reset P&L giornaliero a mezzanotte UTC ---
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if daily_reset_date != today_str:
                daily_pnl = 0.0
                daily_reset_date = today_str

            # --- Log stato ---
            pos_str = f"{position_type.upper()}" if position_type else "FLAT"
            signal_parts = []
            if buy_signal: signal_parts.append(f"BUY({buy_proba:.0%})")
            if short_signal: signal_parts.append(f"SHORT({short_proba:.0%})")
            if sell_signal and position_type == "long": signal_parts.append("CLOSE_LONG")
            if cover_signal and position_type == "short": signal_parts.append("CLOSE_SHORT")
            signal_str = " | ".join(signal_parts) if signal_parts else "HOLD"
            print(
                f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] "
                f"Prezzo: {price:.2f} | {pos_str} | Signal: {signal_str} | "
                f"USDT: {usdt:.2f}"
            )

            # --- Diagnostica dettagliata su chiusura candela 5m ---
            if is_5m_close:
                trend_1h_diag = df_feat["trend_1h"].iloc[-1] if "trend_1h" in df_feat.columns else "N/A"
                adx_1h_diag = df_feat["adx_1h"].iloc[-1] if "adx_1h" in df_feat.columns else 0
                trend_str = "UP" if trend_1h_diag == 1 else "DOWN"
                print(
                    f"  [DIAG] BUY_proba: {buy_proba:.2%} (soglia {eff_buy_thresh:.0%}) | "
                    f"SHORT_proba: {short_proba:.2%} (soglia {eff_short_thresh:.0%}) | "
                    f"Trend_1h: {trend_str} | ADX_1h: {adx_1h_diag:.1f}"
                )

            # --- Variabili per tracking azione/motivo ---
            dash_action = "HOLD"
            dash_reason = ""
            features_dict = df_feat[FEATURES].iloc[-1].to_dict()
            features_dict["trend_up"] = float(df_feat["trend_up"].iloc[-1]) if "trend_up" in df_feat.columns else None

            # --- Notifica stato periodica ---
            if (time.time() - last_status) >= STATUS_INTERVAL:
                try:
                    now_str = pd.Timestamp.now().strftime("%H:%M")
                    if entry_price and position_type:
                        pnl_usd, pnl_pct = _calc_pnl(price, entry_price, entry_qty or 0, position_type)
                        pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                        pos_icon = "🟢" if position_type == "long" else "🔴"
                        send_telegram(
                            f"📊 <b>Status {now_str}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"{pos_icon} {position_type.upper()}\n"
                            f"📌 Entry: ${entry_price:,.2f}\n"
                            f"💰 Prezzo: ${price:,.2f}\n"
                            f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})\n"
                            f"🎯 BUY: {buy_proba:.0%} | SHORT: {short_proba:.0%}"
                        )
                    else:
                        send_telegram(
                            f"📊 <b>Status {now_str}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"⚪ Nessuna posizione\n"
                            f"💵 USDT: ${usdt:,.2f}\n"
                            f"💰 BTC: ${price:,.2f}\n"
                            f"🎯 BUY: {buy_proba:.0%} | SHORT: {short_proba:.0%}"
                        )
                except Exception as e:
                    print(f"[STATUS] Errore: {e}")
                last_status = time.time()

            # ============================================
            # TAKE PROFIT (scalping — priorita' massima)
            # ============================================
            cur_atr = df_feat["atr"].iloc[-1] if len(df_feat) > 0 else 0

            if position_type == "long" and entry_price:
                pnl_pct_now = (price - entry_price) / entry_price
                if pnl_pct_now >= TAKE_PROFIT:
                    qty = entry_qty or 0
                    pnl_usd = (price - entry_price) * qty
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "sell", qty)
                    except Exception as e:
                        print(f"  [ERRORE] TP LONG fallito: {e}")
                        send_telegram(f"🚨 <b>TP LONG fallito</b>\n<code>{e}</code>")
                        continue
                    dash_action = "TAKE_PROFIT_LONG"
                    dash_reason = f"TP {pnl_pct_now:.2%} >= {TAKE_PROFIT:.1%}"
                    entry_price = None; entry_qty = None; entry_time = None; position_type = None; trail_stop = None
                    consecutive_sl = 0
                    daily_pnl += pnl_usd
                    cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    print(f"  -> TAKE PROFIT LONG: SELL {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                    log_trade("SELL(TP)", qty, price, "TAKE_PROFIT", buy_proba, pnl_usd)
                    send_telegram(
                        f"🎯 <b>TAKE PROFIT LONG</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🔴 SELL {qty} BTC @ ${price:,.2f}\n"
                        f"📈 P&amp;L: {pnl_pct_now:.2%} | ${pnl_usd:+,.2f}"
                    )
                    try:
                        save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                            None, None, features_dict, short_proba, None,
                            dash_action, dash_reason, eff_buy_thresh, eff_short_thresh)
                    except Exception:
                        pass
                    time.sleep(SLEEP_SECONDS)
                    continue

            if position_type == "short" and entry_price:
                pnl_pct_now = (entry_price - price) / entry_price
                if pnl_pct_now >= TAKE_PROFIT:
                    qty = entry_qty or 0
                    pnl_usd = (entry_price - price) * qty
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "buy", qty)
                    except Exception as e:
                        print(f"  [ERRORE] TP SHORT fallito: {e}")
                        send_telegram(f"🚨 <b>TP SHORT fallito</b>\n<code>{e}</code>")
                        continue
                    dash_action = "TAKE_PROFIT_SHORT"
                    dash_reason = f"TP {pnl_pct_now:.2%} >= {TAKE_PROFIT:.1%}"
                    entry_price = None; entry_qty = None; entry_time = None; position_type = None; trail_stop = None
                    consecutive_sl = 0
                    daily_pnl += pnl_usd
                    cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    print(f"  -> TAKE PROFIT SHORT: BUY {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                    log_trade("BUY(TP)", qty, price, "TAKE_PROFIT", short_proba, pnl_usd)
                    send_telegram(
                        f"🎯 <b>TAKE PROFIT SHORT</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🟢 BUY(cover) {qty} BTC @ ${price:,.2f}\n"
                        f"📈 P&amp;L: {pnl_pct_now:.2%} | ${pnl_usd:+,.2f}"
                    )
                    try:
                        save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                            None, None, features_dict, short_proba, None,
                            dash_action, dash_reason, eff_buy_thresh, eff_short_thresh)
                    except Exception:
                        pass
                    time.sleep(SLEEP_SECONDS)
                    continue

            # ============================================
            # TRAILING STOP / STOP LOSS
            # ============================================

            # Trailing stop / Stop loss LONG
            if position_type == "long" and entry_price:
                pnl_pct_now = (price - entry_price) / entry_price
                stop_triggered = False

                if pnl_pct_now >= TRAIL_ACTIVATE:
                    new_trail = price - cur_atr * TRAIL_ATR_MULT
                    if trail_stop is None:
                        trail_stop = max(new_trail, entry_price)
                    else:
                        trail_stop = max(trail_stop, new_trail)
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    if price <= trail_stop:
                        stop_triggered = True
                        dash_action = "TRAIL_STOP_LONG"
                        dash_reason = f"Price ${price:,.0f} <= trail ${trail_stop:,.0f} (ATR×{TRAIL_ATR_MULT})"
                else:
                    if pnl_pct_now <= -STOP_LOSS:
                        stop_triggered = True
                        dash_action = "STOP_LOSS_LONG"
                        dash_reason = f"Loss {pnl_pct_now:.2%} <= -{STOP_LOSS:.1%}"

                if stop_triggered:
                    qty = entry_qty or 0
                    pnl_usd = (price - entry_price) * qty
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "sell", qty)
                    except Exception as e:
                        print(f"  [ERRORE] Stop LONG fallito: {e}")
                        send_telegram(f"🚨 <b>Stop LONG fallito</b>\n<code>{e}</code>")
                        continue
                    entry_price = None; entry_qty = None; entry_time = None; position_type = None; trail_stop = None
                    daily_pnl += pnl_usd
                    if "STOP_LOSS" in dash_action:
                        consecutive_sl += 1
                        if consecutive_sl >= 3:
                            pause_until = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
                            print(f"  [CB] {consecutive_sl} SL consecutivi -> pausa 2h")
                            send_telegram(f"⛔ <b>Circuit breaker LONG</b>: {consecutive_sl} SL di fila, pausa 2h")
                        cap = initial_capital if initial_capital else (usdt + abs(daily_pnl))
                        if cap > 0 and daily_pnl / cap <= -0.025:
                            midnight = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                                        + timedelta(days=1))
                            pause_until = midnight.isoformat()
                            print(f"  [CB] Daily P&L {daily_pnl:.2f} <= -2.5% capitale -> pausa fino mezzanotte")
                            send_telegram(f"⛔ <b>Circuit breaker daily</b>: P&amp;L giornaliero ${daily_pnl:.2f} (-2.5%), pausa fino mezzanotte UTC")
                    else:  # trailing stop (trade prima era in profitto)
                        if pnl_usd > 0:
                            consecutive_sl = 0
                        cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    print(f"  -> {dash_action}: SELL {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                    log_trade("SELL(TS)" if "TRAIL" in dash_action else "SELL(SL)", qty, price, dash_action, buy_proba, pnl_usd)
                    send_telegram(
                        f"🛑 <b>{dash_action.replace('_', ' ')}</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🔴 SELL {qty} BTC @ ${price:,.2f}\n"
                        f"📉 P&amp;L: {pnl_pct_now:.2%} | ${pnl_usd:+,.2f}"
                    )
                    try:
                        save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                            None, None, features_dict, short_proba, None,
                            dash_action, dash_reason, eff_buy_thresh, eff_short_thresh)
                    except Exception:
                        pass
                    time.sleep(SLEEP_SECONDS)
                    continue

            # Trailing stop / Stop loss SHORT
            if position_type == "short" and entry_price:
                pnl_pct_now = (entry_price - price) / entry_price
                stop_triggered = False

                if pnl_pct_now >= TRAIL_ACTIVATE:
                    new_trail = price + cur_atr * TRAIL_ATR_MULT
                    if trail_stop is None:
                        trail_stop = min(new_trail, entry_price)
                    else:
                        trail_stop = min(trail_stop, new_trail)
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    if price >= trail_stop:
                        stop_triggered = True
                        dash_action = "TRAIL_STOP_SHORT"
                        dash_reason = f"Price ${price:,.0f} >= trail ${trail_stop:,.0f} (ATR×{TRAIL_ATR_MULT})"
                else:
                    if pnl_pct_now <= -SHORT_STOP_LOSS:
                        stop_triggered = True
                        dash_action = "STOP_LOSS_SHORT"
                        dash_reason = f"Loss {pnl_pct_now:.2%} <= -{SHORT_STOP_LOSS:.1%}"

                if stop_triggered:
                    qty = entry_qty or 0
                    pnl_usd = (entry_price - price) * qty
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "buy", qty)
                    except Exception as e:
                        print(f"  [ERRORE] Stop SHORT fallito: {e}")
                        send_telegram(f"🚨 <b>Stop SHORT fallito</b>\n<code>{e}</code>")
                        continue
                    entry_price = None; entry_qty = None; entry_time = None; position_type = None; trail_stop = None
                    daily_pnl += pnl_usd
                    if "STOP_LOSS" in dash_action:
                        consecutive_sl += 1
                        if consecutive_sl >= 3:
                            pause_until = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
                            print(f"  [CB] {consecutive_sl} SL consecutivi -> pausa 2h")
                            send_telegram(f"⛔ <b>Circuit breaker SHORT</b>: {consecutive_sl} SL di fila, pausa 2h")
                        cap = initial_capital if initial_capital else (usdt + abs(daily_pnl))
                        if cap > 0 and daily_pnl / cap <= -0.025:
                            midnight = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                                        + timedelta(days=1))
                            pause_until = midnight.isoformat()
                            print(f"  [CB] Daily P&L {daily_pnl:.2f} <= -2.5% capitale -> pausa fino mezzanotte")
                            send_telegram(f"⛔ <b>Circuit breaker daily</b>: P&amp;L giornaliero ${daily_pnl:.2f} (-2.5%), pausa fino mezzanotte UTC")
                    else:  # trailing stop (trade prima era in profitto)
                        if pnl_usd > 0:
                            consecutive_sl = 0
                        cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    print(f"  -> {dash_action}: BUY {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f})")
                    log_trade("BUY(TS)" if "TRAIL" in dash_action else "BUY(SL)", qty, price, dash_action, short_proba, pnl_usd)
                    send_telegram(
                        f"🛑 <b>{dash_action.replace('_', ' ')}</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🟢 BUY(cover) {qty} BTC @ ${price:,.2f}\n"
                        f"📈 P&amp;L: {pnl_pct_now:.2%} | ${pnl_usd:+,.2f}"
                    )
                    try:
                        save_dashboard_data(price, buy_proba, signal_str, usdt, btc,
                            None, None, features_dict, short_proba, None,
                            dash_action, dash_reason, eff_buy_thresh, eff_short_thresh)
                    except Exception:
                        pass
                    time.sleep(SLEEP_SECONDS)
                    continue

            # ============================================
            # HOLD MINIMO (sopprime chiusura tecnica, in minuti per scalping)
            # ============================================
            if sell_signal and position_type == "long" and entry_time:
                mins_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
                bars_held = mins_held / 5  # 1 barra = 5 minuti
                if bars_held < MIN_HOLD_BARS:
                    sell_signal = False
                    dash_reason = f"CLOSE soppresso: hold {mins_held:.0f}m < {MIN_HOLD_BARS * 5}m min"
                    print(f"  -> CLOSE LONG soppresso: hold {mins_held:.0f}m < {MIN_HOLD_BARS * 5}m")

            if cover_signal and position_type == "short" and entry_time:
                mins_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
                bars_held = mins_held / 5
                if bars_held < SHORT_MIN_HOLD:
                    cover_signal = False
                    dash_reason = f"COVER soppresso: hold {mins_held:.0f}m < {SHORT_MIN_HOLD * 5}m min"
                    print(f"  -> CLOSE SHORT soppresso: hold {mins_held:.0f}m < {SHORT_MIN_HOLD * 5}m")

            # ============================================
            # PROFIT GATE: sopprimi chiusura tecnica in zona di piccola perdita
            # (consenti recupero prima dello SL; il backstop hard SL rimane sempre attivo)
            # ============================================
            if sell_signal and position_type == "long" and entry_price:
                pnl_pct_now = (price - entry_price) / entry_price
                if -0.003 < pnl_pct_now < 0:
                    sell_signal = False
                    dash_reason = f"CLOSE soppresso: loss piccola ({pnl_pct_now:.2%}), attesa recupero"
                    print(f"  -> CLOSE LONG soppresso: loss piccola {pnl_pct_now:.2%}, attesa recupero")

            if cover_signal and position_type == "short" and entry_price:
                pnl_pct_now = (entry_price - price) / entry_price
                if -0.003 < pnl_pct_now < 0:
                    cover_signal = False
                    dash_reason = f"COVER soppresso: loss piccola ({pnl_pct_now:.2%}), attesa recupero"
                    print(f"  -> CLOSE SHORT soppresso: loss piccola {pnl_pct_now:.2%}, attesa recupero")

            # ============================================
            # APERTURA POSIZIONI (solo se flat, solo su chiusura candela 5m)
            # ============================================

            # Circuit breaker: controlla se trading e' in pausa
            cb_paused = False
            if pause_until:
                try:
                    pause_dt = datetime.fromisoformat(pause_until)
                    if datetime.now(timezone.utc) < pause_dt:
                        cb_paused = True
                        if is_5m_close:
                            print(f"  [CB] Trading in pausa fino {pause_dt.strftime('%H:%M UTC')}")
                            dash_reason = f"Circuit breaker: pausa fino {pause_dt.strftime('%H:%M UTC')}"
                    else:
                        # Pausa scaduta: resetta
                        pause_until = None
                except (ValueError, TypeError):
                    pause_until = None

            # Cooldown post-uscita tecnica: risolvi timestamp scaduti
            if cooldown_until:
                try:
                    cd_dt = datetime.fromisoformat(cooldown_until)
                    if datetime.now(timezone.utc) >= cd_dt:
                        cooldown_until = None
                except (ValueError, TypeError):
                    cooldown_until = None

            # Leggi filtro trend 1h, 15m e ADX 1h
            trend_1h_val = df_feat["trend_1h"].iloc[-1] if "trend_1h" in df_feat.columns else 0
            trend_15m_val = df_feat["trend_15m"].iloc[-1] if "trend_15m" in df_feat.columns else 0
            adx_1h_val = df_feat["adx_1h"].iloc[-1] if "adx_1h" in df_feat.columns else 0

            if is_5m_close and buy_signal and usdt > 10 and not position_type and not cb_paused and not cooldown_until:
                if trend_1h_val != 1 or trend_15m_val != 1:
                    dash_action = "BUY_BLOCKED"
                    dash_reason = f"Trend 1h/15m non UP (1h={trend_1h_val:.0f}, 15m={trend_15m_val:.0f}), BUY {buy_proba:.0%} bloccato"
                    print(f"  -> Trend 1h/15m ribassista, BUY bloccato.")
                else:
                    qty = round((usdt * TRADE_SIZE) / price, 6)
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "buy", qty)
                    except Exception as e:
                        print(f"  [ERRORE] BUY fallito: {e}")
                        send_telegram(f"🚨 <b>BUY fallito</b>\n<code>{e}</code>")
                        continue
                    dash_action = "OPEN_LONG"
                    dash_reason = f"BUY {buy_proba:.0%} >= {eff_buy_thresh:.0%}, trend 1h UP"
                    entry_price = price; entry_qty = qty; trail_stop = None
                    entry_time = datetime.now(timezone.utc); position_type = "long"
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    print(f"  -> ORDER: BUY {qty} BTC @ {price:.2f} (prob: {buy_proba:.0%})")
                    log_trade("BUY", qty, price, "BUY", buy_proba)
                    send_telegram(
                        f"🟢 <b>BUY eseguito (scalp)</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"🛒 {qty} BTC @ ${price:,.2f}\n"
                        f"🤖 Confidenza: {buy_proba:.0%}\n"
                        f"💵 Investito: ${qty * price:,.2f}\n"
                        f"🎯 TP: {TAKE_PROFIT:.1%} | SL: {STOP_LOSS:.1%}"
                    )

            elif is_5m_close and short_signal and short_enabled and usdt > 10 and not position_type and not cb_paused and not cooldown_until:
                if trend_1h_val != 0 and not (short_proba >= eff_short_thresh + 0.05 and trend_15m_val == 0):
                    dash_action = "SHORT_BLOCKED"
                    dash_reason = f"Trend 1h UP + 15m non ribassista, SHORT {short_proba:.0%} bloccato"
                    print(f"  -> Trend 1h rialzista + 15m non ribassista, SHORT bloccato.")
                elif adx_1h_val < SHORT_ADX_MIN:
                    dash_action = "SHORT_BLOCKED"
                    dash_reason = f"ADX 1h {adx_1h_val:.1f} < {SHORT_ADX_MIN}, trend debole"
                    print(f"  -> SHORT bloccato: ADX 1h {adx_1h_val:.1f} < {SHORT_ADX_MIN}")
                else:
                    qty = round((usdt * SHORT_TRADE_SIZE) / price, 6)
                    try:
                        exchange_futures.create_order(SYMBOL, "market", "sell", qty)
                    except Exception as e:
                        print(f"  [ERRORE] SHORT fallito: {e}")
                        send_telegram(f"🚨 <b>SHORT fallito</b>\n<code>{e}</code>")
                        continue
                    dash_action = "OPEN_SHORT"
                    dash_reason = f"SHORT {short_proba:.0%} >= {eff_short_thresh:.0%}, trend 1h DOWN, ADX {adx_1h_val:.0f}"
                    entry_price = price; entry_qty = qty; trail_stop = None
                    entry_time = datetime.now(timezone.utc); position_type = "short"
                    save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                               consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                    print(f"  -> ORDER: SHORT {qty} BTC @ {price:.2f} (prob: {short_proba:.0%}, ADX 1h: {adx_1h_val:.0f})")
                    log_trade("SHORT", qty, price, "SHORT", short_proba)
                    send_telegram(
                        f"🔴 <b>SHORT eseguito (scalp)</b>\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"📉 {qty} BTC @ ${price:,.2f}\n"
                        f"🤖 Confidenza: {short_proba:.0%}\n"
                        f"💵 Margine: ${qty * price:,.2f}\n"
                        f"🎯 TP: {TAKE_PROFIT:.1%} | SL: {SHORT_STOP_LOSS:.1%}"
                    )

            # ============================================
            # CHIUSURA POSIZIONI (segnali tecnici)
            # ============================================

            elif sell_signal and position_type == "long":
                qty = entry_qty or 0
                pnl_usd, pnl_pct = _calc_pnl(price, entry_price, qty, "long")
                try:
                    exchange_futures.create_order(SYMBOL, "market", "sell", qty)
                except Exception as e:
                    print(f"  [ERRORE] CLOSE LONG fallito: {e}")
                    send_telegram(f"🚨 <b>CLOSE LONG fallito</b>\n<code>{e}</code>")
                    continue
                dash_action = "CLOSE_LONG"
                dash_reason = f"SELL tecnico (RSI/MACD/EMA)"
                entry_price = None; entry_qty = None; entry_time = None; position_type = None; trail_stop = None
                daily_pnl += pnl_usd
                if pnl_usd > 0:
                    consecutive_sl = 0
                cooldown_mins = 5 if pnl_usd >= 0 else 15
                cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=cooldown_mins)).isoformat()
                save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                           consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                print(f"  -> ORDER: CLOSE LONG {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f}) | cooldown {cooldown_mins}m")
                log_trade("SELL", qty, price, "SELL_TECH", buy_proba, pnl_usd)
                pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                send_telegram(
                    f"🔴 <b>CLOSE LONG</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💼 {qty} BTC @ ${price:,.2f}\n"
                    f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})"
                )

            elif cover_signal and position_type == "short":
                qty = entry_qty or 0
                pnl_usd, pnl_pct = _calc_pnl(price, entry_price, qty, "short")
                try:
                    exchange_futures.create_order(SYMBOL, "market", "buy", qty)
                except Exception as e:
                    print(f"  [ERRORE] CLOSE SHORT fallito: {e}")
                    send_telegram(f"🚨 <b>CLOSE SHORT fallito</b>\n<code>{e}</code>")
                    continue
                dash_action = "CLOSE_SHORT"
                dash_reason = f"COVER tecnico (RSI/MACD/EMA)"
                entry_price = None; entry_qty = None; entry_time = None; position_type = None; trail_stop = None
                daily_pnl += pnl_usd
                if pnl_usd > 0:
                    consecutive_sl = 0
                cooldown_mins = 5 if pnl_usd >= 0 else 15
                cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=cooldown_mins)).isoformat()
                save_state(entry_price, entry_qty, entry_time, position_type, trail_stop,
                           consecutive_sl, daily_pnl, daily_reset_date, pause_until, cooldown_until)
                print(f"  -> ORDER: CLOSE SHORT {qty} BTC @ {price:.2f} (P&L: ${pnl_usd:+,.2f}) | cooldown {cooldown_mins}m")
                log_trade("COVER", qty, price, "COVER_TECH", short_proba, pnl_usd)
                pnl_icon = "📈" if pnl_usd >= 0 else "📉"
                send_telegram(
                    f"🟢 <b>CLOSE SHORT</b>\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"💼 BUY(cover) {qty} BTC @ ${price:,.2f}\n"
                    f"{pnl_icon} P&amp;L: {pnl_pct:+.2%} (${pnl_usd:+,.2f})"
                )

            else:
                # Determina motivo HOLD
                if position_type:
                    dash_reason = f"In posizione {position_type.upper()}, attesa segnale chiusura"
                elif buy_proba < eff_buy_thresh and short_proba < eff_short_thresh:
                    dash_reason = f"Confidenza bassa (BUY {buy_proba:.0%} < {eff_buy_thresh:.0%}, SHORT {short_proba:.0%} < {eff_short_thresh:.0%})"
                else:
                    dash_reason = "Nessun segnale attivo"
                if is_5m_close:
                    print(f"  -> HOLD (nessuna azione)")

            # --- Dashboard save (fine ciclo) ---
            try:
                save_dashboard_data(
                    price, buy_proba, signal_str, usdt, btc,
                    entry_price, entry_qty, features_dict,
                    short_proba, position_type,
                    dash_action, dash_reason,
                    eff_buy_thresh, eff_short_thresh
                )
            except Exception as e:
                print(f"[DASHBOARD] Errore: {e}")

            consecutive_net_errors = 0

        except TRANSIENT_ERRORS as e:
            consecutive_net_errors += 1
            tb_short = traceback.format_exc().strip().split("\n")
            tb_last = "\n".join(tb_short[-4:]) if len(tb_short) >= 4 else "\n".join(tb_short)
            err_class = " > ".join(
                c.__name__ for c in type(e).__mro__ if c is not object
            )
            err_msg = f"{type(e).__name__}: {e}"
            ts = datetime.now().strftime("%H:%M:%S")
            if consecutive_net_errors <= MAX_RETRIES:
                wait = RETRY_BACKOFF[consecutive_net_errors - 1]
                print(f"[RETRY {consecutive_net_errors}/{MAX_RETRIES}] {ts} {err_msg}")
                print(f"  -> Riprovo tra {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"[ERRORE] {ts} {err_msg} (dopo {MAX_RETRIES} tentativi)")
                print(f"  Traceback:\n{traceback.format_exc()}")
                send_telegram(
                    f"🚨 <b>Errore di rete</b> (dopo {MAX_RETRIES} tentativi)\n"
                    f"⏰ {ts}\n"
                    f"🔗 <b>Tipo:</b> <code>{err_class}</code>\n"
                    f"💬 <code>{err_msg[:300]}</code>\n"
                    f"📋 <b>Traceback:</b>\n<code>{tb_last[:300]}</code>"
                )
                consecutive_net_errors = 0

        except Exception as e:
            tb_short = traceback.format_exc().strip().split("\n")
            tb_last = "\n".join(tb_short[-4:]) if len(tb_short) >= 4 else "\n".join(tb_short)
            err_class = " > ".join(
                c.__name__ for c in type(e).__mro__ if c is not object
            )
            err_msg = f"{type(e).__name__}: {e}"
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[ERRORE] {ts} {err_msg}")
            print(f"  Tipo: {err_class}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            send_telegram(
                f"🚨 <b>Errore</b>\n"
                f"⏰ {ts}\n"
                f"🔗 <b>Tipo:</b> <code>{err_class}</code>\n"
                f"💬 <code>{err_msg[:300]}</code>\n"
                f"📋 <b>Traceback:</b>\n<code>{tb_last[:300]}</code>"
            )
            consecutive_net_errors = 0

        print(f"  -> Prossimo ciclo tra {SLEEP_SECONDS}s...\n")
        time.sleep(SLEEP_SECONDS)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    backtest_only = "--backtest" in sys.argv

    if backtest_only:
        # === Modalita' backtest-only: usa modelli salvati, skip training ===
        print("=== MODALITA' BACKTEST-ONLY ===\n")
        model_buy, model_short = load_model()
        if model_buy is None:
            print("ERRORE: nessun modello salvato. Esegui prima il training completo.")
            sys.exit(1)

        print("=== Fase 1 - download dati ===")
        df_raw = fetch_ohlcv()
        print(f"Scaricate {len(df_raw)} candele per {SYMBOL} ({TIMEFRAME})")

        print("\n=== Fase 2 - feature engineering ===")
        df_feat = build_features(df_raw)
        print(f"Dataset: {len(df_feat)} righe, {len(FEATURES)} features")

        # Walk-forward per generare predizioni out-of-sample (usa params cached)
        best_params_buy = optimize_hyperparams(df_feat, target_label=1, cache_file=OPTUNA_BUY_FILE)
        best_params_short = None
        if ENABLE_SHORT:
            best_params_short = optimize_hyperparams(df_feat, target_label=-1, cache_file=OPTUNA_SHORT_FILE)

        print("\n=== Fase 3 - walk-forward (solo predizioni, params cached) ===")
        model_buy, model_short, wf_predictions = walk_forward_train(
            df_feat, best_params_buy=best_params_buy, best_params_short=best_params_short
        )

        print("\n=== Fase 4 - backtest ===")
        backtest(df_raw, df_feat, model_buy, model_short, pred_series=wf_predictions)
        sys.exit(0)

    # === Modalita' normale: training completo + (opzionale) bot live ===
    model_buy, model_short = load_model()

    if model_buy is not None:
        print("Modello caricato da disco - skip training.")
        print(f"Per forzare il retraining, cancella {MODEL_BUY_FILE} e {MODEL_SHORT_FILE}\n")
        try:
            df_raw = fetch_ohlcv()
            save_price_history(df_raw)
        except Exception:
            pass
    else:
        print("=== CRYPTOBOT: fase 1 - download dati ===")
        df_raw  = fetch_ohlcv()
        save_price_history(df_raw)
        print(f"Scaricate {len(df_raw)} candele per {SYMBOL} ({TIMEFRAME})")

        print("\n=== CRYPTOBOT: fase 2 - feature engineering ===")
        df_feat = build_features(df_raw)
        print(f"Dataset dopo feature engineering: {len(df_feat)} righe, {len(FEATURES)} features")

        # Fase 2b: Ottimizzazione iperparametri (separata per BUY e SHORT)
        print("\n=== CRYPTOBOT: fase 2b - ottimizzazione iperparametri BUY ===")
        best_params_buy = optimize_hyperparams(df_feat, target_label=1, cache_file=OPTUNA_BUY_FILE)

        best_params_short = None
        if ENABLE_SHORT:
            print("\n=== CRYPTOBOT: fase 2b - ottimizzazione iperparametri SHORT ===")
            best_params_short = optimize_hyperparams(df_feat, target_label=-1, cache_file=OPTUNA_SHORT_FILE)

        # Fase 3: Walk-forward (dual model)
        print("\n=== CRYPTOBOT: fase 3 - walk-forward training (LONG+SHORT) ===")
        model_buy, model_short, wf_predictions = walk_forward_train(
            df_feat, best_params_buy=best_params_buy, best_params_short=best_params_short
        )

        # Feature importance BUY (ultimo fold)
        print("\n=== Feature Importance BUY (ultimo fold) ===")
        importances = model_buy.feature_importances_
        feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in feat_imp:
            bar = "#" * int(imp * 50)
            print(f"  {feat:>15}: {imp:.4f} {bar}")

        # Feature importance SHORT (se disponibile)
        if model_short is not None:
            print("\n=== Feature Importance SHORT (ultimo fold) ===")
            importances_s = model_short.feature_importances_
            feat_imp_s = sorted(zip(FEATURES, importances_s), key=lambda x: x[1], reverse=True)
            for feat, imp in feat_imp_s:
                bar = "#" * int(imp * 50)
                print(f"  {feat:>15}: {imp:.4f} {bar}")

        # Salva i modelli per riusi futuri
        save_model(model_buy, model_short)

        # Fase 4: Backtest
        print("\n=== CRYPTOBOT: fase 4 - backtest (walk-forward out-of-sample, LONG+SHORT) ===")
        backtest(df_raw, df_feat, model_buy, model_short, pred_series=wf_predictions)

    # Decommenta la riga sotto per avviare il bot live sul testnet
    #run_bot(model_buy, model_short)
