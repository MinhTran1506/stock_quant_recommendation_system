# API Reference

Base URL: `http://localhost:8000`

All protected endpoints require `Authorization: Bearer <token>` in the request header.

---

## Authentication

### Obtain a Token

```
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

username=admin&password=yourpassword
```

Response:
```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer"
}
```

### Register a New User

```
POST /api/v1/auth/register
Content-Type: application/json

{ "username": "alice", "password": "secret", "email": "alice@example.com" }
```

### Get Current User Profile

```
GET /api/v1/auth/me
Authorization: Bearer <token>
```

### Change Password

```
PATCH /api/v1/auth/change-password
Authorization: Bearer <token>
Content-Type: application/json

{ "current_password": "old", "new_password": "new" }
```

---

## Stocks

Base path: `/api/v1/stocks`

All stock routes require authentication.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/stocks` | List all stocks in universe (with optional filters) |
| `GET` | `/api/v1/stocks/{ticker}` | Single stock detail (metadata, sector, exchange) |
| `GET` | `/api/v1/stocks/{ticker}/prices` | Historical OHLCV prices |
| `GET` | `/api/v1/stocks/{ticker}/signals` | Latest quant signals for a stock |

### Query Parameters — `GET /api/v1/stocks`

| Parameter | Type | Description |
|-----------|------|-------------|
| `exchange` | string | Filter by `HOSE` or `HNX` |
| `sector` | string | Filter by sector name |
| `limit` | int | Max results (default 100) |
| `offset` | int | Pagination offset |

### Query Parameters — `GET /api/v1/stocks/{ticker}/prices`

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | date | Start date (`YYYY-MM-DD`) |
| `end` | date | End date (`YYYY-MM-DD`) |
| `interval` | string | `daily` (default) or `intraday` |

---

## Predictions

Base path: `/api/v1/predictions`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/predictions/latest` | Latest predictions for all stocks |
| `GET` | `/api/v1/predictions/{ticker}` | Predictions for a specific stock |
| `GET` | `/api/v1/predictions/top` | Top-ranked stocks by composite score |

### Response — `/api/v1/predictions/{ticker}`

```json
{
  "ticker": "VNM",
  "score": 82.4,
  "model_predictions": {
    "tft": { "q10": -0.012, "q50": 0.018, "q90": 0.051 },
    "nbeats": { "q10": -0.009, "q50": 0.022, "q90": 0.048 }
  },
  "horizons": [1, 3, 5, 10, 20],
  "generated_at": "2024-01-15T18:45:00Z"
}
```

---

## Backtest

Base path: `/api/v1/backtest`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/backtest/run` | Submit a new backtest job |
| `GET` | `/api/v1/backtest/{job_id}` | Get backtest status and results |
| `GET` | `/api/v1/backtest/history` | List previous backtest runs |

### Request Body — `POST /api/v1/backtest/run`

```json
{
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 1000000000,
  "commission_pct": 0.0015,
  "slippage_pct": 0.001,
  "max_position_pct": 0.10,
  "max_positions": 20,
  "stop_loss_pct": 0.07,
  "rebalance_frequency": "weekly",
  "universe_filter": ["VNM", "VIC", "HPG", "VHM"],
  "strategy": "factor_model",
  "engine": "vectorbt"
}
```

**`strategy`** options: `factor_model`, `stat_arb`, `momentum_regime`, `mean_reversion`

**`engine`** options: `vectorbt` (fast), `backtrader` (realistic)

### Response — `GET /api/v1/backtest/{job_id}`

```json
{
  "job_id": "...",
  "status": "completed",
  "results": {
    "total_return_pct": 38.2,
    "annualised_return_pct": 11.4,
    "benchmark_return_pct": 8.1,
    "sharpe_ratio": 1.42,
    "sortino_ratio": 1.91,
    "max_drawdown_pct": 14.3,
    "win_rate": 0.57,
    "total_trades": 412,
    "equity_curve": [{"date": "2020-01-02", "value": 1000000000}, ...],
    "trade_log": [...]
  }
}
```

---

## Portfolio

Base path: `/api/v1/portfolio`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/portfolio` | List user's portfolios |
| `POST` | `/api/v1/portfolio` | Create a new portfolio |
| `GET` | `/api/v1/portfolio/{id}` | Portfolio detail (positions, weights, P&L) |
| `POST` | `/api/v1/portfolio/{id}/optimize` | Re-optimize weights |
| `GET` | `/api/v1/portfolio/{id}/risk` | Current risk report |

### Request Body — `POST /api/v1/portfolio/{id}/optimize`

```json
{
  "optimizer": "black_litterman",
  "risk_free_rate": 0.045,
  "risk_aversion": 2.5
}
```

**`optimizer`** options: `mean_variance`, `black_litterman`, `risk_parity`, `min_variance`, `max_diversification`

---

## Strategy

Base path: `/api/v1/strategy`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/strategy/signals` | Current signals from all active strategies |
| `GET` | `/api/v1/strategy/regime` | Current market regime (BULL/SIDEWAYS/BEAR) |
| `POST` | `/api/v1/strategy/run` | Trigger manual signal generation |

---

## Quant

Base path: `/api/v1/quant`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/quant/factors` | Factor scores for all stocks |
| `GET` | `/api/v1/quant/stat-arb` | Active cointegrated pairs |
| `GET` | `/api/v1/quant/momentum` | Momentum + regime state |
| `GET` | `/api/v1/quant/universe` | Filtered/ranked stock universe |

---

## Universe

Base path: `/api/v1/universe`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/universe` | Full stock universe with metadata |
| `POST` | `/api/v1/universe/filter` | Screen stocks by factor criteria |

### Request Body — `POST /api/v1/universe/filter`

```json
{
  "min_score": 60,
  "sectors": ["Consumer Staples", "Financials"],
  "exchange": "HOSE",
  "min_market_cap": 1000000000000,
  "top_n": 20
}
```

---

## News

Base path: `/api/v1/news`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/news` | Latest news articles with sentiment scores |
| `GET` | `/api/v1/news/{ticker}` | News for a specific stock |

---

## Health & Metrics

These endpoints are unauthenticated.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check `{"status": "ok"}` |
| `GET` | `/ready` | Readiness check (verifies DB + Redis) |
| `GET` | `/metrics` | Prometheus metrics in text format |

---

## WebSocket — Real-time Prices

```
ws://localhost:8000/ws/prices
```

Authentication: Pass the JWT token as a query parameter:

```
ws://localhost:8000/ws/prices?token=<access_token>
```

### Message Format (Server → Client)

```json
{
  "ticker": "VNM",
  "price": 78500,
  "change_pct": 1.23,
  "volume": 1234567,
  "timestamp": "2024-01-15T09:15:00+07:00"
}
```

---

## Error Codes

| HTTP Status | Meaning |
|-------------|---------|
| `400` | Bad request — invalid parameters |
| `401` | Unauthorized — missing or invalid token |
| `403` | Forbidden — insufficient permissions |
| `404` | Not found |
| `422` | Validation error — request body schema mismatch |
| `429` | Too many requests — rate limit exceeded |
| `500` | Internal server error |

---

## Curl Examples

```bash
# Set token once
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=yourpassword" | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# List stocks
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/stocks

# Get top predictions
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/predictions/top

# Run backtest
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"start_date":"2022-01-01","end_date":"2024-01-01","initial_capital":1000000000,"strategy":"factor_model","engine":"vectorbt"}' \
  http://localhost:8000/api/v1/backtest/run

# Get market regime
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/strategy/regime
```
