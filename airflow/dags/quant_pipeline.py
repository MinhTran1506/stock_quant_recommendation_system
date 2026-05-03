"""
airflow/dags/quant_pipeline.py — Quantitative Strategy Execution DAG
═════════════════════════════════════════════════════════════════════

Schedules:
  - Daily (weekdays 16:00 ICT): factor scores, regime update, signals
  - Weekly (Sunday 02:00 ICT): strategy rebalancing, RL agent retrain
  - Intraday (every 30 min during session): microstructure signals refresh

DAG dependency graph:
  update_regime
       │
       ├─► compute_factor_scores ──► generate_factor_signals
       │
       ├─► run_stat_arb_scan ──► generate_stat_arb_signals
       │
       └─► run_momentum ──► generate_momentum_signals
                │
                └─► [weekly] rebalance_portfolios
                        │
                        └─► [weekly] retrain_rl_agent
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "quant_team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=1),
    "email_on_failure": True,
}


# ─── Daily quant signals DAG ──────────────────────────────────────────────────
with DAG(
    dag_id="quant_daily_signals",
    default_args=DEFAULT_ARGS,
    description="Daily quantitative strategy signal generation",
    schedule_interval="0 9 * * 1-5",   # 16:00 ICT = 09:00 UTC
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["quant", "signals", "daily"],
) as daily_dag:

    def update_market_regime(**ctx):
        """Fit HMM on latest market data and cache regime state."""
        import asyncio, sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        async def _run():
            from db.session import init_db, get_db
            from db.models import EODPrice, Stock
            from quant.strategies.momentum_regime import MarketRegimeDetector
            from sqlalchemy import select, desc
            import pandas as pd
            import json
            import redis.asyncio as aioredis
            from config import get_settings

            settings = get_settings()
            await init_db()

            async for session in get_db():
                # Build market proxy: equal-weighted returns of active stocks
                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True).limit(50)
                )
                stocks = st_result.scalars().all()

                rows = []
                for s in stocks:
                    pr = await session.execute(
                        select(EODPrice)
                        .where(EODPrice.stock_id == s.id)
                        .order_by(desc(EODPrice.date))
                        .limit(504)
                    )
                    prices = pr.scalars().all()
                    for p in prices:
                        rows.append({"ticker": s.ticker, "date": str(p.date.date()), "close": float(p.close)})

                if not rows:
                    print("No price data for regime detection")
                    return

                df = pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close")
                market_returns = df.pct_change().dropna().mean(axis=1)

                detector = MarketRegimeDetector(n_states=3)
                detector.fit(market_returns)
                state = detector.predict(market_returns)

                regime_data = {
                    "regime": state.regime.value,
                    "bull_prob": state.bull_prob,
                    "bear_prob": state.bear_prob,
                    "sideways_prob": state.sideways_prob,
                    "momentum_scalar": state.momentum_scalar,
                    "vol_30d": state.vol_30d,
                    "updated_at": datetime.utcnow().isoformat(),
                }

                # Cache regime in Redis (TTL: 26h to survive weekends)
                r = aioredis.from_url(settings.redis_url)
                await r.setex("quant:market_regime", 26 * 3600, json.dumps(regime_data))
                await r.aclose()
                print(f"Market regime: {state.regime.value} (bull={state.bull_prob:.2f})")

        asyncio.run(_run())

    def compute_factor_scores(**ctx):
        """Compute multi-factor alpha scores for all active stocks."""
        import asyncio, sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        async def _run():
            from db.session import init_db, get_db
            from db.models import EODPrice, Stock, Fundamental
            from quant.strategies.factor_model import FactorModel
            from sqlalchemy import select, desc
            import pandas as pd
            import json
            import redis.asyncio as aioredis
            from config import get_settings

            settings = get_settings()
            await init_db()

            async for session in get_db():
                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True)
                )
                stocks = st_result.scalars().all()

                price_rows = []
                for s in stocks:
                    pr = await session.execute(
                        select(EODPrice).where(EODPrice.stock_id == s.id)
                        .order_by(desc(EODPrice.date)).limit(252)
                    )
                    prices = pr.scalars().all()
                    for p in prices:
                        price_rows.append({"ticker": s.ticker, "date": str(p.date.date()), "close": float(p.close)})

                if not price_rows:
                    return

                prices_df = pd.DataFrame(price_rows).pivot(index="date", columns="ticker", values="close")

                # Load fundamentals
                fund_result = await session.execute(select(Fundamental))
                funds = fund_result.scalars().all()
                fund_df = pd.DataFrame([{
                    "ticker": str(f.stock_id),
                    "pb_ratio": f.pb_ratio,
                    "roe": f.roe,
                    "roa": f.roa,
                } for f in funds]).set_index("ticker") if funds else pd.DataFrame()

                model = FactorModel()
                scores = model.compute_scores(prices_df, fund_df, pd.DataFrame())

                # Cache in Redis
                r = aioredis.from_url(settings.redis_url)
                await r.setex(
                    "quant:factor_scores",
                    26 * 3600,
                    scores.reset_index().to_json(orient="records"),
                )
                await r.aclose()
                print(f"Factor scores computed for {len(scores)} stocks")

        asyncio.run(_run())

    def generate_stat_arb_signals(**ctx):
        """Scan for cointegrated pairs and generate spread signals."""
        import asyncio, sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        async def _run():
            from db.session import init_db, get_db
            from db.models import EODPrice, Stock
            from quant.strategies.stat_arb import PairsFinder, StatArbStrategy
            from sqlalchemy import select, desc
            import pandas as pd
            import json
            import redis.asyncio as aioredis
            from config import get_settings

            settings = get_settings()
            await init_db()

            async for session in get_db():
                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True).limit(60)
                )
                stocks = st_result.scalars().all()

                rows = []
                for s in stocks:
                    pr = await session.execute(
                        select(EODPrice).where(EODPrice.stock_id == s.id)
                        .order_by(desc(EODPrice.date)).limit(252)
                    )
                    prices = pr.scalars().all()
                    for p in prices:
                        rows.append({"ticker": s.ticker, "date": str(p.date.date()), "close": float(p.close)})

                if not rows:
                    return

                prices_df = pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close").fillna(method="ffill").dropna(axis=1)

                finder = PairsFinder(lookback_days=252)
                pairs = finder.find_pairs(prices_df)

                strategy = StatArbStrategy(long_only=True)
                signals = strategy.generate_signals(prices_df, pairs)

                signals_data = [
                    {
                        "ticker_a": s.ticker_a, "ticker_b": s.ticker_b,
                        "z_score": s.z_score, "spread": s.spread,
                        "hedge_ratio": s.hedge_ratio, "signal": s.signal,
                        "half_life": s.half_life, "timestamp": s.timestamp,
                    }
                    for s in signals
                ]

                r = aioredis.from_url(settings.redis_url)
                await r.setex("quant:stat_arb_signals", 26 * 3600, json.dumps(signals_data))
                await r.aclose()
                print(f"Stat arb: {len(pairs)} pairs, {len(signals)} signals")

        asyncio.run(_run())

    def generate_momentum_signals(**ctx):
        """Generate regime-adaptive momentum signals."""
        import asyncio, sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        async def _run():
            from db.session import init_db, get_db
            from db.models import EODPrice, Stock
            from quant.strategies.momentum_regime import RegimeAdaptiveMomentum
            from sqlalchemy import select, desc
            import pandas as pd
            import json
            import redis.asyncio as aioredis
            from config import get_settings

            settings = get_settings()
            await init_db()

            async for session in get_db():
                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True)
                )
                stocks = st_result.scalars().all()
                rows = []
                for s in stocks:
                    pr = await session.execute(
                        select(EODPrice).where(EODPrice.stock_id == s.id)
                        .order_by(desc(EODPrice.date)).limit(312)
                    )
                    prices = pr.scalars().all()
                    for p in prices:
                        rows.append({"ticker": s.ticker, "date": str(p.date.date()), "close": float(p.close)})

                if not rows:
                    return

                prices_df = pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close").fillna(method="ffill")
                market_returns = prices_df.pct_change().dropna().mean(axis=1)

                strategy = RegimeAdaptiveMomentum(long_only=True)
                strategy.fit(market_returns)
                signals_df, regime = strategy.generate_signals(prices_df, market_returns)

                result = {
                    "regime": regime.regime.value if regime else "BULL",
                    "momentum_scalar": regime.momentum_scalar if regime else 1.0,
                    "signals": signals_df[signals_df["signal"] == 1].reset_index().to_dict("records"),
                }

                r = aioredis.from_url(settings.redis_url)
                await r.setex("quant:momentum_signals", 26 * 3600, json.dumps(result))
                await r.aclose()
                print(f"Momentum: {len(result['signals'])} buy signals (regime={result['regime']})")

        asyncio.run(_run())

    def is_weekly_rebalance(**ctx):
        """Only run rebalancing on Mondays."""
        return ctx["logical_date"].weekday() == 0  # Monday

    def rebalance_portfolios(**ctx):
        """Rebalance paper portfolios using Black-Litterman on latest signals."""
        import asyncio, sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        async def _run():
            from db.session import init_db, get_db
            from db.models import Portfolio, Stock, EODPrice
            from quant.portfolio.optimizer import PortfolioConstructor
            from sqlalchemy import select, desc
            import pandas as pd
            import redis.asyncio as aioredis
            import json
            from config import get_settings

            settings = get_settings()
            await init_db()

            r = aioredis.from_url(settings.redis_url)

            # Load cached factor scores as BL views
            factor_cache = await r.get("quant:factor_scores")
            factor_df = pd.DataFrame(json.loads(factor_cache)).set_index("ticker") if factor_cache else pd.DataFrame()

            async for session in get_db():
                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True).limit(50)
                )
                stocks = st_result.scalars().all()
                tickers = [s.ticker for s in stocks]

                rows = []
                for s in stocks:
                    pr = await session.execute(
                        select(EODPrice).where(EODPrice.stock_id == s.id)
                        .order_by(desc(EODPrice.date)).limit(126)
                    )
                    prices = pr.scalars().all()
                    for p in prices:
                        rows.append({"ticker": s.ticker, "date": str(p.date.date()), "close": float(p.close)})

                if not rows:
                    await r.aclose()
                    return

                prices_df = pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close").fillna(method="ffill")
                returns = prices_df.pct_change().dropna()

                constructor = PortfolioConstructor()
                weights = constructor.construct(
                    method="black_litterman",
                    tickers=[t for t in tickers if t in prices_df.columns],
                    returns=returns,
                    factor_scores=factor_df if not factor_df.empty else None,
                )

                # Cache the weights
                await r.setex("quant:bl_weights", 7 * 24 * 3600, json.dumps(weights))
                print(f"BL rebalance: {len([w for w in weights.values() if w > 0.01])} active positions")

            await r.aclose()

        asyncio.run(_run())

    # ── Task definitions ──────────────────────────────────────────────────
    t_regime = PythonOperator(task_id="update_market_regime",    python_callable=update_market_regime)
    t_factor = PythonOperator(task_id="compute_factor_scores",   python_callable=compute_factor_scores)
    t_statarb = PythonOperator(task_id="generate_stat_arb_signals", python_callable=generate_stat_arb_signals)
    t_momentum = PythonOperator(task_id="generate_momentum_signals", python_callable=generate_momentum_signals)
    t_is_monday = ShortCircuitOperator(task_id="check_is_monday", python_callable=is_weekly_rebalance)
    t_rebalance = PythonOperator(task_id="rebalance_portfolios",  python_callable=rebalance_portfolios)

    # DAG flow
    t_regime >> [t_factor, t_statarb, t_momentum]
    t_factor >> t_momentum
    t_momentum >> t_is_monday >> t_rebalance
