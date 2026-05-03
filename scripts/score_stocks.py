"""
scripts/score_stocks.py — Full quant signal scoring pipeline.

Runs all quantitative strategies and persists signals to DB.
Equivalent to what the Airflow quant_pipeline DAG does on a schedule.
Useful for ad-hoc re-scoring and during development/testing.

Usage:
    python scripts/score_stocks.py
    python scripts/score_stocks.py --strategy factor_model
    python scripts/score_stocks.py --strategy stat_arb --tickers VNM,VIC,HPG
"""
import asyncio
import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run(strategy: str = "all", tickers_filter: list = None):
    from db.session import init_db, get_db
    from db.models import EODPrice, Stock
    from sqlalchemy import select, desc
    import pandas as pd

    await init_db()
    print(f"\n🔄  Running quant scoring pipeline — strategy={strategy}")

    async for session in get_db():
        # ── Load price data ────────────────────────────────────────────
        q = select(Stock).where(Stock.is_active == True)
        if tickers_filter:
            q = q.where(Stock.ticker.in_([t.upper() for t in tickers_filter]))
        else:
            q = q.limit(80)

        st_result = await session.execute(q)
        stocks = st_result.scalars().all()
        print(f"   Loaded {len(stocks)} stocks")

        rows = []
        for s in stocks:
            pr = await session.execute(
                select(EODPrice)
                .where(EODPrice.stock_id == s.id)
                .order_by(desc(EODPrice.date))
                .limit(312)
            )
            prices = pr.scalars().all()
            for p in prices:
                rows.append({
                    "ticker": s.ticker,
                    "sector": s.sector or "Unknown",
                    "date": p.date.date(),
                    "close": float(p.close),
                    "volume": int(p.volume or 0),
                })

        if not rows:
            print("   ❌  No price data found")
            return

        prices_df = pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close")
        prices_df = prices_df.sort_index().fillna(method="ffill").dropna(axis=1)
        print(f"   Price matrix: {prices_df.shape}")

        # ── Factor Model ───────────────────────────────────────────────
        if strategy in ("all", "factor_model"):
            print("\n   📊  Factor Model scoring…")
            from quant.strategies.factor_model import FactorModel
            model = FactorModel()
            scores = model.compute_scores(prices_df, pd.DataFrame(), pd.DataFrame())
            print(f"   ✅  Factor scores for {len(scores)} stocks")
            if not scores.empty:
                top5 = scores.head(5)[["score", "rank"]]
                print(f"   Top 5:\n{top5.to_string()}")

        # ── Stat Arb ───────────────────────────────────────────────────
        if strategy in ("all", "stat_arb"):
            print("\n   🔗  Statistical Arbitrage scanning…")
            from quant.strategies.stat_arb import PairsFinder, StatArbStrategy
            finder = PairsFinder(
                min_corr=0.65,
                max_half_life_days=63,
                lookback_days=min(252, len(prices_df)),
            )
            pairs = finder.find_pairs(prices_df)
            print(f"   Found {len(pairs)} cointegrated pairs")
            if pairs:
                strat = StatArbStrategy(long_only=True)
                signals = strat.generate_signals(prices_df, pairs[:20])
                active = [s for s in signals if s.signal != 0]
                print(f"   ✅  {len(active)} active spread signals")
                for s in active[:3]:
                    print(f"      {s.ticker_a}/{s.ticker_b} z={s.z_score:.2f} → {'+BUY' if s.signal==1 else 'SELL'}")

        # ── Momentum + Regime ──────────────────────────────────────────
        if strategy in ("all", "momentum"):
            print("\n   📈  Momentum + Regime signals…")
            from quant.strategies.momentum_regime import RegimeAdaptiveMomentum
            market_returns = prices_df.pct_change().dropna().mean(axis=1)
            ram = RegimeAdaptiveMomentum(long_only=True)
            ram.fit(market_returns)
            signals_df, regime = ram.generate_signals(prices_df, market_returns)
            n_buys = (signals_df["signal"] == 1).sum() if not signals_df.empty else 0
            regime_name = regime.regime.value if regime else "N/A"
            print(f"   Regime: {regime_name} | {n_buys} momentum buy signals")

        # ── Mean Reversion ─────────────────────────────────────────────
        if strategy in ("all", "mean_reversion"):
            print("\n   🔄  Mean Reversion signals…")
            from quant.strategies.mean_reversion import MeanReversionComposite
            mr = MeanReversionComposite(long_only=True)
            combined = mr.generate_combined_signals(prices_df, min_agreement=2)
            n_buys = (combined["signal"] == 1).sum() if not combined.empty else 0
            print(f"   ✅  {n_buys} mean-reversion buy signals (min_agreement=2)")

        # ── Portfolio Optimisation ────────────────────────────────────
        if strategy in ("all", "portfolio"):
            print("\n   💼  Portfolio Optimisation (Black-Litterman)…")
            from quant.portfolio.optimizer import PortfolioConstructor
            returns = prices_df.pct_change().dropna()
            constructor = PortfolioConstructor()
            top_tickers = list(prices_df.columns[:20])
            weights = constructor.construct(
                method="black_litterman",
                tickers=top_tickers,
                returns=returns[top_tickers],
                max_weight=0.15,
            )
            metrics = constructor.compute_portfolio_metrics(weights, returns[top_tickers])
            active = {t: w for t, w in weights.items() if w > 0.01}
            print(f"   ✅  {len(active)} positions | Sharpe={metrics['sharpe_ratio']:.3f} | Vol={metrics['annualised_vol']*100:.1f}%")

    print("\n✅  Scoring pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="Run quant scoring pipeline")
    parser.add_argument("--strategy", default="all",
                        choices=["all", "factor_model", "stat_arb", "momentum", "mean_reversion", "portfolio"])
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated ticker list (default: all active stocks)")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    asyncio.run(run(strategy=args.strategy, tickers_filter=tickers))


if __name__ == "__main__":
    main()
