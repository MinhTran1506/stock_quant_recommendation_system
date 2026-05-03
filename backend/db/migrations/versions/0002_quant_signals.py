"""
db/migrations/versions/0002_quant_signals.py
Adds tables for quantitative strategy signal storage.

Includes:
  - quant_pairs            : tracked cointegrated pairs
  - quant_signals          : per-ticker strategy signals (all strategies)
  - quant_portfolio_weights: optimised portfolio weights snapshots
  - quant_regime_snapshots : historical HMM regime states
  - quant_risk_reports     : risk metric snapshots for monitoring
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON
import uuid

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── quant_pairs ────────────────────────────────────────────────────────────
    op.create_table(
        "quant_pairs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("ticker_a", sa.String(20), nullable=False),
        sa.Column("ticker_b", sa.String(20), nullable=False),
        sa.Column("hedge_ratio", sa.Float, nullable=False),
        sa.Column("spread_mean", sa.Float, default=0.0),
        sa.Column("spread_std", sa.Float, default=1.0),
        sa.Column("half_life_days", sa.Float),
        sa.Column("johansen_stat", sa.Float),         # Johansen trace statistic
        sa.Column("adf_pvalue", sa.Float),            # ADF p-value on spread
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("discovered_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("last_validated_at", sa.DateTime),
        sa.Column("metadata_", sa.String("metadata"), type_=JSON, default=dict),
    )
    op.create_unique_constraint(
        "uq_quant_pairs_tickers", "quant_pairs", ["ticker_a", "ticker_b"]
    )
    op.create_index("ix_quant_pairs_active", "quant_pairs", ["is_active"])

    # ── quant_signals (TimescaleDB hypertable) ─────────────────────────────────
    op.create_table(
        "quant_signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("generated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("signal", sa.Integer, nullable=False),   # +1 | 0 | -1
        sa.Column("confidence", sa.Float),
        sa.Column("z_score", sa.Float),
        sa.Column("indicator_value", sa.Float),
        sa.Column("metadata_", sa.String("metadata"), type_=JSON, default=dict),
    )
    op.create_index("ix_qs_ticker_strategy_time",
                    "quant_signals", ["ticker", "strategy", "generated_at"])
    # Convert to TimescaleDB hypertable
    op.execute(
        "SELECT create_hypertable('quant_signals', 'generated_at', "
        "chunk_time_interval => INTERVAL '1 week', if_not_exists => TRUE);"
    )

    # ── quant_portfolio_weights ────────────────────────────────────────────────
    op.create_table(
        "quant_portfolio_weights",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("method", sa.String(50), nullable=False),
        sa.Column("generated_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("weights", JSON, nullable=False),          # {ticker: weight}
        sa.Column("metrics", JSON, default=dict),            # sharpe, vol, etc.
        sa.Column("regime", sa.String(20)),
        sa.Column("n_stocks", sa.Integer),
    )
    op.create_index("ix_qpw_method_time", "quant_portfolio_weights", ["method", "generated_at"])

    # ── quant_regime_snapshots (TimescaleDB hypertable) ────────────────────────
    op.create_table(
        "quant_regime_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("snapshot_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("regime", sa.String(20), nullable=False),
        sa.Column("bull_prob", sa.Float),
        sa.Column("bear_prob", sa.Float),
        sa.Column("sideways_prob", sa.Float),
        sa.Column("momentum_scalar", sa.Float),
        sa.Column("vol_30d", sa.Float),
        sa.Column("trend_12m", sa.Float),
    )
    op.execute(
        "SELECT create_hypertable('quant_regime_snapshots', 'snapshot_at', "
        "chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);"
    )

    # ── quant_risk_reports (TimescaleDB hypertable) ────────────────────────────
    op.create_table(
        "quant_risk_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True),
                  sa.ForeignKey("portfolios.id"), nullable=True),
        sa.Column("snapshot_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("var_95_1d", sa.Float),
        sa.Column("cvar_95_1d", sa.Float),
        sa.Column("current_drawdown", sa.Float),
        sa.Column("max_drawdown", sa.Float),
        sa.Column("annualised_vol", sa.Float),
        sa.Column("sharpe_ratio", sa.Float),
        sa.Column("sortino_ratio", sa.Float),
        sa.Column("beta", sa.Float),
        sa.Column("breaches", JSON, default=list),
        sa.Column("action_required", sa.String(20), default="NONE"),
    )
    op.execute(
        "SELECT create_hypertable('quant_risk_reports', 'snapshot_at', "
        "chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);"
    )

    # ── quant_factor_scores ────────────────────────────────────────────────────
    op.create_table(
        "quant_factor_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("scored_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("composite_score", sa.Float),
        sa.Column("rank", sa.Integer),
        sa.Column("factor_mom", sa.Float),
        sa.Column("factor_value", sa.Float),
        sa.Column("factor_quality", sa.Float),
        sa.Column("factor_low_vol", sa.Float),
        sa.Column("factor_size", sa.Float),
        sa.Column("factor_growth", sa.Float),
        sa.Column("factor_liquidity", sa.Float),
    )
    op.create_index("ix_qfs_ticker_scored", "quant_factor_scores",
                    ["ticker", "scored_at"])
    op.execute(
        "SELECT create_hypertable('quant_factor_scores', 'scored_at', "
        "chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);"
    )


def downgrade() -> None:
    for table in [
        "quant_factor_scores", "quant_risk_reports",
        "quant_regime_snapshots", "quant_portfolio_weights",
        "quant_signals", "quant_pairs",
    ]:
        op.drop_table(table)
