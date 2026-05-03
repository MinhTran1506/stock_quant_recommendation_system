-- ─── TimescaleDB initialisation ─────────────────────────────────────────────
-- This script runs on first container start via docker-entrypoint-initdb.d

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create additional databases needed by services
CREATE DATABASE mlflow;
CREATE DATABASE airflow;

-- ─── Hypertables (run after Alembic creates the base tables) ─────────────────
-- These are converted with SELECT create_hypertable() after table creation.
-- See: backend/db/migrations/env.py for the migration that calls these.

-- eod_prices hypertable (partition by day)
-- SELECT create_hypertable('eod_prices', 'date', if_not_exists => TRUE);

-- intraday_prices hypertable (partition by hour for high-frequency data)
-- SELECT create_hypertable('intraday_prices', 'timestamp',
--     chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

-- orderbook_snapshots hypertable
-- SELECT create_hypertable('orderbook_snapshots', 'timestamp',
--     chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

-- predictions hypertable
-- SELECT create_hypertable('predictions', 'generated_at', if_not_exists => TRUE);

-- ─── Compression policies (7-day chunks, compress after 30 days) ─────────────
-- ALTER TABLE eod_prices SET (
--     timescaledb.compress,
--     timescaledb.compress_orderby = 'date DESC',
--     timescaledb.compress_segmentby = 'stock_id'
-- );
-- SELECT add_compression_policy('eod_prices', INTERVAL '30 days');

-- ALTER TABLE intraday_prices SET (
--     timescaledb.compress,
--     timescaledb.compress_orderby = 'timestamp DESC',
--     timescaledb.compress_segmentby = 'stock_id'
-- );
-- SELECT add_compression_policy('intraday_prices', INTERVAL '7 days');

-- ─── Continuous aggregates (materialised views) ───────────────────────────────
-- Daily OHLCV from intraday ticks (auto-refreshes every hour)
-- CREATE MATERIALIZED VIEW intraday_daily_ohlcv
-- WITH (timescaledb.continuous) AS
-- SELECT
--     stock_id,
--     time_bucket('1 day', timestamp) AS bucket,
--     FIRST(open, timestamp)          AS open,
--     MAX(high)                       AS high,
--     MIN(low)                        AS low,
--     LAST(close, timestamp)          AS close,
--     SUM(volume)                     AS volume
-- FROM intraday_prices
-- GROUP BY stock_id, bucket
-- WITH NO DATA;
-- SELECT add_continuous_aggregate_policy('intraday_daily_ohlcv',
--     start_offset => INTERVAL '3 days',
--     end_offset   => INTERVAL '1 hour',
--     schedule_interval => INTERVAL '1 hour');
