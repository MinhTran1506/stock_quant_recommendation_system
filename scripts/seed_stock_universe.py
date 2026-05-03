"""
scripts/seed_stock_universe.py — Seed Vietnam stock universe into the database.

Sources:
  1. Vietstock API (if credentials configured)
  2. vnstock open-source library (fallback for prototyping)
     https://github.com/thinh-vu/vnstock

Run: docker-compose exec backend python scripts/seed_stock_universe.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from sqlalchemy import select

from db.session import init_db, get_db
from db.models import Stock, Exchange


# ── Vietnam stock universe (representative subset for bootstrapping) ──────────
# In production, fetch this from Vietstock / FiinGroup API or vnstock
HOSE_STOCKS = [
    ("VNM", "Vinamilk", "Consumer Staples", "Food & Beverage"),
    ("VIC", "Vingroup", "Real Estate", "Diversified Real Estate"),
    ("VHM", "Vinhomes", "Real Estate", "Residential REIT"),
    ("HPG", "Hoa Phat Group", "Materials", "Steel"),
    ("FPT", "FPT Corporation", "Technology", "IT Services"),
    ("MWG", "Mobile World Investment", "Consumer Discretionary", "Electronics Retail"),
    ("TCB", "Techcombank", "Financials", "Banking"),
    ("VPB", "VPBank", "Financials", "Banking"),
    ("STB", "Sacombank", "Financials", "Banking"),
    ("MBB", "MBBank", "Financials", "Banking"),
    ("BID", "BIDV", "Financials", "Banking"),
    ("VCB", "Vietcombank", "Financials", "Banking"),
    ("CTG", "VietinBank", "Financials", "Banking"),
    ("ACB", "Asia Commercial Bank", "Financials", "Banking"),
    ("HDB", "HDBank", "Financials", "Banking"),
    ("SSI", "SSI Securities", "Financials", "Capital Markets"),
    ("VND", "VNDirect Securities", "Financials", "Capital Markets"),
    ("MSN", "Masan Group", "Consumer Staples", "Food & Beverage"),
    ("SAB", "Sabeco", "Consumer Staples", "Beverages"),
    ("GAS", "PV Gas", "Energy", "Oil & Gas"),
    ("PLX", "Petrolimex", "Energy", "Petroleum Retail"),
    ("POW", "PV Power", "Utilities", "Electric Utilities"),
    ("REE", "REE Corporation", "Industrials", "Engineering"),
    ("PNJ", "Phu Nhuan Jewelry", "Consumer Discretionary", "Jewelry"),
    ("KDH", "Khang Dien House", "Real Estate", "Residential"),
    ("DXG", "Dat Xanh Group", "Real Estate", "Diversified Real Estate"),
    ("NLG", "Nam Long Investment", "Real Estate", "Residential"),
    ("VRE", "Vincom Retail", "Real Estate", "Retail REIT"),
    ("DPM", "PetroVietnam Fertilizer", "Materials", "Chemicals"),
    ("DCM", "Phu My Fertilizer", "Materials", "Chemicals"),
]

HNX_STOCKS = [
    ("SHB", "Saigon-Hanoi Bank", "Financials", "Banking"),
    ("NVB", "Nam A Bank", "Financials", "Banking"),
    ("PVS", "PetroVietnam Technical Services", "Energy", "Oil Field Services"),
    ("VCS", "VICOSTONE", "Materials", "Building Materials"),
    ("TV2", "Tu Van Thiet Ke Xay Dung Dien", "Industrials", "Engineering"),
]


async def seed():
    await init_db()

    seeded = 0
    skipped = 0

    async for session in get_db():
        for ticker, name, sector, industry in HOSE_STOCKS:
            existing = await session.execute(
                select(Stock).where(Stock.ticker == ticker)
            )
            if existing.scalar_one_or_none():
                skipped += 1
                continue

            stock = Stock(
                ticker=ticker,
                name=name,
                exchange=Exchange.HOSE,
                sector=sector,
                industry=industry,
                is_active=True,
                listing_date=datetime(2000, 1, 1),
            )
            session.add(stock)
            seeded += 1

        for ticker, name, sector, industry in HNX_STOCKS:
            existing = await session.execute(
                select(Stock).where(Stock.ticker == ticker)
            )
            if existing.scalar_one_or_none():
                skipped += 1
                continue

            stock = Stock(
                ticker=ticker,
                name=name,
                exchange=Exchange.HNX,
                sector=sector,
                industry=industry,
                is_active=True,
                listing_date=datetime(2000, 1, 1),
            )
            session.add(stock)
            seeded += 1

        await session.commit()

    print(f"✅ Seeded {seeded} stocks, skipped {skipped} existing")


if __name__ == "__main__":
    asyncio.run(seed())
