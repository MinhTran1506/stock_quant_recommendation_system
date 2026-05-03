"""
strategy/orchestrator.py — Strategy execution layer.

Architecture:
  StrategyOrchestrator
    └── reads model signals from Kafka / DB
    └── applies position sizing (Kelly, fixed-fractional, volatility-targeting)
    └── generates orders via ExecutionAdapter
          ├── PaperExecutionAdapter   (always available)
          └── LiveExecutionAdapter    (gated by LIVE_TRADING_ENABLED flag)

Vietnam regulatory note:
  Live automated order placement requires explicit exchange/broker registration.
  The LiveExecutionAdapter is entirely disabled by default and raises
  RegulatoryBlockError if LIVE_TRADING_ENABLED=false.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from config import get_settings
from data.kafka.consumer import KafkaProducerManager

settings = get_settings()
logger = structlog.get_logger(__name__)


class RegulatoryBlockError(RuntimeError):
    """Raised when live trading is attempted without regulatory clearance."""


# ─── Order ────────────────────────────────────────────────────────────────────
@dataclass
class OrderRequest:
    ticker: str
    side: str          # BUY | SELL
    order_type: str    # MARKET | LIMIT
    quantity: int
    limit_price: Optional[float] = None
    strategy_id: str = ""
    portfolio_id: str = ""
    reason: str = ""


@dataclass
class OrderResponse:
    order_id: str
    ticker: str
    side: str
    status: str        # PENDING | FILLED | REJECTED
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    timestamp: str = ""
    is_paper: bool = True
    error_message: str = ""


# ─── Position Sizer ───────────────────────────────────────────────────────────
class PositionSizer:
    """
    Compute order quantity from signal strength and portfolio state.

    Supports:
      - Fixed fractional (default): risk fixed % of capital per trade
      - Volatility targeting: scale position so daily P&L vol = target
      - Kelly: theoretically optimal but aggressive; uses half-Kelly in practice
    """

    def __init__(
        self,
        method: str = "fixed_fractional",
        risk_per_trade_pct: float = 0.02,   # 2% of capital per trade
        vol_target_pct: float = 0.15,        # 15% annualised vol target
        max_position_pct: float = 0.10,
    ):
        self.method = method
        self.risk_per_trade_pct = risk_per_trade_pct
        self.vol_target_pct = vol_target_pct
        self.max_position_pct = max_position_pct

    def compute_quantity(
        self,
        ticker: str,
        score: float,           # meta-model score 0–100
        price: float,
        portfolio_value: float,
        realised_vol: float,    # annualised daily vol (0–1)
        stop_loss_pct: float = 0.07,
    ) -> int:
        """Return integer quantity of shares to trade."""
        if portfolio_value <= 0 or price <= 0:
            return 0

        if self.method == "fixed_fractional":
            # Risk = capital * risk_pct; position = risk / stop_loss
            risk_capital = portfolio_value * self.risk_per_trade_pct
            qty = int(risk_capital / (price * stop_loss_pct))

        elif self.method == "volatility_targeting":
            # Target position size so position vol = vol_target_pct of portfolio
            daily_vol_target = self.vol_target_pct / (252 ** 0.5)
            daily_vol = max(realised_vol / (252 ** 0.5), 1e-6)
            position_value = portfolio_value * daily_vol_target / daily_vol
            qty = int(position_value / price)

        elif self.method == "kelly":
            # Half-Kelly: f* = (edge / odds) / 2
            # Use score as proxy for win probability
            p_win = score / 100.0
            p_loss = 1 - p_win
            # Assume avg win = stop_loss, avg loss = stop_loss (simplified)
            if p_loss <= 0:
                return 0
            kelly_f = (p_win / p_loss - 1) / 2  # half-Kelly
            kelly_f = max(0, min(kelly_f, self.max_position_pct))
            qty = int((portfolio_value * kelly_f) / price)

        else:
            qty = int((portfolio_value * self.risk_per_trade_pct) / price)

        # Cap at max position pct
        max_qty = int((portfolio_value * self.max_position_pct) / price)
        return max(0, min(qty, max_qty))


# ─── Execution Adapters ───────────────────────────────────────────────────────
class PaperExecutionAdapter:
    """
    Paper trading executor — simulates fills at next-open price
    with configurable slippage and commission.
    No external API calls.
    """

    def __init__(
        self,
        slippage_pct: float = 0.001,
        commission_pct: float = 0.0015,
    ):
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

    async def submit_order(
        self,
        order: OrderRequest,
        current_price: float,
    ) -> OrderResponse:
        """Simulate immediate fill at current price with slippage."""
        slippage_factor = (1 + self.slippage_pct) if order.side == "BUY" else (1 - self.slippage_pct)
        fill_price = current_price * slippage_factor
        commission = fill_price * order.quantity * self.commission_pct

        order_id = str(uuid4())
        logger.info(
            "Paper order filled",
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            qty=order.quantity,
            fill_price=fill_price,
        )

        return OrderResponse(
            order_id=order_id,
            ticker=order.ticker,
            side=order.side,
            status="FILLED",
            filled_qty=order.quantity,
            avg_fill_price=round(fill_price, 2),
            commission=round(commission, 2),
            timestamp=datetime.utcnow().isoformat(),
            is_paper=True,
        )

    async def cancel_order(self, order_id: str) -> bool:
        logger.info("Paper order cancelled", order_id=order_id)
        return True


class LiveExecutionAdapter:
    """
    Live broker execution adapter.

    ⚠️  REGULATORY NOTICE:
    This adapter is PERMANENTLY DISABLED until:
      1. Legal review confirms automated trading is permitted
      2. Exchange/broker registration is complete
      3. LIVE_TRADING_ENABLED=true is explicitly set by a superuser

    Any attempt to instantiate this without clearance raises
    RegulatoryBlockError immediately.
    """

    def __init__(self, broker_api_key: str, broker_api_url: str):
        if not settings.live_trading_enabled:
            raise RegulatoryBlockError(
                "Live trading is disabled. "
                "Complete Phase 0 regulatory review and set LIVE_TRADING_ENABLED=true "
                "only after receiving exchange/broker authorisation.\n"
                "Ref: Vietnam Securities Law + SSC regulations on program trading."
            )
        import httpx
        self._client = httpx.AsyncClient(
            base_url=broker_api_url,
            headers={"Authorization": f"Bearer {broker_api_key}"},
            timeout=10.0,
        )
        logger.warning("LIVE TRADING ENABLED — ensure regulatory clearance is confirmed")

    async def submit_order(self, order: OrderRequest, current_price: float) -> OrderResponse:
        """Submit a live order to the broker API."""
        payload = {
            "symbol": order.ticker,
            "side": order.side,
            "type": order.order_type,
            "quantity": order.quantity,
            "price": order.limit_price,
        }
        try:
            resp = await self._client.post("/orders", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return OrderResponse(
                order_id=data["orderId"],
                ticker=order.ticker,
                side=order.side,
                status=data.get("status", "PENDING"),
                filled_qty=data.get("filledQty", 0),
                avg_fill_price=data.get("avgPrice", 0.0),
                commission=data.get("commission", 0.0),
                timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
                is_paper=False,
            )
        except Exception as e:
            logger.error("Live order submission failed", error=str(e), ticker=order.ticker)
            return OrderResponse(
                order_id="",
                ticker=order.ticker,
                side=order.side,
                status="REJECTED",
                error_message=str(e),
                is_paper=False,
            )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            resp = await self._client.delete(f"/orders/{order_id}")
            return resp.status_code == 200
        except Exception:
            return False


# ─── Strategy Orchestrator ────────────────────────────────────────────────────
class StrategyOrchestrator:
    """
    Main strategy runner.

    Lifecycle per rebalance cycle:
      1. Fetch latest model scores from DB / cache
      2. Apply universe filters (liquidity, sector limits)
      3. Compute target positions via PositionSizer
      4. Diff target vs current positions to generate orders
      5. Submit orders via execution adapter
      6. Publish order events to Kafka for audit trail
    """

    def __init__(
        self,
        strategy_id: str,
        portfolio_id: str,
        execution_mode: str = "paper",  # "paper" | "live"
        sizer: Optional[PositionSizer] = None,
    ):
        self.strategy_id = strategy_id
        self.portfolio_id = portfolio_id
        self.sizer = sizer or PositionSizer()
        self._kafka: Optional[KafkaProducerManager] = None

        if execution_mode == "live":
            self._adapter = LiveExecutionAdapter(
                broker_api_key=settings.broker_api_key or "",
                broker_api_url=settings.broker_api_url or "",
            )
        else:
            self._adapter = PaperExecutionAdapter()

    async def _get_kafka(self) -> KafkaProducerManager:
        if self._kafka is None:
            self._kafka = KafkaProducerManager()
            await self._kafka.start()
        return self._kafka

    async def rebalance(
        self,
        scored_stocks: List[Dict],   # [{"ticker", "score", "price", "realised_vol"}, ...]
        current_positions: Dict[str, int],  # {ticker: qty}
        portfolio_value: float,
        top_n: int = 20,
        min_score: float = 60.0,
    ) -> List[OrderResponse]:
        """
        Full rebalance cycle: score → size → diff → execute.
        """
        logger.info("Starting rebalance", strategy=self.strategy_id,
                    n_candidates=len(scored_stocks))

        # 1. Filter and select top-N stocks
        candidates = [s for s in scored_stocks if s["score"] >= min_score]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        target_tickers = {s["ticker"] for s in candidates[:top_n]}

        # 2. Compute target quantities
        target_positions: Dict[str, int] = {}
        for stock in candidates[:top_n]:
            qty = self.sizer.compute_quantity(
                ticker=stock["ticker"],
                score=stock["score"],
                price=stock["price"],
                portfolio_value=portfolio_value,
                realised_vol=stock.get("realised_vol", 0.20),
            )
            if qty > 0:
                target_positions[stock["ticker"]] = qty

        # 3. Diff vs current positions
        orders_to_submit: List[OrderRequest] = []

        # Close positions not in target universe
        for ticker, qty in current_positions.items():
            if ticker not in target_tickers and qty > 0:
                orders_to_submit.append(OrderRequest(
                    ticker=ticker,
                    side="SELL",
                    order_type="MARKET",
                    quantity=qty,
                    strategy_id=self.strategy_id,
                    portfolio_id=self.portfolio_id,
                    reason="exit_rebalance",
                ))

        # Open/adjust positions
        for ticker, target_qty in target_positions.items():
            current_qty = current_positions.get(ticker, 0)
            delta = target_qty - current_qty
            if abs(delta) >= 100:  # minimum lot size (adjust per exchange rules)
                orders_to_submit.append(OrderRequest(
                    ticker=ticker,
                    side="BUY" if delta > 0 else "SELL",
                    order_type="MARKET",
                    quantity=abs(delta),
                    strategy_id=self.strategy_id,
                    portfolio_id=self.portfolio_id,
                    reason="rebalance",
                ))

        # 4. Execute orders
        responses: List[OrderResponse] = []
        kafka = await self._get_kafka()

        for order in orders_to_submit:
            # Get current price for this ticker from scored_stocks lookup
            price_lookup = {s["ticker"]: s["price"] for s in scored_stocks}
            price = price_lookup.get(order.ticker, 0)
            if price <= 0:
                logger.warning("No price for ticker, skipping", ticker=order.ticker)
                continue

            response = await self._adapter.submit_order(order, current_price=price)
            responses.append(response)

            # Publish order event to Kafka for audit / downstream consumers
            await kafka.publish_order_event(
                order_id=response.order_id,
                event={
                    "type": "order_submitted",
                    "strategy_id": self.strategy_id,
                    "portfolio_id": self.portfolio_id,
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": response.status,
                    "fill_price": response.avg_fill_price,
                    "is_paper": response.is_paper,
                    "ts": datetime.utcnow().isoformat(),
                },
            )

        logger.info(
            "Rebalance complete",
            strategy=self.strategy_id,
            orders=len(responses),
            fills=sum(1 for r in responses if r.status == "FILLED"),
        )
        return responses
