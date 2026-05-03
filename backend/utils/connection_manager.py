"""
utils/connection_manager.py — WebSocket connection manager.
Groups connections by topic (e.g., ticker symbol) for targeted broadcasts.
"""
import asyncio
from collections import defaultdict
from typing import Dict, List

import structlog
from fastapi import WebSocket

from utils.metrics import WS_ACTIVE_CONNECTIONS

logger = structlog.get_logger(__name__)


class WebSocketConnectionManager:
    """
    Manages WebSocket connections grouped by topic.

    Usage:
      manager = WebSocketConnectionManager()

      # On connect:
      await manager.connect(websocket, group="VNM")

      # Broadcast to all subscribers of a group:
      await manager.broadcast(group="VNM", message=json_str)

      # On disconnect:
      manager.disconnect(websocket, group="VNM")
    """

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, group: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[group].append(websocket)
        WS_ACTIVE_CONNECTIONS.labels(group=group).set(
            len(self._connections[group])
        )
        logger.debug("WS connected", group=group, total=len(self._connections[group]))

    def disconnect(self, websocket: WebSocket, group: str) -> None:
        connections = self._connections.get(group, [])
        if websocket in connections:
            connections.remove(websocket)
        WS_ACTIVE_CONNECTIONS.labels(group=group).set(len(connections))
        logger.debug("WS disconnected", group=group, remaining=len(connections))

    async def broadcast(self, group: str, message: str) -> None:
        """Send message to all active connections in a group."""
        connections = list(self._connections.get(group, []))
        if not connections:
            return

        dead: List[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws, group)

    async def broadcast_all(self, message: str) -> None:
        """Broadcast to every connected client across all groups."""
        for group in list(self._connections.keys()):
            await self.broadcast(group, message)

    def connection_count(self, group: str = "") -> int:
        if group:
            return len(self._connections.get(group, []))
        return sum(len(v) for v in self._connections.values())
