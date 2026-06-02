"""
FastAPI Dashboard Server
Serves the live RuView-style dashboard and WebSocket data stream.
"""

import asyncio
import json
import logging
import time
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

log = logging.getLogger(__name__)

app = FastAPI(title="Spatial Intelligence System")

# Global reference to the main system (set by main.py)
_system = None
_connected_clients: Set[WebSocket] = set()


def set_system(system):
    global _system
    _system = system


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connected_clients.add(ws)
    log.info(f"Dashboard client connected — {len(_connected_clients)} total")
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.05)
                await _handle_command(ws, msg)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        _connected_clients.discard(ws)
        log.info(f"Client disconnected — {len(_connected_clients)} remaining")
    except Exception as e:
        log.warning(f"WebSocket error: {e}")
        _connected_clients.discard(ws)


async def _handle_command(ws: WebSocket, msg: str):
    """Handle commands from the dashboard UI."""
    try:
        cmd = json.loads(msg)
        action = cmd.get("action")

        if action == "calibrate" and _system:
            zone = cmd.get("zone", "default")
            _system.fingerprinter.start_calibration(zone)
            await ws.send_text(json.dumps({"type": "ack", "msg": f"Calibration started for zone '{zone}'"}))

        elif action == "set_mode" and _system:
            mode = cmd.get("mode", "active")
            _system.set_mode(mode)
            await ws.send_text(json.dumps({"type": "ack", "msg": f"Mode set to {mode}"}))

        elif action == "ping":
            await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

    except Exception as e:
        log.debug(f"Command parse error: {e}")


async def broadcast(payload: dict):
    """Broadcast a state update to all connected dashboard clients."""
    if not _connected_clients:
        return
    msg = json.dumps(payload)
    dead = set()
    for ws in _connected_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _connected_clients -= dead


@app.get("/api/state")
async def get_state():
    if _system and _system.latest_state:
        return _system.fusion.serialize_state(_system.latest_state)
    return {"status": "no_data"}


@app.get("/api/history")
async def get_history():
    if _system:
        return {"events": _system.fingerprinter.get_anomaly_history(100)}
    return {"events": []}


@app.get("/api/status")
async def get_status():
    return {
        "uptime_s": time.time() - (_system.start_time if _system else time.time()),
        "mode": _system.mode if _system else "unknown",
        "radar_connected": _system.radar_connected if _system else False,
        "pir_connected": _system.pir_connected if _system else False,
        "calibrating": _system.fingerprinter.is_calibrating if _system else False,
        "cal_progress": _system.fingerprinter.calibration_progress if _system else 0.0,
        "clients_connected": len(_connected_clients)
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port, log_level="warning")
