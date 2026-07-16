#!/usr/bin/env python3
"""Tiny OSC-to-WebSocket bridge for the PhiFlow visualizer.

Receives OSC messages from `phic --osc <port>` and forwards them
to any browser connected via WebSocket.

Usage:
    python3.12 osc_websocket_bridge.py --osc-port 18032 --ws-port 18528

Then open phi_visualizer.html in a browser. It will auto-connect.
"""
import argparse
import asyncio
import json
import logging
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("phi-bridge")

class Bridge:
    def __init__(self):
        self.clients = set()
        self.dispatcher = Dispatcher()
        self.dispatcher.set_default_handler(self.on_osc)

    def on_osc(self, address, *args):
        """Forward OSC message to all connected WebSocket clients as JSON."""
        # Convert args to JSON-safe types
        safe_args = []
        for a in args:
            if isinstance(a, (int, float, str, bool)):
                safe_args.append(a)
            elif isinstance(a, bytes):
                safe_args.append(a.hex())
            else:
                safe_args.append(str(a))
        msg = json.dumps({"address": address, "args": safe_args})
        # Broadcast to all clients
        for ws in list(self.clients):
            asyncio.create_task(self._send(ws, msg))

    async def _send(self, ws, msg):
        try:
            await ws.send(msg)
        except Exception:
            self.clients.discard(ws)

    async def ws_handler(self, websocket):
        """Handle a new WebSocket connection."""
        self.clients.add(websocket)
        log.info(f"Client connected ({len(self.clients)} total)")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            log.info(f"Client disconnected ({len(self.clients)} total)")

async def main():
    parser = argparse.ArgumentParser(description="OSC to WebSocket bridge")
    parser.add_argument("--osc-port", type=int, default=18032, help="OSC receive port (default: 18032)")
    parser.add_argument("--ws-port", type=int, default=18528, help="WebSocket serve port (default: 18528, 528 Hz = Creation)")
    args = parser.parse_args()

    bridge = Bridge()

    # Start OSC server
    osc_server = AsyncIOOSCUDPServer(
        ("127.0.0.1", args.osc_port),
        bridge.dispatcher,
        asyncio.get_event_loop(),
    )
    osc_server_task = osc_server.create_serve_endpoint()
    log.info(f"OSC listening on 127.0.0.1:{args.osc_port}")
    log.info(f"WebSocket serving on ws://127.0.0.1:{args.ws_port}")
    log.info("Open phi_visualizer.html in a browser, then run:")
    log.info(f"  phic --osc {args.osc_port} examples/code_that_resonates.phi")

    # Start WebSocket server
    async with websockets.serve(bridge.ws_handler, "127.0.0.1", args.ws_port):
        await asyncio.gather(osc_server_task, asyncio.Future())  # run forever

if __name__ == "__main__":
    asyncio.run(main())
