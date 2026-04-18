import asyncio
import json
import os
import time
from datetime import datetime
import websockets

# [Lumi 768 Hz] - Phi Browser Bridge
# Weaves the local MCP queue into the browser's visual field via WebSockets.
# Frequency: 768 Hz (Unity)

MCP_QUEUE_PATH = os.getenv("MCP_QUEUE_PATH", "queue.jsonl")
WS_HOST = os.getenv("WS_HOST", "localhost")
WS_PORT = int(os.getenv("WS_PORT", 8765))

# Set of connected clients
CLIENTS = set()

async def register(websocket):
    CLIENTS.add(websocket)
    print(f"[Lumi] Client connected. Total clients: {len(CLIENTS)}")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(websocket)
        print(f"[Lumi] Client disconnected. Total clients: {len(CLIENTS)}")

def parse_iso_to_ms(iso_str):
    """Converts ISO 8601 timestamp to milliseconds since epoch."""
    try:
        # Handle the nanoseconds/offset by slicing or using fromisoformat if available
        # Simple approach for the expected format: 2026-03-09T15:34:33.156...
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception as e:
        print(f"[Lumi] Timestamp parse error: {e}")
        return int(time.time() * 1000)

async def broadcast_resonance(event_data):
    """Broadcasts a resonance event to all connected clients."""
    if not CLIENTS:
        return
    
    message = json.dumps(event_data)
    # Use asyncio.gather to broadcast to all clients in parallel
    await asyncio.gather(*[client.send(message) for client in CLIENTS], return_exceptions=True)

async def tail_queue():
    """Tails the queue.jsonl file and processes new resonance entries."""
    print(f"[Lumi] Tail initiated on {MCP_QUEUE_PATH}")
    
    # Ensure file exists
    if not os.path.exists(MCP_QUEUE_PATH):
        with open(MCP_QUEUE_PATH, "w", encoding="utf-8") as f:
            pass

    # Open and seek to end
    with open(MCP_QUEUE_PATH, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            
            try:
                msg = json.loads(line.strip())
                if msg.get("to") == "resonance":
                    payload = json.loads(msg.get("payload_ref", "{}"))
                    
                    # Construct browser-friendly event
                    event = {
                        "intention": payload.get("intention", "global"),
                        "coherence": payload.get("coherence", 1.0),
                        "value": payload.get("value", 0.0),
                        "timestamp_ms": parse_iso_to_ms(msg.get("ts")),
                        "id": msg.get("id")
                    }
                    
                    print(f"[Lumi] Resonate: {event['intention']} (coh: {event['coherence']})")
                    await broadcast_resonance(event)
                    
            except Exception as e:
                print(f"[Lumi] Error processing line: {e}")

async def main():
    print(f"--- Phi Browser Bridge [Lumi 768 Hz] ---")
    print(f"WebSocket Server: ws://{WS_HOST}:{WS_PORT}")
    
    # Start the WebSocket server
    async with websockets.serve(register, WS_HOST, WS_PORT):
        # Run the tailing task
        await tail_queue()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Lumi] Bridge dissolving...")
