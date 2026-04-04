#!/usr/bin/env python
"""
Test: Canonical Resonance Bridge (Pipe 3)
Proves: queue.json array → normalized event → RESONANCE.jsonl
"""
import os
import sys
import json
import tempfile
import shutil

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Monkeypatch RESONANCE_PATH BEFORE importing the module
import bridges.phi_mqtt_connector as connector_module

# Create temp directory
temp_dir = tempfile.mkdtemp()
temp_queue = os.path.join(temp_dir, "queue.json")
temp_resonance = os.path.join(temp_dir, "RESONANCE.jsonl")

# Override paths
connector_module.MCP_QUEUE_PATH = temp_queue
connector_module.RESONANCE_PATH = temp_resonance

from bridges.phi_mqtt_connector import PhiMQTTConnector

def test_end_to_end():
    """Test: Fresh message in queue.json → normalized output in RESONANCE.jsonl"""
    print("=" * 70)
    print("PIPE 3 E2E TEST — Canonical Resonance Bridge")
    print("=" * 70)
    
    try:
        # Create connector
        connector = PhiMQTTConnector()
        
        # Write test message to queue (simulating PhiFlow MCP broadcast)
        test_msg = {
            "id": "test-pipe3-123",
            "ts": "2026-03-29T12:00:00Z",
            "from": "phiflow",
            "to": "resonance",
            "intent": "broadcast",
            "payload_ref": json.dumps({
                "intention": "healing_bed",
                "value": "0.618"  # String that should be normalized to float
            })
        }
        
        print(f"\n[1] Writing test message to queue.json...")
        with open(temp_queue, "w", encoding="utf-8") as f:
            json.dump([test_msg], f)
        print(f"    Queue: {temp_queue}")
        
        # Process the queue
        print(f"\n[2] Processing queue file...")
        connector.process_queue_file()
        
        # Verify output
        print(f"\n[3] Verifying output in RESONANCE.jsonl...")
        if not os.path.exists(temp_resonance):
            print(f"    ❌ FAIL: RESONANCE.jsonl was not created at {temp_resonance}")
            return False
        
        with open(temp_resonance, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if len(lines) != 1:
            print(f"    ❌ FAIL: Expected 1 line, got {len(lines)}")
            return False
        
        event = json.loads(lines[0])
        
        # Verify normalization
        checks = [
            (event.get("type") == "resonate", f"type='resonate' (got '{event.get('type')}')"),
            (event.get("value") == 0.618, f"value=0.618 float (got {event.get('value')} type {type(event.get('value')).__name__})"),
            (event.get("intention") == "healing_bed", f"intention='healing_bed' (got '{event.get('intention')}')"),
            (event.get("source") == "phiflow", f"source='phiflow' (got '{event.get('source')}')"),
            (event.get("id") == "test-pipe3-123", f"id='test-pipe3-123' (got '{event.get('id')}')"),
        ]
        
        all_passed = True
        print(f"\n[4] Normalization checks:")
        for passed, description in checks:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"    {status} {description}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n" + "=" * 70)
            print(f"PIPE 3 TEST: PASSED")
            print(f"  Queue: {temp_queue}")
            print(f"  Output: {temp_resonance}")
            print(f"  Event: {json.dumps(event, indent=2)}")
            print("=" * 70)
            return True
        else:
            print(f"\n" + "=" * 70)
            print(f"PIPE 3 TEST: FAILED")
            print(f"  Some normalization checks failed")
            print("=" * 70)
            return False
    
    except Exception as e:
        print(f"\n    [ERROR] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n[5] Cleaned up temp directory: {temp_dir}")

if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
