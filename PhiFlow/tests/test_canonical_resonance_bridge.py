import os
import sys
import json
import time
import threading
import tempfile
import shutil
import unittest
from datetime import datetime

# Import the connector - add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
from bridges.phi_mqtt_connector import PhiMQTTConnector

class TestCanonicalResonanceBridge(unittest.TestCase):
    def setUp(self):
        # Create temp files for queue.json and RESONANCE.jsonl
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = os.path.join(self.temp_dir, "queue.json")
        self.resonance_path = os.path.join(self.temp_dir, "RESONANCE.jsonl")
        
        # Override environment variables for the test
        os.environ["MCP_QUEUE_PATH"] = self.queue_path
        
        # We need to monkeypatch the RESONANCE_PATH in the module
        import bridges.phi_mqtt_connector as module
        self.orig_resonance_path = module.RESONANCE_PATH
        module.RESONANCE_PATH = self.resonance_path
        
        # Write initial empty queue
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    def tearDown(self):
        # Clean up temp files
        import bridges.phi_mqtt_connector as module
        module.RESONANCE_PATH = self.orig_resonance_path
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_normalization(self):
        connector = PhiMQTTConnector()
        
        # 1. Simulate a message from PhiFlow
        test_msg = {
            "id": "test-id-123",
            "ts": "2026-03-29T12:00:00Z",
            "from": "phiflow",
            "to": "resonance",
            "intent": "broadcast",
            "payload_ref": json.dumps({
                "intention": "healing_bed",
                "value": "0.618"
            })
        }
        
        # Write to queue
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump([test_msg], f)
            
        # 2. Process the queue
        # Wait slightly so mtime is definitely newer
        time.sleep(0.1)
        
        import sys
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        
        connector.process_queue_file()
        
        sys.stdout = sys.__stdout__
        print("\n--- CONNECTOR OUTPUT ---")
        print(captured_output.getvalue())
        print("------------------------\n")
        
        # 3. Verify the output in RESONANCE.jsonl
        self.assertTrue(os.path.exists(self.resonance_path))
        with open(self.resonance_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        
        # Verify normalization
        self.assertEqual(event["type"], "resonate")
        self.assertEqual(event["value"], 0.618) # String should be parsed to float
        self.assertEqual(event["intention"], "healing_bed")
        self.assertEqual(event["source"], "phiflow")
        self.assertEqual(event["id"], "test-id-123")
        self.assertEqual(event["ts"], "2026-03-29T12:00:00Z")

if __name__ == "__main__":
    unittest.main()
