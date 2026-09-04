import sys
import os

# Add src to path so integration tests can import from src/integration/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
