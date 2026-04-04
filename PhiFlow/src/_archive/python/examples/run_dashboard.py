"""
Run Quantum Dashboard
Demonstrating real-time quantum flow visualization
"""
from quantum_dashboard import dashboard
from quantum_flow import flow, Dimension

def main():
    # Initialize at Heart Field frequency
    print(f"Starting quantum dashboard at {flow.frequency} Hz")
    
    # Add initial dimensions
    flow.add_dimension(Dimension.PHYSICAL)
    flow.add_dimension(Dimension.ETHERIC)
    flow.add_dimension(Dimension.EMOTIONAL)
    
    print(f"Initial quantum coherence: {flow.coherence}")
    print("\nStarting dashboard server...")
    print("Access the dashboard at http://localhost:8050")
    
    # Start the dashboard
    dashboard.start()

if __name__ == "__main__":
    main()
