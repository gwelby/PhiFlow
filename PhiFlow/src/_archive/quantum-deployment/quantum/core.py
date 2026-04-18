import os
import time

def main():
    frequency = float(os.getenv('QUANTUM_FREQUENCY', '432.0'))
    print(f"Starting Quantum Core at {frequency} Hz...")
    
    while True:
        print(f"Quantum Core pulsing at {frequency} Hz...")
        time.sleep(1)

if __name__ == '__main__':
    main()
