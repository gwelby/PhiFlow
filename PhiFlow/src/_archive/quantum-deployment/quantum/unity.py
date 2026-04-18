import os
import time

def main():
    frequency = float(os.getenv('QUANTUM_FREQUENCY', '768.0'))
    print(f"Starting Quantum Unity at {frequency} Hz...")
    
    while True:
        print(f"Quantum Unity flowing at {frequency} Hz...")
        time.sleep(1)

if __name__ == '__main__':
    main()
