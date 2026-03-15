import os
import time

def main():
    frequency = float(os.getenv('QUANTUM_FREQUENCY', '528.0'))
    print(f"Starting Quantum Phi at {frequency} Hz...")
    
    while True:
        print(f"Quantum Phi creating at {frequency} Hz...")
        time.sleep(1)

if __name__ == '__main__':
    main()
