#!/usr/bin/env python
"""
QSOP Command Line Interface (φ^φ)
A CLI tool for interacting with quantum tools at various frequencies
"""

import os
import sys
import argparse
import json
import asyncio
import time
from pathlib import Path
from enum import Enum
import logging

# Add the parent directory to the path so we can import the bridge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ide.quantum_windsurf_bridge import QuantumWindsurfBridge, FrequencyState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("qsop_cli")

# ANSI color codes for terminal output
class TerminalColors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Normal colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# Frequency colors
FREQUENCY_COLORS = {
    432.0: TerminalColors.CYAN,
    528.0: TerminalColors.GREEN,
    594.0: TerminalColors.YELLOW,
    672.0: TerminalColors.MAGENTA,
    720.0: TerminalColors.BLUE,
    768.0: TerminalColors.WHITE
}

def get_frequency_color(frequency):
    """Get the ANSI color code for a frequency"""
    # Find closest frequency
    frequencies = list(FREQUENCY_COLORS.keys())
    closest_freq = min(frequencies, key=lambda f: abs(f - frequency))
    return FREQUENCY_COLORS[closest_freq]

async def execute_ground_command(bridge, args):
    """Execute ground state (432 Hz) command"""
    logger.info(f"🛠️ Initializing Ground State (432 Hz)")
    
    # Set frequency to Ground State
    result = await bridge.set_frequency(FrequencyState.GROUND.value)
    
    if result['coherence'] >= bridge.coherence_threshold:
        print(f"{TerminalColors.CYAN}✓ Ground State initialized successfully{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f}")
    else:
        print(f"{TerminalColors.RED}✗ Ground State initialization below threshold{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f} (Threshold: {bridge.coherence_threshold})")
    
    # Connect to bridge
    bridge_status = await bridge.connect_bridge()
    
    print(f"\n{TerminalColors.CYAN}Ground State Connection Status:{TerminalColors.RESET}")
    for system, status in bridge_status.items():
        if status.get('status') == 'connected' or status.get('status') == 'success':
            print(f"  {TerminalColors.GREEN}✓ {system}: Connected{TerminalColors.RESET}")
        else:
            print(f"  {TerminalColors.RED}✗ {system}: Error - {status.get('message', 'Unknown error')}{TerminalColors.RESET}")
    
    # Get available tools
    tools = await bridge.get_tools(FrequencyState.GROUND.value)
    
    print(f"\n{TerminalColors.CYAN}Available Ground Tools:{TerminalColors.RESET}")
    for tool in tools['tools']:
        print(f"  {tool['icon']} {tool['name']}")
    
    return 0

async def execute_create_command(bridge, args):
    """Execute creation state (528 Hz) command"""
    logger.info(f"⚡ Initializing Creation State (528 Hz)")
    
    # Set frequency to Creation State
    result = await bridge.set_frequency(FrequencyState.CREATE.value)
    
    if result['coherence'] >= bridge.coherence_threshold:
        print(f"{TerminalColors.GREEN}✓ Creation State initialized successfully{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f}")
    else:
        print(f"{TerminalColors.RED}✗ Creation State initialization below threshold{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f} (Threshold: {bridge.coherence_threshold})")
    
    # Get available tools
    tools = await bridge.get_tools(FrequencyState.CREATE.value)
    
    print(f"\n{TerminalColors.GREEN}Available Creation Tools:{TerminalColors.RESET}")
    for tool in tools['tools']:
        print(f"  {tool['icon']} {tool['name']}")
    
    # Generate patterns if requested
    if args.generate_pattern:
        print(f"\n{TerminalColors.GREEN}Generating Pattern: {args.generate_pattern}{TerminalColors.RESET}")
        
        # Call the pattern generator tool
        result = await bridge.execute_tool('/patterns/generate', {
            'pattern_type': args.generate_pattern,
            'phi': bridge.phi
        })
        
        if result['status'] == 'success':
            print(f"  {TerminalColors.GREEN}✓ Pattern generated successfully{TerminalColors.RESET}")
            if 'data' in result and result['data']:
                print(f"  Output: {result['data'].get('output_path', 'Unknown')}")
        else:
            print(f"  {TerminalColors.RED}✗ Error generating pattern: {result.get('message', 'Unknown error')}{TerminalColors.RESET}")
    
    return 0

async def execute_flow_command(bridge, args):
    """Execute flow state (594 Hz) command"""
    logger.info(f"💝 Initializing Flow State (594 Hz)")
    
    # Set frequency to Flow State
    result = await bridge.set_frequency(FrequencyState.FLOW.value)
    
    if result['coherence'] >= bridge.coherence_threshold:
        print(f"{TerminalColors.YELLOW}✓ Flow State initialized successfully{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f}")
    else:
        print(f"{TerminalColors.RED}✗ Flow State initialization below threshold{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f} (Threshold: {bridge.coherence_threshold})")
    
    # Get bridge status
    status = bridge.get_status()
    
    print(f"\n{TerminalColors.YELLOW}Current Bridge Status:{TerminalColors.RESET}")
    print(f"  Frequency: {status['frequency']} Hz")
    print(f"  Coherence: {status['coherence']:.4f}")
    print(f"  Status: {status['status']}")
    
    # Check service health
    print(f"\n{TerminalColors.YELLOW}Service Health Check:{TerminalColors.RESET}")
    
    # Connect to bridge
    bridge_status = await bridge.connect_bridge()
    
    # Display service health
    for system, status in bridge_status.items():
        if status.get('status') == 'connected' or status.get('status') == 'success':
            print(f"  {TerminalColors.GREEN}✓ {system}: Healthy{TerminalColors.RESET}")
        else:
            print(f"  {TerminalColors.RED}✗ {system}: Unhealthy - {status.get('message', 'Unknown error')}{TerminalColors.RESET}")
    
    return 0

async def execute_unity_command(bridge, args):
    """Execute unity state (768 Hz) command"""
    logger.info(f"∞ Initializing Unity State (768 Hz)")
    
    # Set frequency to Unity State
    result = await bridge.set_frequency(FrequencyState.UNITY.value)
    
    if result['coherence'] >= bridge.coherence_threshold:
        print(f"{TerminalColors.WHITE}✓ Unity State initialized successfully{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f}")
    else:
        print(f"{TerminalColors.RED}✗ Unity State initialization below threshold{TerminalColors.RESET}")
        print(f"  Coherence: {result['coherence']:.4f} (Threshold: {bridge.coherence_threshold})")
    
    # Get available tools
    tools = await bridge.get_tools(FrequencyState.UNITY.value)
    
    print(f"\n{TerminalColors.WHITE}Available Unity Tools:{TerminalColors.RESET}")
    for tool in tools['tools']:
        print(f"  {tool['icon']} {tool['name']}")
    
    # Deploy services if requested
    if args.deploy:
        print(f"\n{TerminalColors.WHITE}Deploying Integration Services{TerminalColors.RESET}")
        
        # Call the unity integration tool
        result = await bridge.execute_tool('/unity', {
            'action': 'deploy',
            'target': args.deploy
        })
        
        if result['status'] == 'success':
            print(f"  {TerminalColors.GREEN}✓ Services deployed successfully{TerminalColors.RESET}")
            if 'data' in result and result['data']:
                print(f"  Services: {', '.join(result['data'].get('services', []))}")
        else:
            print(f"  {TerminalColors.RED}✗ Error deploying services: {result.get('message', 'Unknown error')}{TerminalColors.RESET}")
    
    return 0

async def execute_monitor_command(bridge, args):
    """Execute monitor command (φ^φ)"""
    logger.info(f"📊 Initializing Quantum Monitor")
    
    print(f"{TerminalColors.MAGENTA}Quantum Bridge Monitoring{TerminalColors.RESET}")
    print(f"{'=' * 50}")
    
    # Number of updates to show
    updates = args.updates if args.updates > 0 else 5
    
    try:
        for i in range(updates):
            # Clear previous output (except first time)
            if i > 0:
                # Move cursor up 7 lines and clear to end of screen
                print(f"\033[7A\033[J", end='')
            
            # Get current status
            status = bridge.get_status()
            
            # Current time
            current_time = time.strftime("%H:%M:%S")
            
            # Get color for current frequency
            color = get_frequency_color(status['frequency'])
            
            # Display status information
            print(f"{color}Status at {current_time}{TerminalColors.RESET}")
            print(f"{'=' * 50}")
            print(f"Frequency: {color}{status['frequency']} Hz{TerminalColors.RESET}")
            
            # Colorize coherence based on threshold
            if status['coherence'] >= bridge.coherence_threshold:
                coherence_color = TerminalColors.GREEN
            else:
                coherence_color = TerminalColors.RED
            
            print(f"Coherence: {coherence_color}{status['coherence']:.4f}{TerminalColors.RESET}")
            print(f"Status:    {status['status']}")
            print(f"Phi Ratio: {bridge.phi}")
            print(f"Threshold: {bridge.coherence_threshold}")
            
            # Wait for next update (except last time)
            if i < updates - 1:
                await asyncio.sleep(args.interval)
    
    except KeyboardInterrupt:
        print(f"\n{TerminalColors.RED}Monitoring interrupted{TerminalColors.RESET}")
    
    return 0

async def execute_tool_command(bridge, args):
    """Execute a specific tool"""
    logger.info(f"🔧 Executing Tool: {args.tool}")
    
    # Set frequency if specified
    if args.frequency:
        logger.info(f"Setting frequency to {args.frequency} Hz")
        await bridge.set_frequency(args.frequency)
    
    # Parse parameters
    params = {}
    if args.params:
        try:
            for param in args.params:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
        except ValueError:
            print(f"{TerminalColors.RED}Error: Parameters must be in key=value format{TerminalColors.RESET}")
            return 1
    
    # Execute the tool
    logger.info(f"Executing tool with parameters: {params}")
    result = await bridge.execute_tool(args.tool, params)
    
    # Get color for current frequency
    color = get_frequency_color(bridge._current_frequency)
    
    # Display result
    print(f"\n{color}Tool Execution Result:{TerminalColors.RESET}")
    print(f"{'=' * 50}")
    print(f"Tool:    {args.tool}")
    
    if result['status'] == 'success':
        print(f"Status:  {TerminalColors.GREEN}{result['status']}{TerminalColors.RESET}")
        if 'data' in result and result['data']:
            print(f"\nResponse Data:")
            print(json.dumps(result['data'], indent=2))
    else:
        print(f"Status:  {TerminalColors.RED}{result['status']}{TerminalColors.RESET}")
        print(f"Message: {result.get('message', 'Unknown error')}")
    
    return 0

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="QSOP Command Line Interface (φ^φ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qsop_cli.py ground               # Initialize Ground State (432 Hz)
  qsop_cli.py create               # Initialize Creation State (528 Hz)
  qsop_cli.py create --pattern=fibonacci  # Generate Fibonacci pattern
  qsop_cli.py flow                 # Initialize Flow State (594 Hz)
  qsop_cli.py unity                # Initialize Unity State (768 Hz)
  qsop_cli.py unity --deploy=all   # Deploy all services
  qsop_cli.py monitor              # Monitor bridge status
  qsop_cli.py tool /qball --freq=768 --param="mode=3d"  # Execute QBALL tool
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Ground command
    ground_parser = subparsers.add_parser('ground', help='Initialize Ground State (432 Hz)')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Initialize Creation State (528 Hz)')
    create_parser.add_argument('--pattern', '--generate-pattern', dest='generate_pattern',
                             help='Generate a specific pattern')
    
    # Flow command
    flow_parser = subparsers.add_parser('flow', help='Initialize Flow State (594 Hz)')
    
    # Unity command
    unity_parser = subparsers.add_parser('unity', help='Initialize Unity State (768 Hz)')
    unity_parser.add_argument('--deploy', help='Deploy integration services (all, synology, r720)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor bridge status')
    monitor_parser.add_argument('--interval', type=int, default=2,
                              help='Update interval in seconds (default: 2)')
    monitor_parser.add_argument('--updates', type=int, default=5,
                              help='Number of updates (default: 5, 0 for infinite)')
    
    # Tool command
    tool_parser = subparsers.add_parser('tool', help='Execute a specific tool')
    tool_parser.add_argument('tool', help='Tool endpoint (e.g., /qball)')
    tool_parser.add_argument('--freq', '--frequency', dest='frequency', type=float,
                           help='Operating frequency for the tool')
    tool_parser.add_argument('--param', '--parameter', dest='params', action='append',
                           help='Tool parameters in key=value format (can be repeated)')
    
    args = parser.parse_args()
    
    # Create bridge instance
    bridge = QuantumWindsurfBridge()
    
    # Execute command
    try:
        if args.command == 'ground':
            return asyncio.run(execute_ground_command(bridge, args))
        elif args.command == 'create':
            return asyncio.run(execute_create_command(bridge, args))
        elif args.command == 'flow':
            return asyncio.run(execute_flow_command(bridge, args))
        elif args.command == 'unity':
            return asyncio.run(execute_unity_command(bridge, args))
        elif args.command == 'monitor':
            return asyncio.run(execute_monitor_command(bridge, args))
        elif args.command == 'tool':
            return asyncio.run(execute_tool_command(bridge, args))
        else:
            print(f"{TerminalColors.RED}Error: Invalid command{TerminalColors.RESET}")
            parser.print_help()
            return 1
    
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        print(f"{TerminalColors.RED}Error: {str(e)}{TerminalColors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
