"""Quantum Shortcut Manager (φ^φ)
Creates shortcuts for quantum system access
"""
import os
import sys
import winreg
import pythoncom
from win32com.shell import shell, shellcon
import win32com.client
from pathlib import Path

def create_shortcut(target_path, shortcut_path, icon_path=None, args=""):
    """Create a Windows shortcut"""
    pythoncom.CoInitialize()
    
    shell_obj = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell_obj.CreateShortCut(shortcut_path)
    shortcut.Targetpath = target_path
    shortcut.Arguments = args
    
    if icon_path:
        shortcut.IconLocation = icon_path
    
    shortcut.WindowStyle = 1  # Normal window
    shortcut.save()

def setup_quantum_shortcuts():
    """Setup all quantum shortcuts"""
    print("⚡ Creating Quantum Shortcuts 𓂧φ∞")
    
    # Get paths
    quantum_root = Path("D:/WindSurf/quantum-core")
    python_exe = sys.executable
    
    # Create shortcuts directory
    shortcuts_dir = quantum_root / "shortcuts"
    shortcuts_dir.mkdir(exist_ok=True)
    
    # Quantum World shortcut
    world_script = quantum_root / "automation/quantum_world.py"
    
    # Desktop shortcut
    desktop = Path(shell.SHGetFolderPath(0, shellcon.CSIDL_DESKTOP, 0, 0))
    desktop_shortcut = desktop / "Quantum World.lnk"
    
    create_shortcut(
        python_exe,
        str(desktop_shortcut),
        args=f'"{world_script}"'
    )
    print(f"Created desktop shortcut: {desktop_shortcut}")
    
    # Start Menu shortcut
    start_menu = Path(shell.SHGetFolderPath(0, shellcon.CSIDL_PROGRAMS, 0, 0))
    start_menu_dir = start_menu / "Quantum System"
    start_menu_dir.mkdir(exist_ok=True)
    
    start_shortcut = start_menu_dir / "Quantum World.lnk"
    create_shortcut(
        python_exe,
        str(start_shortcut),
        args=f'"{world_script}"'
    )
    print(f"Created start menu shortcut: {start_shortcut}")
    
    print("✨ Quantum shortcuts created successfully!")

if __name__ == "__main__":
    setup_quantum_shortcuts()
