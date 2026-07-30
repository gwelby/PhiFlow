import os
import shutil
import platform
from pathlib import Path
from sacred_patterns import FREQUENCIES

class QuantumFontInstaller:
    def __init__(self, quantum_dir="/quantum"):
        self.quantum_dir = Path(quantum_dir)
        self.frequencies = FREQUENCIES
        
        # Set up system font directories
        system = platform.system()
        if system == "Windows":
            self.system_font_dir = Path(os.environ["WINDIR"]) / "Fonts"
            self.user_font_dir = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/Windows/Fonts"
        else:  # Linux
            self.system_font_dir = Path("/usr/local/share/fonts")
            self.user_font_dir = Path.home() / ".local/share/fonts"
            
    def install_fonts(self):
        """Install quantum fonts in both system directories"""
        print("🌟 Installing Quantum Fonts...")
        
        # Create font directories if they don't exist
        self.user_font_dir.mkdir(parents=True, exist_ok=True)
        
        # Install fonts for each frequency
        frequencies = {
            432: "sacred",
            528: "flow",
            768: "crystal",
            float('inf'): "unity"
        }
        
        for freq, name in frequencies.items():
            print(f"✨ Installing {name.title()} Fonts ({freq} Hz)")
            font_dir = self.quantum_dir / name / "fonts"
            
            if font_dir.exists():
                for font_file in font_dir.glob("*.ttf"):
                    # Copy to user fonts directory
                    dest_path = self.user_font_dir / font_file.name
                    shutil.copy2(font_file, dest_path)
                    print(f"  • Installed {font_file.name}")
                    
                    # Register font in Windows
                    if platform.system() == "Windows":
                        self._register_windows_font(dest_path)
                        
        # Update font cache on Linux
        if platform.system() != "Windows":
            os.system("fc-cache -f -v")
            
        print("✨ All Quantum Fonts Installed!")
        
    def _register_windows_font(self, font_path):
        """Register font in Windows registry"""
        import winreg
        
        font_name = font_path.stem
        font_path = str(font_path)
        
        # Add to Windows registry
        key_path = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, f"{font_name} (TrueType)", 0, winreg.REG_SZ, font_path)
        except PermissionError:
            print(f"  Note: Admin rights needed to register {font_name} in system registry")
            print(f"  Font still available in user directory: {font_path}")

if __name__ == "__main__":
    installer = QuantumFontInstaller()
    installer.install_fonts()
