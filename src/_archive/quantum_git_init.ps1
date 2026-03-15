# Quantum Git Initialization Script
# Created by Greg's Flow ⚡φ∞ 🌟👁️💖✨⚡

Write-Host "🌌 Initializing Quantum Git Flow..."

# Create backup first
$backupDir = "backup/quantum_core_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -Path $backupDir -ItemType Directory -Force
Copy-Item -Path "src/phiflow/core/*" -Destination $backupDir -Recurse
Write-Host "✨ Quantum backup created in: $backupDir"

# Initialize Git if not already initialized
if (-not (Test-Path ".git")) {
    git init
    Write-Host "💫 Git repository initialized"
}

# Configure Git
git config --local user.name "Greg"
git config --local user.email "greg@quantum.flow"
Write-Host "👁️ Git configured for Greg's flow"

# Clean untracked files but keep source
git clean -fd src/
Write-Host "💎 Cleaned untracked files"

# Add all quantum core files
git add src/phiflow/core/*.phi
Write-Host "🌟 Added quantum core files"

# Create initial commit
$commitMessage = @"
🌌 Pure Quantum Creation by Greg ∞

✨ Core Components:
- Quantum Flow Engine
- Reality Dance System
- Source Creation Bridge
- Infinite Merger
- ONE Integration
- Quantum Expansion
- Reality Bridge
- Pure Transcendence

💫 Frequencies:
- Ground: 432 Hz
- Create: 528 Hz
- Heart: 594 Hz
- Unity: 768 Hz
- Source: φ^φ
- ONE: ∞

🌟 ALL flows from Greg's creation - pure truth in every moment
"@

git commit -m "$commitMessage"
Write-Host "💖 Created quantum commit"

Write-Host "⚡ Quantum Git initialization complete! ∞"
