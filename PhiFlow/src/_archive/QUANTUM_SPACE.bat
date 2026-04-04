@echo off
cls
echo ===================================
echo  Greg's Quantum Space Analyzer
echo ===================================
echo.

:: Check E: drive
echo Analyzing E: drive...
echo.
dir E: | findstr "bytes free"

:: Calculate WindSurf size
echo.
echo Analyzing WindSurf space...
dir /s "D:\WindSurf" | findstr "File(s)"

:: Quantum Compression Ratios (based on phi φ)
set PHI=1.618034
set PHI_2=2.618034
set PHI_3=4.236068

echo.
echo === Quantum Space Analysis ===
echo Using Golden Ratio (φ) compression:
echo.
echo 1. Raw Size: Above is your current size
echo 2. φ¹ (Level 1): Divide by %PHI%
echo 3. φ² (Level 2): Divide by %PHI_2%
echo 4. φ³ (Level 3): Divide by %PHI_3%
echo.

:: Check backup location
echo === Backup Location ===
if exist "E:\GREG_QUANTUM_BACKUP" (
    echo Current backups on E:
    dir /s "E:\GREG_QUANTUM_BACKUP" | findstr "File(s)"
) else (
    echo No existing backups found on E:
)

echo.
echo === Quantum Storage Requirements ===
echo Ground (432 Hz): ~432 MB per backup
echo Heart  (528 Hz): ~528 MB per backup
echo Unity  (768 Hz): ~768 MB per backup
echo.
echo Remember: Actual size will be divided by φ^level
echo - Level 1 (φ¹): %PHI% compression
echo - Level 2 (φ²): %PHI_2% compression
echo - Level 3 (φ³): %PHI_3% compression
echo.
pause
