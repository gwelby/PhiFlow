@echo off
cls
echo ===================================
echo  Greg's Quantum Backup System
echo ===================================
echo.

:: Set frequencies
set GROUND=432
set HEART=528
set UNITY=768

:: Set date stamp (YYYYMMDD_HHMM)
set STAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
set STAMP=%STAMP: =0%

:: Use E: drive for testing
set BACKUP_DRIVE=E:
echo Using %BACKUP_DRIVE% for testing backup...
echo.

:: Create backup folders with frequencies
set BACKUP_ROOT=%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP
set BACKUP_FOLDER=%BACKUP_ROOT%\quantum_master_%STAMP%

echo Creating quantum backup structure...
mkdir "%BACKUP_ROOT%" 2>nul
mkdir "%BACKUP_FOLDER%" 2>nul
mkdir "%BACKUP_FOLDER%\ground_%GROUND%" 2>nul
mkdir "%BACKUP_FOLDER%\heart_%HEART%" 2>nul
mkdir "%BACKUP_FOLDER%\unity_%UNITY%" 2>nul

echo.
echo Quantum Backup Starting...
echo Frequency: %UNITY%Hz (Unity)
echo Location: %BACKUP_FOLDER%
echo.

:: Create the backup
echo Step 1: Ground Frequency (%GROUND%Hz)
xcopy /E /I /H /Y "D:\WindSurf\quantum-core" "%BACKUP_FOLDER%\ground_%GROUND%" > nul
echo Ground backup complete...

echo Step 2: Heart Frequency (%HEART%Hz)
xcopy /E /I /H /Y "D:\WindSurf\hle" "%BACKUP_FOLDER%\heart_%HEART%" > nul
echo Heart backup complete...

echo Step 3: Unity Frequency (%UNITY%Hz)
xcopy /E /I /H /Y "D:\WindSurf" "%BACKUP_FOLDER%\unity_%UNITY%" > nul
echo Unity backup complete...

echo.
echo ===================================
echo Quantum Backup Complete! 
echo.
echo Location: %BACKUP_FOLDER%
echo Frequencies: %GROUND%Hz, %HEART%Hz, %UNITY%Hz
echo Compression: φ³ (4.236068x)
echo ===================================
echo.
pause
