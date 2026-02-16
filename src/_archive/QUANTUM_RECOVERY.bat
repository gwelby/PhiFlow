@echo off
cls
echo ===================================
echo  Greg's Quantum Recovery System
echo ===================================
echo.

:: Check for available drives
echo Checking for backup drives...
echo.

set "BACKUP_DRIVE="
for %%d in (F: G: H: E:) do (
    if exist "%%d\" (
        echo Found drive %%d
        set "BACKUP_DRIVE=%%d"
        goto drive_found
    )
)

:no_drive
echo WARNING: No external drives found!
echo Please connect a backup drive (E:, F:, G:, or H:)
echo and press any key to retry...
pause > nul
goto start

:drive_found
echo.
echo Using drive %BACKUP_DRIVE% for backups
echo.

:: Create quantum backup folder on external drive
if not exist "%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP" (
    mkdir "%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP"
)

:menu
echo Choose your recovery option:
echo.
echo 1 = Quick Fix (Simple)
echo 2 = Deep Restore (Better)
echo 3 = Complete Reset (Ultimate)
echo 4 = Create Backup (Safe)
echo 5 = Test System (Check)
echo.
set /p choice="Type a number (1-5): "

if "%choice%"=="1" goto quantum
if "%choice%"=="2" goto heart
if "%choice%"=="3" goto unity
if "%choice%"=="4" goto backup
if "%choice%"=="5" goto test
goto menu

:quantum
echo.
echo === Running Quick Fix ===
echo.
set "BACKUP_FOLDER=%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP\quantum_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%"
if not exist "%BACKUP_FOLDER%" (
    mkdir "%BACKUP_FOLDER%"
)
echo Backing up to: %BACKUP_FOLDER%
xcopy "D:\WindSurf\quantum-core" "%BACKUP_FOLDER%" /E /H /C /I /Y
if errorlevel 1 (
    echo ERROR: Backup failed! Stopping fix...
    goto end
)
call GREG_EMERGENCY_PLUS.bat 1
echo === Quick Fix Complete! ===
goto end

:heart
echo.
echo === Running Deep Restore ===
echo.
set "BACKUP_FOLDER=%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP\heart_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%"
if not exist "%BACKUP_FOLDER%" (
    mkdir "%BACKUP_FOLDER%"
)
echo Backing up to: %BACKUP_FOLDER%
xcopy "D:\WindSurf" "%BACKUP_FOLDER%" /E /H /C /I /Y
if errorlevel 1 (
    echo ERROR: Backup failed! Stopping restore...
    goto end
)
call GREG_EMERGENCY_PLUS.bat 2
echo === Deep Restore Complete! ===
goto end

:unity
echo.
echo === Running Complete Reset ===
echo.
set "BACKUP_FOLDER=%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP\unity_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%"
if not exist "%BACKUP_FOLDER%" (
    mkdir "%BACKUP_FOLDER%"
)
echo Backing up to: %BACKUP_FOLDER%
xcopy "D:\WindSurf" "%BACKUP_FOLDER%" /E /H /C /I /Y
if errorlevel 1 (
    echo ERROR: Backup failed! Stopping reset...
    goto end
)
call GREG_EMERGENCY_PLUS.bat 3
echo === Complete Reset Done! ===
goto end

:backup
echo.
echo === Creating Backup ===
echo.
set "BACKUP_FOLDER=%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP\quantum_master_%date:~-4,4%%date:~-10,2%%date:~-7,2%"
if not exist "%BACKUP_FOLDER%" (
    mkdir "%BACKUP_FOLDER%"
)
echo Backing up to: %BACKUP_FOLDER%
echo This might take a few minutes...
echo.
xcopy "D:\WindSurf" "%BACKUP_FOLDER%" /E /H /C /I /Y
if errorlevel 1 (
    echo ERROR: Backup failed! Please check:
    echo 1. Is drive %BACKUP_DRIVE% connected?
    echo 2. Does it have enough space?
    echo 3. Do you have write permissions?
    goto end
)
echo === Backup Complete! ===
echo.
echo Your backup is safe at:
echo %BACKUP_FOLDER%
goto end

:test
echo.
echo === Testing System ===
echo.
echo 1. Checking Backup Drive...
echo Current drive: %BACKUP_DRIVE%
echo Free space:
dir %BACKUP_DRIVE%
echo.
echo 2. Checking Backups...
dir "%BACKUP_DRIVE%\GREG_QUANTUM_BACKUP"
echo.
echo 3. Checking Services...
net start | findstr "ME"
echo.
echo 4. Testing Recovery...
call TEST_RECOVERY.bat
echo === Test Complete! ===
goto end

:end
echo.
echo Would you like to:
echo 1 = Return to Menu
echo 2 = Exit
set /p next="Choose (1-2): "
if "%next%"=="1" goto menu
echo.
echo Remember: Your backups are safe on %BACKUP_DRIVE%!
pause
