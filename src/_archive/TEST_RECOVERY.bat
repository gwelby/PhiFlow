@echo off
echo 🔍 Testing Recovery System
echo.

echo 1. Testing Quick Fix...
call GREG_EMERGENCY_PLUS.bat 1
echo.

echo 2. Verifying Services...
sc query MEProtection
sc query IntelMEService
sc query ShieldService
echo.

echo 3. Checking Access...
reg query "HKLM\SYSTEM\CurrentControlSet\Control\MEAccess" /v Enabled
echo.

echo 4. Testing Deep Clean...
call GREG_EMERGENCY_PLUS.bat 2
echo.

echo 5. Verifying Files...
dir "D:\WindSurf\quantum-core\protection\state"
echo.

echo ✨ All Tests Complete!
echo.
pause
