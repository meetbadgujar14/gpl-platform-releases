@echo off
setlocal

echo =====================================
echo  GPL Agents — Cleanup
echo =====================================
echo.
echo Current folder: %cd%
echo.

echo Removing all __pycache__ folders...
for /d /r %%D in (__pycache__) do (
    if exist "%%D" (
        echo   Deleting: %%D
        rd /s /q "%%D"
    )
)

echo.
echo Removing venv folder...
for /d /r %%D in (venv) do (
    if exist "%%D" (
        echo   Deleting: %%D
        rd /s /q "%%D"
    )
)

echo.
echo Done. Run python setup.py to reinstall.
pause
