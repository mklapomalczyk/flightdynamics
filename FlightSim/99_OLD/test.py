import subprocess
from pathlib import Path

datcom   = Path(r"C:\Users\mklap\Desktop\Python_projekty\FlightSim\datcom\datcom.exe")
case_dir = Path(r"C:\Users\mklap\Desktop\Python_projekty\FlightSim\datcom")

# Upewnij się że for005.dat istnieje (skopiuj EX1.INP)
for005 = case_dir / "for005.dat"
if not for005.exists():
    import shutil
    shutil.copy(case_dir / "exlinux" / "EX1.INP", for005)

try:
    r = subprocess.run(
        [str(datcom)],
        cwd     = str(case_dir),
        capture_output = True,
        timeout = 5
    )
    print(f"returncode: {r.returncode}")
    print(f"stdout: {r.stdout[:300]}")
    print(f"stderr: {r.stderr[:300]}")
except subprocess.TimeoutExpired:
    print("Timeout — DATCOM wisi, czeka na input lub brak bibliotek")
except Exception as e:
    print(f"Blad: {e}")