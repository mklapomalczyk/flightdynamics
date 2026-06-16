# with open(r"C:\Users\mklap\Desktop\Python_projekty\FlightSim\datcom_runs\rocket_70mm_baseline\datcom.out") as f:
    # content = f.read()
# # Znajdz sekcje z bledami
# for i, line in enumerate(content.splitlines()):
    # if 'ERROR' in line or 'error' in line.lower() or 'ALPHA' in line:
        # print(f"{i:4d}: {line}")
        
with open(r"C:\Users\mklap\Desktop\Python_projekty\FlightSim\datcom_runs\rocket_70mm_baseline\datcom.out") as f:
    lines = f.readlines()
for i in range(149, 180):
    print(f"{i+1:4d}: {lines[i].rstrip()}")