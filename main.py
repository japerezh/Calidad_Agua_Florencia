# ============================================================
# main.py
# Proyecto: Calidad del Agua — Caquetá
# Orquestador del pipeline completo
# Ejecutar DESPUÉS de: preprocessing.py → models.py → alerts.py → validation.py
# O ejecutar directamente: python main.py (corre todos los scripts)
# ============================================================

import subprocess
import sys
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("=" * 65)
print("  PIPELINE COMPLETO — CALIDAD DEL AGUA CAQUETÁ")
print(f"  {TIMESTAMP}")
print("=" * 65)

scripts = [
    ('src/preprocessing.py', '[1/4] PREPROCESAMIENTO'),
    ('src/models.py',        '[2/4] MODELOS IA'),
    ('src/alerts.py',        '[3/4] MOTOR DE ALERTAS'),
    ('src/validation.py',    '[4/4] VALIDACIÓN TÉCNICA'),
]

for script, label in scripts:
    print(f"\n{label}...")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"\n  ERROR en {script} — pipeline detenido.")
        sys.exit(1)

print(f"\n{'='*65}")
print("  PIPELINE COMPLETO")
print(f"  Todos los archivos actualizados en reports/")
print(f"  Ejecutado: {TIMESTAMP}")
print(f"{'='*65}")