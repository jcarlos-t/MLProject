#!/usr/bin/env python3
"""Execute all nine notebooks in their documented dependency order."""
import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
for key, name in [('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('JUPYTER_CONFIG_DIR','jupyter_config')]:
    folder=ROOT/'scripts/data/runtime'/name;folder.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(folder)
for p in sorted((ROOT/'scripts/notebooks').glob('*.ipynb')):
    print('Ejecutando',p.name,flush=True)
    nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=14400,kernel_name='python3',resources={'metadata':{'path':str(p.parent)}}).execute()
    nbformat.write(nb,p)
    print('OK',p.name,flush=True)
