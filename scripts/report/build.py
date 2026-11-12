#!/usr/bin/env python3
"""Render results, validate references, compile both PDFs and check page limits."""
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from mlcore.report import render


def validate_sources():
    source = (ROOT/'docs/informe.tex').read_text()
    labels = re.findall(r'\\label\{([^}]+)\}', source)
    if len(labels) != len(set(labels)): raise ValueError('Labels duplicados')
    refs = re.findall(r'\\ref\{([^}]+)\}', source)
    if set(refs)-set(labels): raise ValueError('Referencias inexistentes')
    bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', source))
    citations = {s.strip() for group in re.findall(r'\\cite\{([^}]+)\}', source) for s in group.split(',')}
    if citations != bibs: raise ValueError(f'Bibliografia sin citar o citas inexistentes: {citations ^ bibs}')
    floats = []
    for match in re.finditer(r'\\begin\{(figure\*?|table)\}.*?\\end\{\1\}', source, re.S):
        label = re.search(r'\\label\{([^}]+)\}', match.group())
        if label:
            lab = label.group(1); first = source.find('\\ref{'+lab+'}')
            if first < 0 or first > match.start(): raise ValueError('Flotante sin citar antes: '+lab)
            floats.append((first, lab))
    if [x[0] for x in floats] != sorted(x[0] for x in floats): raise ValueError('Orden de flotantes incoherente')
    for name in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}', source):
        if not (ROOT/'docs/figures'/name).exists(): raise FileNotFoundError(name)
    deck=(ROOT/'docs/presentacion/presentacion.html').read_text()
    if len(re.findall(r'<section class="slide',deck)) != 13: raise ValueError('Se requieren 13 diapositivas')


def pages(pdf):
    info = subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    return int(re.search(r'^Pages:\s*(\d+)',info,re.M).group(1))


if __name__ == '__main__':
    render(); validate_sources()
    for cmd in ('tectonic','chromium','pdfinfo'):
        if not shutil.which(cmd): sys.exit('Falta dependencia de compilacion: '+cmd)
    subprocess.run(['tectonic','informe.tex'],cwd=ROOT/'docs',check=True)
    deck=ROOT/'docs/presentacion'
    subprocess.run(['chromium','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
        '--no-pdf-header-footer','--print-to-pdf='+str(deck/'presentacion.pdf'),
        (deck/'presentacion.html').as_uri()],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    paper_n, deck_n = pages(ROOT/'docs/informe.pdf'), pages(deck/'presentacion.pdf')
    if paper_n > 12: sys.exit(f'Informe de {paper_n} paginas: supera el maximo de 12')
    if deck_n != 13: sys.exit(f'Presentacion de {deck_n} paginas, se requieren 13')
    print(f'PDF verificados: informe {paper_n}/12 paginas; presentacion {deck_n}/13 paginas')
