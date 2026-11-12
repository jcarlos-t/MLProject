"""Exporta las particiones de la ejecucion corregida vigente, por identificador estable."""
from pathlib import Path
import json
from mlcore.experiment import current
from mlcore.data import read_data, TEST

def main():
    out = current(); df, _ = read_data(out)
    test = df.Season.isin(TEST); tv, te = df[~test], df[test]
    assert not set(tv.match_id) & set(te.match_id)
    assert tv.Date.max() < te.Date.min()
    tv.to_csv(out/'train_val.csv',index=False); te.to_csv(out/'test.csv',index=False)
    info = dict(run_id=out.name, train_val_n=len(tv), test_n=len(te), test_seasons=list(TEST),
                test_status='previamente observado; reevaluacion tras auditoria')
    (out/'split_info.json').write_text(json.dumps(info,indent=2))
    print(info)
if __name__ == '__main__': main()
