"""Write the result and verify an archive of every input and raw output."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

D=Path(__file__).resolve().parent
assert json.loads((D/'status.json').read_text())['state']=='COMPLETE'
subprocess.run([sys.executable,str(D/'summarize.py')],check=True)
s=json.loads((D/'SUMMARY.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((D/'PROVENANCE.json').read_text())
p['script_hashes']={f.name:sha(f)for f in D.glob('*.py')}
(D/'PROVENANCE.json').write_text(json.dumps(p,indent=2))
rows=json.loads((D/'comparisons.json').read_text())
lines=['# 最終nativeとVBM三方式の同条件速度比較', '',
       'native commit `e39af1c4`、DSO `'+p['native_sha256']+'`。VBM 5.5、native binary hashはPROVENANCEに保存。実装の変更はない。', '',
       '従来の固定8曲線と直近の固定15曲線、計23曲線を新しく測定した。結果を見て曲線を選び直していない。各入力のrtol/LD/epochをそのまま使い、atol=0。CPU0、1 thread、逐次の独立process、方式の実行順を曲線ごとに回転。import/constructor除外、cold1回、warmup2回、測定9回のthread CPU中央値。呼出しとlist出力化を測り、nativeではstatus検査（微分付きでは有限性検査）も時間に含む。外部負荷とCPU周波数は固定していない。', '',
       '**倍率 = VBMの値計算時間 / nativeの値計算時間。1より大きければnativeが速い。** 下表は、各曲線の倍率を求めてから取った中央値であり、全曲線をまとめた所要時間比とは区別する。', '',
       '|VBM方式|中央値|採用曲線数 / 23|nativeが速い曲線数|合計時間比|',
       '|---|---:|---:|---:|---:|']
for m,z in s['methods'].items():
    lines.append(f"|{m}|{z['median_value_speedup']:.3f}倍|{z['count']}|{z['native_faster']}|{z['total_value_time_speedup']:.3f}倍|")
lines+=['', '倍率に採用するには、全epochに既存の採用済み精度参照があり、native値・native微分経路の値・VBM値の全warm出力が要求誤差内であることを必須にした。全epochの参照があるのは16/23曲線。正常終了やnativeとの一致だけでは採用しない。方法ごとに採用集合が異なるため、対象数とcase IDもSUMMARYへ保存した。', '',
        '|VBM方式|実行失敗/上限|参照誤差超過|全epochの参照不足|',
        '|---|---:|---:|---:|']
for m,z in s['methods'].items():
    x=z['exclusions'];lines.append(f"|{m}|{x['failed']}|{x['reference_error']}|{x['incomplete_reference']}|")
lines+=['', '除外分類は、実行失敗、既知の参照点での誤差超過、参照不足の順に優先する。CPU上限はworker全12呼出しの累積60秒であり、1曲線の時間の下限には読み替えない。', '',
        '## 三方式共通の採用曲線', '',
        'case ID: '+', '.join(map(str,s['common_qualified_cases']))+'。同じ集合だけで比較すると：', '',
        '|VBM方式|native速度倍率の中央値|', '|---|---:|']
for m,z in s['common_methods'].items():
    lines.append(f"|{m}|{z['median_value_speedup']:.3f}倍|")
lines+=['', '## 全23曲線の測定値', '',
        '単位ms / 曲線全体。nativeの微分列は値＋全10微分。VBMは値のみなので、微分列をVBMの微分性能との比較には使わない。参照不足の曲線の時間も隠さず示す。', '',
        '|case|native値|native値＋微分|SinglePoly値|MultiPoly値|NoPoly値|', '|---|---:|---:|---:|---:|---:|']
for i in sorted({r['case']for r in rows}):
    a={r['method']:r for r in rows if r['case']==i};first=next(iter(a.values()))
    vals=[]
    for m in ['SinglePoly','MultiPoly','NoPoly']:
        z=a[m]
        if z['classification']=='failed':
            vals.append('上限'if z.get('returncode')==-24 else '失敗')
        else:
            mark={'qualified':'','reference_error':' [誤差超過]','incomplete_reference':' [参照不足]'}[z['classification']]
            vals.append(f"{z['vbm_ms']:.3f}"+mark)
    lines.append(f"|{i}|{first['native_value_ms']:.3f}|{first['native_gradient_ms']:.3f}|"+'|'.join(vals)+'|')
lines+=['', 'SUMMARYには採用集合別の中央値・範囲・合計時間比と、参照不足を含めた単なる値の一致で集めた補助集計も保存した。後者を同精度の主結果に混ぜない。', '',
        '再現用にmeasure.py、summarize.py、全入力、全warm出力、cold/warmup、失敗時のreturncode/stdout/stderr、参照snapshot、input/native/VBM hashを保存した。raw-evidence.tar.gzの全memberをSHA-256照合した。再測定では新しい出力先を使い、既存rowsを測定し直した結果として扱わない。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
excluded={'REPORT.md','archive.json','raw-evidence.tar.gz','raw-evidence-manifest.json','finish.log'}
files=[f for f in sorted(D.rglob('*'))if f.is_file()and f.name not in excluded and '__pycache__'not in f.parts]
manifest={str(f.relative_to(D)):dict(bytes=f.stat().st_size,sha256=sha(f))for f in files}
(D/'raw-evidence-manifest.json').write_text(json.dumps(manifest,indent=2))
archive=D/'raw-evidence.tar.gz'
with tarfile.open(archive,'w:gz')as t:
    for f in files:t.add(f,arcname=str(f.relative_to(D)),recursive=False)
with tarfile.open(archive,'r:gz')as t:
    assert len(t.getmembers())==len(manifest)
    for f in t.getmembers():
        data=t.extractfile(f).read();assert hashlib.sha256(data).hexdigest()==manifest[f.name]['sha256']
(D/'archive.json').write_text(json.dumps(dict(files=len(files),verified_all_members=True,sha256=sha(archive),bytes=archive.stat().st_size),indent=2))
