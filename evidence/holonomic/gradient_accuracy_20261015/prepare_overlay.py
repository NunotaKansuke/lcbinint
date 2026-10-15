from pathlib import Path
import hashlib, json, shutil
root = Path(__file__).resolve().parents[3]
source = root / 'benchmarks/holonomic/fused_gradient'
out = root / 'build-algebraic-boundary/gradient-probe'
for name in ['build_overlay.py', 'build_material_overlay.py', 'build_api_overlay.py']:
    path = source / name
    text = path.read_text().replace('build-seeded-events/fused-gradient', 'build-algebraic-boundary/gradient-probe')
    if name == 'build_material_overlay.py':
        text = text.replace("runpy.run_path(str(Path(__file__).with_name('build_overlay.py')))", '')
    exec(compile(text, str(path), 'exec'), {'__file__': str(path), '__name__': '__main__'})
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src/lcbinint/magnification/holonomic').glob('*.hpp')}
(out/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(out/'api-include')
