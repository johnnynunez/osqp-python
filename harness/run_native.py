"""Build and exercise public upstream source on native hosted Windows ARM64."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

package, mode, expected_sha = sys.argv[1:]
root = pathlib.Path.cwd()
source = root / 'source'
stage = root / 'evidence'
stage.mkdir(exist_ok=True)
python = sys.executable

def run(label, args, cwd=None, required=True, timeout=700):
    print(label, args, flush=True)
    proc = subprocess.run(args, cwd=cwd or root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    (stage / (label+'.log')).write_text(proc.stdout, encoding='utf-8')
    print(proc.stdout, flush=True)
    if required and proc.returncode:
        raise RuntimeError(f'{label}: exit {proc.returncode}')
    return proc

actual = run('source-sha', ['git', 'rev-parse', 'HEAD'], source).stdout.strip()
assert actual == expected_sha, (actual, expected_sha)
run('submodules', ['git', 'submodule', 'status', '--recursive'], source)
provenance = {'package': package, 'mode': mode, 'source_sha': actual, 'wrapper_sha': os.environ['GITHUB_SHA'], 'run_url': f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}", 'scope': 'CPU native CPython 3.12; not a validated Isaac installation'}
if package == 'daqp' and mode == 'pin':
    url = 'https://files.pythonhosted.org/packages/3e/de/02ddf33b0de61788b3d7110febeb332837332ebdc4f1b83c1bf1f405efc4/daqp-0.8.5.tar.gz'
    data = urllib.request.urlopen(url).read()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == '71b0df979de85874bbed01c687a94e93926f445b4dddd236dcb3cea82431ab68'
    archive = stage / 'daqp-0.8.5.tar.gz'; archive.write_bytes(data)
    with tarfile.open(archive) as tf:
        tf.extractall(root / 'sdist', filter='data')
    project = root / 'sdist' / 'daqp-0.8.5'
    provenance['source_sdist'] = {'url': url, 'sha256': digest}
    provenance['source_sha_role'] = 'upstream tests only: v0.8.5 tag declares Python 0.8.6; wheel built from hash-verified published 0.8.5 sdist'
else:
    project = source / 'interfaces/daqp-python' if package == 'daqp' else source
(stage / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
run('tooling', [python, '-m', 'pip', 'install', 'build==1.6.1', 'setuptools==84.0.0', 'wheel', 'cmake==3.31.10', 'pybind11==3.1.0', 'scikit-build-core==1.1.1', 'setuptools-scm', 'cython==3.3.0', 'numpy==2.5.4', 'scipy==1.18.1', 'pytest==9.1.1', 'jinja2', 'joblib'])
# Preserve the old project-level SciPy pin failure; do not resolve it away.
run('original-scipy-pin', [python, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps', 'scipy==1.11.2', '-d', str(stage/'old-pin-probe')], required=False)
run('build', [python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', str(stage/'wheels'), str(project)])
wheels = list((stage/'wheels').glob('*.whl'))
assert len(wheels) == 1
# Fresh install environment independent of build/source checkout.
run('fresh-venv', [python, '-m', 'venv', str(root/'installed')])
testpython = str(root/'installed/Scripts/python.exe')
run('install', [testpython, '-m', 'pip', 'install', 'numpy==2.5.4', 'scipy==1.18.1', 'pytest==9.1.1', 'setuptools==84.0.0', 'pybind11==3.1.0', 'cmake==3.31.10', str(wheels[0])])
run('pip-check', [testpython, '-m', 'pip', 'check'])
run('freeze', [testpython, '-m', 'pip', 'freeze'])
# Execute independent numerical oracles even if upstream tests later fail.
run('oracle', [testpython, str(root/'harness/native_verify.py'), package, str(wheels[0]), str(stage/'native-verification.json')])
if package == 'osqp':
    testdir = source/'src/osqp/tests'
    # Explicit optional torch frontier, never counted as executed NN tests.
    torch = run('optional-torch-probe', [testpython, '-m', 'pip', 'install', '--only-binary=:all:', 'torch'], required=False, timeout=180)
    excludes = [] if torch.returncode == 0 else ['--ignore='+str(testdir/'nn_test.py')]
    (stage/'optional-coverage.json').write_text(json.dumps({'torch_install_exit': torch.returncode, 'excluded': excludes}), encoding='utf-8')
elif package == 'qdldl':
    testdir = source/'tests'; excludes = []
else:
    testdir = source/'interfaces/daqp-python/test'; excludes = []
run('upstream-tests', [testpython, '-m', 'pytest', str(testdir), '-ra', '--junitxml='+str(stage/'upstream.xml'), *excludes], timeout=900)
run('source-status-after', ['git', 'status', '--short'], source)
print('NATIVE BUILD, ORACLES, UPSTREAM TESTS PASS', flush=True)
