"""Native installed-wheel numerical and packaging verification (CPU only)."""
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import pathlib
import platform
import struct
import sys
import sysconfig
import zipfile

import numpy as np
import scipy
import scipy.sparse as sp

package, wheelpath, output = sys.argv[1:]
wheel = pathlib.Path(wheelpath)
result = {'package': package, 'wheel': wheel.name, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(), 'python': sys.version, 'executable': sys.executable, 'platform': sysconfig.get_platform(), 'machine': platform.machine(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'payloads': [], 'checks': []}

def pe(data):
    assert data[:2] == b'MZ'
    pos = struct.unpack_from('<I', data, 60)[0]
    assert data[pos:pos+4] == b'PE\x00\x00'
    return struct.unpack_from('<H', data, pos+4)[0]

assert sys.version_info[:2] == (3, 12)
assert sysconfig.get_platform() == 'win-arm64'
assert pe(pathlib.Path(sys.executable).read_bytes()) == 0xAA64
with zipfile.ZipFile(wheel) as z:
    records = [n for n in z.namelist() if n.endswith('.dist-info/RECORD')]
    assert len(records) == 1
    record = records[0]
    listed = set()
    for name, digest, size in csv.reader(io.StringIO(z.read(record).decode())):
        listed.add(name)
        data = z.read(name)
        if name != record:
            assert digest.startswith('sha256='), (name, digest)
            assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode() == digest.split('=', 1)[1]
            assert len(data) == int(size)
        if name.lower().endswith(('.pyd', '.dll', '.exe')):
            machine = pe(data)
            result['payloads'].append({'path': name, 'machine': hex(machine), 'sha256': hashlib.sha256(data).hexdigest()})
            assert machine == 0xAA64, (name, hex(machine))
    assert listed == {n for n in z.namelist() if not n.endswith('/')}
    assert result['payloads']
    result['metadata'] = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
mod = importlib.import_module(package)
result['module'] = mod.__file__
result['version'] = importlib.metadata.version(package)
assert 'site-packages' in mod.__file__.lower()
result['checks'].append('native interpreter + all wheel PE + RECORD + installed import')

def close(actual, expected, tol=1e-8):
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)

if package == 'qdldl':
    dense = np.array([[4., 1., 1.], [1., 3., 0.], [1., 0., -2.]])
    rhs = np.array([2., -1., 3.])
    for dtype in [np.float64, np.float32]:
        matrix = sp.csc_matrix(dense.astype(dtype))
        solver = mod.Solver(matrix)
        x = solver.solve(rhs)
        assert x.dtype == np.float64
        close(dense @ x, rhs)
        close(x, np.linalg.solve(dense, rhs))
        L, D, p = solver.factors()
        perm = np.eye(3)[:, p]
        unit = L.toarray() + np.eye(3)
        close(perm @ unit @ np.diag(D) @ unit.T @ perm.T, dense)
        updated = dense.copy(); updated[0, 0] += 2
        solver.update(sp.csc_matrix(updated))
        close(updated @ solver.solve(rhs), rhs)
    upper = mod.Solver(sp.triu(sp.csc_matrix(dense), format='csc'), upper=True)
    close(dense @ upper.solve(rhs), rhs)
    try:
        mod.Solver(sp.eye(3, format='csc')).solve(np.ones(2))
    except ValueError:
        pass
    else:
        raise AssertionError('wrong RHS shape accepted')
    result['checks'].append('float32/64 sparse solve + dense oracle + LDL reconstruction + update + upper + shape error')
elif package == 'osqp':
    P = sp.csc_matrix([[4., 1.], [1., 2.]])
    q = np.array([-1., -1.])
    expected = np.linalg.solve(P.toarray(), -q)
    for dtype in [np.float32, np.float64]:
        solver = mod.OSQP(algebra='builtin')
        solver.setup(P=P.astype(dtype), q=q.astype(dtype), A=sp.eye(2, format='csc', dtype=dtype), l=np.zeros(2), u=np.ones(2), eps_abs=1e-10, eps_rel=1e-10, verbose=False)
        res = solver.solve()
        assert res.info.status == 'solved' and res.info.status_val == 1
        assert res.x.dtype == np.float64
        close(res.x, expected)
        close(P @ res.x + q + res.y, np.zeros(2))
        assert np.all(res.x >= -1e-8) and np.all(res.x <= 1+1e-8)
        close(res.info.obj_val, 0.5 * expected @ P @ expected + q @ expected)
        solver.update(q=np.array([-10., -10.]))
        res = solver.solve()
        assert res.info.status == 'solved'
        close(res.x, np.ones(2))
    bad = mod.OSQP(algebra='builtin')
    bad.setup(P=sp.eye(1, format='csc'), q=np.zeros(1), A=sp.csc_matrix([[1.], [1.]]), l=np.array([1., -np.inf]), u=np.array([np.inf, 0.]), verbose=False)
    assert bad.solve().info.status == 'primal infeasible'
    result['checks'].append('float32/64 CSC QP + dense optimum + KKT + bounds + objective + update + infeasible status')
elif package == 'daqp':
    H = np.array([[4., 1.], [1., 2.]])
    f = np.array([-1., -1.])
    A = np.eye(2)
    upper = np.ones(2)
    lower = np.zeros(2)
    sense = np.zeros(2, dtype=np.int32)
    x, value, flag, info = mod.solve(H, f, A, upper, lower, sense)
    assert flag == 1, (flag, info)
    expected = np.linalg.solve(H, -f)
    close(x, expected)
    close(H @ x + f, np.zeros(2))
    close(value, 0.5 * expected @ H @ expected + f @ expected)
    assert x.dtype == np.float64 and np.all(x >= 0) and np.all(x <= 1)
    x, value, flag, info = mod.solve(H, np.array([-10., -10.]), A, upper, lower, sense)
    assert flag == 1
    close(x, np.ones(2))
    _, _, flag, _ = mod.solve(H, f, np.array([[1., 0.], [1., 0.]]), np.array([0., 3.]), np.array([-1., 1.]), sense)
    assert flag < 0, flag
    try:
        mod.solve(H.astype(np.float32), f, A, upper, lower, sense)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError('float32 should not silently pass double-buffer ABI')
    result['checks'].append('double/int32 ABI + analytic QP + gradient + bounds + objective + infeasible status + float32 rejection')
# The numerical acceptance must reject a poisoned answer, not just call solvers.
try:
    close(np.ones(2), np.zeros(2))
except AssertionError:
    result['checks'].append('poisoned-answer negative control rejected')
else:
    raise AssertionError('negative control failed')
result['status'] = 'PASS'
pathlib.Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
