# exec after defining TAG and ARGS in the kernel, e.g.  TAG='r1'; ARGS='--models xgb'
import subprocess, os, shlex
os.makedirs('/content/out', exist_ok=True)
logf = open(f'/content/out/{TAG}.log', 'w')
p = subprocess.Popen(['python', 'train.py', '--out', f'/content/out/{TAG}'] + shlex.split(ARGS), cwd='/content/code',
                     stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
print('launched', TAG, p.pid)
