# Runs INSIDE the Colab kernel and never returns until /content/queue/STOP exists.
# Colab keeps a VM alive only while its kernel is active, so the training jobs are driven from this cell:
#   - unpacks /content/up/pack.tgz.part* once /content/up/UPLOAD_DONE appears
#   - runs job files /content/queue/*.json ({"tag", "args", "lane"}) one at a time per lane
#   - writes /content/out/status.json for polling through the contents API
import glob, json, os, shlex, subprocess, time

Q, OUTD = '/content/queue', '/content/out'
for d in (Q, OUTD, '/content/up', '/content/code'): os.makedirs(d, exist_ok=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
running = {}          # lane -> (job dict, Popen)
if os.path.exists(f'{Q}/STOP'): os.remove(f'{Q}/STOP')      # a fresh driver ignores a previous stop request
done = []
t0 = time.time(); last_beat = 0

def data_ready():
    if os.path.exists('/content/repro_early/READY'): return True
    if not os.path.exists('/content/up/UPLOAD_DONE'): return False
    subprocess.run('cat /content/up/repro.tgz.part* | tar -xzf - -C /content && touch /content/repro_early/READY && rm -f /content/up/repro.tgz.part*',
                   shell=True, check=True)
    print('unpacked', subprocess.run('du -sh /content/repro_early', shell=True, capture_output=True, text=True).stdout, flush=True)
    return True

def write_status():
    st = {'uptime_min': round((time.time() - t0) / 60, 1), 'data': os.path.exists('/content/repro_early/READY'),
          'running': {str(k): v[0]['tag'] for k, v in running.items()},
          'queued': sorted(os.path.basename(p) for p in glob.glob(f'{Q}/*.json')), 'done': done}
    with open(f'{OUTD}/status.json.tmp', 'w') as f: json.dump(st, f, indent=1)
    os.replace(f'{OUTD}/status.json.tmp', f'{OUTD}/status.json')
    return st

while not os.path.exists(f'{Q}/STOP'):
    if data_ready():
        for lane, (job, p) in list(running.items()):
            if os.path.exists(f"{Q}/CANCEL_{job['tag']}"):          # per-job cancel: touch /content/queue/CANCEL_<tag>
                p.terminate(); os.remove(f"{Q}/CANCEL_{job['tag']}"); print('cancelled', job['tag'], flush=True)
            if p.poll() is not None:
                done.append({'tag': job['tag'], 'lane': lane, 'rc': p.returncode, 'min': round((time.time() - job['t']) / 60, 1)})
                print('finished', done[-1], flush=True)
                del running[lane]
        for path in sorted(glob.glob(f'{Q}/*.json')):
            job = json.load(open(path)); lane = job.get('lane', 0)
            if lane in running: continue
            os.remove(path)
            job['t'] = time.time()
            log = open(f"{OUTD}/{job['tag']}.log", 'a')
            out = job.get('out', job['tag'])
            p = subprocess.Popen(['python', job.get('script', 'train_e2e.py'), '--out', f'{OUTD}/{out}'] + shlex.split(job['args']), cwd='/content/code',
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[lane] = (job, p)
            print('started', job['tag'], 'lane', lane, flush=True)
    st = write_status()
    if time.time() - last_beat > 60:
        print('beat', st['uptime_min'], 'running', st['running'], 'queued', len(st['queued']), flush=True); last_beat = time.time()
    time.sleep(10)
for job, p in running.values(): p.terminate()
print('driver stopped', flush=True)
