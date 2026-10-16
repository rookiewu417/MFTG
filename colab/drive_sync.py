"""Google Drive transfer with the colab CLI's own login (its token carries the drive.file scope: files this client created).
Run with the colab CLI's python (~/.local/share/uv/tools/google-colab-cli/bin/python3: google-auth + requests).

  up FILE DRIVE_DIR      upload FILE into DRIVE_DIR (e.g. MFTG/e2eD5/e2e/dGR, folders created as needed); same name + size -> skip
  pull DRIVE_DIR DEST    download every file under DRIVE_DIR into DEST/<relative path>, skipping files already present
  ls DRIVE_DIR           list files under DRIVE_DIR

Token: $COLAB_TOKEN or ~/.config/colab-cli/token.json, refreshed in memory only (never written back). The token comes from the
gcloud ADC client, so Drive calls need a quota project with the Drive API enabled: $DRIVE_QUOTA_PROJECT (default rosy-zoo-507505-h7).
"""
import json, os, sys
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession

API, UP = 'https://www.googleapis.com/drive/v3/files', 'https://www.googleapis.com/upload/drive/v3/files'
FOLDER = 'application/vnd.google-apps.folder'
info = json.load(open(os.environ.get('COLAB_TOKEN', Path.home() / '.config/colab-cli/token.json')))
S = AuthorizedSession(Credentials.from_authorized_user_info(info, scopes=info['scopes']))
S.headers['x-goog-user-project'] = os.environ.get('DRIVE_QUOTA_PROJECT', 'rosy-zoo-507505-h7')


def children(pid, name=None, folder=None):
    q = [f"'{pid}' in parents", 'trashed=false']
    if name: q.append("name='%s'" % name.replace("'", "\\'"))
    if folder is not None: q.append(f"mimeType{'=' if folder else '!='}'{FOLDER}'")
    out, tok = [], None
    while True:
        r = S.get(API, params={'q': ' and '.join(q), 'fields': 'nextPageToken,files(id,name,size,mimeType)', 'pageSize': 1000,
                               **({'pageToken': tok} if tok else {})})
        r.raise_for_status(); j = r.json(); out += j['files']; tok = j.get('nextPageToken')
        if not tok: return out


def folder(path, create=False):
    pid = 'root'
    for part in [p for p in path.split('/') if p]:
        f = children(pid, part, True)
        if f: pid = f[0]['id']; continue
        if not create: return None
        r = S.post(API, json={'name': part, 'mimeType': FOLDER, 'parents': [pid]}, params={'fields': 'id'}); r.raise_for_status()
        pid = r.json()['id']
    return pid


def up(file, ddir):
    p = Path(file); pid = folder(ddir, create=True)
    if any(int(f.get('size', -1)) == p.stat().st_size for f in children(pid, p.name, False)): print('exists', p.name); return
    r = S.post(UP, params={'uploadType': 'resumable'}, json={'name': p.name, 'parents': [pid]}); r.raise_for_status()
    with open(p, 'rb') as fh:
        r = S.put(r.headers['Location'], data=fh, headers={'Content-Length': str(p.stat().st_size)})
    r.raise_for_status(); print('uploaded', ddir + '/' + p.name)


def walk(pid, rel=''):
    for f in children(pid):
        if f['mimeType'] == FOLDER: yield from walk(f['id'], f"{rel}{f['name']}/")
        else: yield rel + f['name'], f


def pull(ddir, dest):
    pid = folder(ddir)
    if pid is None: sys.exit(f'no such Drive folder: {ddir}')
    for rel, f in walk(pid):
        t = Path(dest) / rel
        if t.exists() and t.stat().st_size == int(f.get('size', -1)): continue
        t.parent.mkdir(parents=True, exist_ok=True)
        with S.get(f'{API}/{f["id"]}', params={'alt': 'media'}, stream=True) as r:
            r.raise_for_status()
            with open(str(t) + '.part', 'wb') as fh:
                for chunk in r.iter_content(1 << 20): fh.write(chunk)
        os.replace(str(t) + '.part', t); print('got', rel)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'up': up(*a)
    elif cmd == 'pull': pull(*a)
    elif cmd == 'ls':
        pid = folder(a[0])
        for rel, f in (walk(pid) if pid else []): print(f.get('size', '-'), rel)
