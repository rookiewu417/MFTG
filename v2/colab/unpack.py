import subprocess, os, glob
parts = sorted(glob.glob('/content/up/pack.tgz.part*'))
print('parts', len(parts))
subprocess.run("cat /content/up/pack.tgz.part* | tar -xzf - -C /content && rm -rf /content/data && mv /content/pack /content/data", shell=True, check=True)
print(subprocess.run("du -sh /content/data; ls -la /content/data", shell=True, capture_output=True, text=True).stdout)
