import subprocess
print(subprocess.run("ps aux | grep '[t]rain.py' | awk '{print $2, $3, $4, $12, $13, $14, $15}'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; free -g | sed -n 2p", shell=True, capture_output=True, text=True).stdout)
