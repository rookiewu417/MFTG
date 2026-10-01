#!/bin/bash
# usage: tail.sh TAG [n]
echo "import subprocess; print(subprocess.run('tail -n ${2:-20} /content/out/$1.log', shell=True, capture_output=True, text=True).stdout)" | timeout 120 colab exec -s mlf2 2>&1 | grep -v -i "new version\|colab update\|silence"
