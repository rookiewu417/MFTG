#!/bin/bash
printf "TAG=%q\nARGS=%q\n" "$1" "$2" | sed "s/^TAG=\(.*\)/TAG='\1'/; s/^ARGS=\(.*\)/ARGS='\1'/" > /dev/null
{ echo "TAG='$1'"; echo "ARGS='$2'"; cat /home/ivenwu/projects/ml-factor/v2/colab/launch.py; } | timeout 120 colab exec -s mlf2 2>&1 | grep launched
