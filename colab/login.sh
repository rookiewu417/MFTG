#!/bin/bash
# usage: colab/login.sh NAME   -> one-time OAuth login of a secondary Google account (interactive: open the URL, paste the code)
# main stays the default login (~/.config/colab-cli/token.json).
set -e
[ -n "$1" ] && [ "$1" != main ] || { echo "usage: colab/login.sh NAME (NAME != main)"; exit 1; }
mkdir -p /home/ivenwu/.colab-accounts/$1
COLAB_ACCT=$1 /home/ivenwu/projects/MFTG/colab/acct/colab usage
