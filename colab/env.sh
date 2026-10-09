# sourced by the other colab/*.sh scripts
SES=${SES:-e2eA}                                  # Colab session name
LINE=${LINE:-mftg}                                # which line the results belong to: mftg | repro  -> runs/$LINE/
MFTG_ROOT=/home/ivenwu/projects/MFTG
SCRATCH=${MFTG_SCRATCH:-/tmp/mftg_colab}; mkdir -p "$SCRATCH"   # small temp files (status, job json, markers)
