# sourced by the other colab/*.sh scripts
SES=${SES:-e2eA}                                  # Colab session name
LINE=${LINE:-mftg}                                # which line the results belong to: mftg | repro  -> runs/$LINE/
MFTG_ROOT=/home/ivenwu/projects/MFTG
SCRATCH=${MFTG_SCRATCH:-/tmp/mftg_colab}; mkdir -p "$SCRATCH"   # small temp files (status, job json, markers)
K=$MFTG_ROOT/build/keeper
# Google account of the session: main = the default login; others live under ~/.colab-accounts/<name> (colab/login.sh).
# Recorded per session in build/keeper/account_<SES> by colab/start.sh, so every script picks the right account.
ACCT=${ACCT:-$(cat $K/account_$SES 2>/dev/null)}; ACCT=${ACCT:-main}
if [ "$ACCT" != main ]; then export COLAB_ACCT=$ACCT; case ":$PATH:" in *":$MFTG_ROOT/colab/acct:"*) ;; *) export PATH=$MFTG_ROOT/colab/acct:$PATH;; esac; fi
