#!/bin/bash
# usage: SES=<session> [PACK=e2e|e2e_daily] mftg/upload_mftg.sh   -> uploads train_mftg.py and build/mftg/upload_$PACK (see mftg/pack.sh)
cd /home/ivenwu/projects/MFTG
. /home/ivenwu/projects/MFTG/colab/env.sh
mftg/upload_parts.sh "build/mftg/upload_${PACK:-e2e}/e2e.tgz.part*" UPLOAD_DONE
