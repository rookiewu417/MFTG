# `colab exec` for the keeper (run with the colab CLI's own python, see keeper.sh).
# Before the user's cell, `colab exec` runs a setup cell without a timeout, so jupyter_kernel_client applies its 10 s
# default; behind the busy driver cell that deadline passes and its wait loop calls Event.wait(0) forever -> one local
# core at 100% for the keeper's whole life. Give calls without a timeout a very long one instead.
import sys
from colab_cli import runtime
from colab_cli.cli import main

_execute_code = runtime.ColabRuntime.execute_code


def execute_code(self, code, *a, timeout=None, **k):
    return _execute_code(self, code, *a, timeout=1e7 if timeout is None else timeout, **k)


runtime.ColabRuntime.execute_code = execute_code
sys.argv = ['colab'] + sys.argv[1:]
sys.exit(main())
