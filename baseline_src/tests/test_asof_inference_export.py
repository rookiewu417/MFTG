import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baseline_src/
import tempfile
from pathlib import Path
import numpy as np
from asof_inference_export import export_asof

with tempfile.TemporaryDirectory() as t:
    src=Path(t)/'src.npz'; out=Path(t)/'out.npz'
    pred=np.arange(6,dtype=np.float32); fd=np.array([20240101,20240101,20240108,20240108,20240115,20240115],np.int32)
    np.savez(src,pred=pred,feature_date=fd,valid=np.ones(6,bool),label_end=fd+10)
    r=export_asof(src,out,20240108,20240115); z=np.load(out)
    assert r['label_used_for_valid'] is False and np.array_equal(z['pred'],pred)
    assert z['valid'].tolist()==[False,False,True,True,True,True]
print('ASOF_EXPORT_PRED_PRESERVED=PASS')
