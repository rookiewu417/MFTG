import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # baseline_src/
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from factor_postprocess import process

with tempfile.TemporaryDirectory() as td:
    root=Path(td); (root/'meta').mkdir()
    codes=np.array(['A','B','C','D']); dates=np.array([20240101,20240108,20240115,20240122],dtype=np.int32)
    np.savez(root/'p.npz',pred=np.array([1.,2.,3.,4.]*4,dtype=np.float32),feature_date=np.repeat(dates,4),stock=np.tile(np.arange(4),4),valid=np.ones(16,bool),codes=codes)
    pd.DataFrame({'ts_code':codes,'name':['A','B','C','D'],'list_date':pd.to_datetime(['2020-01-01']*4),'delist_date':[pd.NaT]*4}).to_parquet(root/'meta/stock_basic.parquet')
    pd.DataFrame({'ts_code':['B'],'name':['ST B'],'start_date':pd.to_datetime(['2024-01-10']),'end_date':[pd.NaT],'ann_date':pd.to_datetime(['2024-01-20']),'change_reason':['x']}).to_parquet(root/'meta/name.parquet')
    pd.DataFrame({'index_code':['801010.SI'],'industry_name':['x'],'level':['L1'],'src':['SW2021']}).to_parquet(root/'meta/industry_criteria.parquet')
    pd.DataFrame({'index_code':['801010.SI']*3,'ts_code':['A','A','B'],'in_date':pd.to_datetime(['2020-01-01','2024-01-15','2020-01-01']),'out_date':pd.to_datetime(['2024-01-14',None,None]),'is_new':['1','1','1']}).to_parquet(root/'meta/sw_index_member.parquet')
    s=process(root/'p.npz',root/'o.npz',root,str(root/'codes.csv')) if False else None
    pd.Series(codes).to_csv(root/'codes.csv',index=False,header=False)
    s=process(root/'p.npz',root/'o.npz',root,str(root/'codes.csv'),industry_src='SW2021')
    assert s['filtered_stock_status']==1  # only post-announcement ST row is filtered
    z=np.load(root/'o.npz'); assert z['valid'][1*4+1]  # B remains valid before announcement
    assert z['valid'][2*4+1]  # B is still valid before announcement
    assert not z['valid'][3*4+1]  # B is filtered once announcement is known
    assert not z['valid'][3*4+2]  # C has no industry and is invalid for industry-only output
    for dt in dates:
        x=z['pred'][(z['valid']) & (z['feature_date']==dt)]
        if len(x)>1: assert abs(float(x.mean()))<1e-6 and abs(float(x.std())-1)<1e-5
    print('ASOF_ST_INDUSTRY_RESIDUAL=PASS',s)
