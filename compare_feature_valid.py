import argparse, json
import numpy as np

def main(a):
    old=np.load(a.old,allow_pickle=False); new=np.load(a.new,allow_pickle=False)
    ko={(int(d),str(c)):i for i,(d,c) in enumerate(zip(old['feature_date'],old['stock_code']))}
    dif=[]; new_rows=0
    for d,c,p in zip(new['feature_date'],new['stock_code'],new['pred']):
        j=ko.get((int(d),str(c)))
        if j is None:
            new_rows += int(np.isfinite(p)); continue
        if np.isfinite(p) and np.isfinite(old['pred'][j]): dif.append(abs(float(p)-float(old['pred'][j])))
    r={'old_rows':len(old['pred']),'new_rows':len(new['pred']),'intersection':len(new['pred'])-new_rows,'new_finite_rows':new_rows,'maxabs':max(dif) if dif else None,'meanabs':float(np.mean(dif)) if dif else None}
    print(json.dumps(r,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);main(p.parse_args())
