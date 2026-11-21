#!/usr/bin/env python3
import numpy as np
import pandas as pd
from stage7e_gt_linkage import canon_id,iou_matrix,match_one_frame
assert canon_id('1.0')=='1'
assert canon_id(3)=='3'
gt=pd.DataFrame({'gt_id':['1','2'],'x':[0,40],'y':[0,40],'w':[20,20],'h':[20,20]})
pr=pd.DataFrame({'x1':[0,41,80],'y1':[0,41,80],'x2':[20,61,90],'y2':[20,61,90]},index=[7,8,9])
r=match_one_frame(gt,pr)
assert sorted(r)==[7,8] and r[7][0]=='1' and r[8][0]=='2'
assert len(match_one_frame(gt,pr.iloc[[2]]))==0
assert abs(iou_matrix(np.array([[0,0,20,20]]),np.array([[0,0,20,20]]))[0,0]-1)<1e-12
for n,fc,expected in [(40,20,1),(70,50,1),(79,50,1),(80,50,2)]:
 starts=list(range(0,n-(20+fc)+1,10))
 assert len(starts)==expected,(n,starts,expected)
print('Stage7E synthetic tests passed. NOT research results.')
