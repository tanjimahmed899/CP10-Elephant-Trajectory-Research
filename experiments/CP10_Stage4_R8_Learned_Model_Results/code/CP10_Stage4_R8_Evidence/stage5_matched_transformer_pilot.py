#!/usr/bin/env python3
"""Paired learned-model R8 pilot: same architecture, seed, train/validation/test windows.
Train raw coordinate predictor on raw-coordinate future labels and corrected predictor on corrected labels.
Raw predictions are mapped (using recorded future affine camera transforms) to the observation-start
reference frame FOR EVALUATION ONLY. This comparison is retrospective, not a deployable
camera-motion-free predictor and does not establish exact-3D camera geometry.
"""
from pathlib import Path
import argparse,math,random,copy,json
import numpy as np,pandas as pd
import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader

D_MODEL=64;NHEAD=4;NUM_LAYERS=2;DIM_FEEDFORWARD=128;DROPOUT=.1;LR=5e-4;WEIGHT_DECAY=1e-4
class Position(nn.Module):
 def __init__(self):
  super().__init__();pe=torch.zeros(32,64);pos=torch.arange(32,dtype=torch.float32).unsqueeze(1)
  div=torch.exp(torch.arange(0,64,2,dtype=torch.float32)*(-math.log(10000.0)/64));pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div);self.register_buffer('pe',pe.unsqueeze(0))
 def forward(self,x): return x+self.pe[:,:x.size(1)]
class Transformer(nn.Module):
 def __init__(self,T):
  super().__init__();self.input_projection=nn.Linear(2,64);self.position=Position()
  layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,dropout=.10,activation='gelu',batch_first=True,norm_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=2);self.norm=nn.LayerNorm(64)
  self.head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.10),nn.Linear(64,T*2));self.T=T
 def forward(self,x):
  z=self.encoder(self.position(self.input_projection(x)));return self.head(self.norm(z.mean(dim=1))).view(-1,self.T,2)

class TD(Dataset):
 def __init__(self,h,f,w):self.h=torch.tensor(h,dtype=torch.float32);self.f=torch.tensor(f,dtype=torch.float32);self.w=torch.tensor(w,dtype=torch.float32)
 def __len__(self):return len(self.h)
 def __getitem__(self,i):return self.h[i],self.f[i],self.w[i]

def path_len(hist):return np.linalg.norm(np.diff(hist.astype(np.float64),axis=1),axis=2).sum(axis=1)
def canon(hist,future,floor):
 # TrainReal origin, recent-heading and path-length normalization exactly as original pipeline.
 origin=hist[:,-1,:].astype(np.float64)
 t=np.arange(10,dtype=np.float64);tc=t-t.mean();vel=((hist[:,-10:].astype(float)-hist[:,-10:].mean(axis=1,keepdims=True))*tc[None,:,None]).sum(axis=1)/(tc**2).sum()
 theta=np.arctan2(vel[:,1],vel[:,0]);theta[np.linalg.norm(vel,axis=1)<1e-10]=0.
 c=np.cos(theta);s=np.sin(theta)
 # R.T = [[c,-s],[s,c]]; rows coordinates x'=x*c-y*s, y'=x*s+y*c? Actually @R.T with R=[[c,s],[-s,c]] gives x'=x*c+y*s, y'=-x*s+y*c
 # Using row multiplication: [x,y]@R.T => [x*c+y*s, -x*s+y*c]
 scale=np.maximum(path_len(hist),floor)
 def fwd(x):
  dx=x.astype(float)-origin[:,None,:]
  return np.stack([(dx[...,0]*c[:,None]+dx[...,1]*s[:,None])/scale[:,None],(-dx[...,0]*s[:,None]+dx[...,1]*c[:,None])/scale[:,None]],axis=-1).astype(np.float32)
 return fwd(hist),fwd(future),origin,theta,scale

def inv(canon_pred,orig,ang,scale):
 c=np.cos(ang)[:,None];s=np.sin(ang)[:,None];x=canon_pred[...,0]*scale[:,None];y=canon_pred[...,1]*scale[:,None]
 # original inverse is pred_can @ R (R=[[c,s],[-s,c]]): new x = pred_x*c-pred_y*s ; new y=pred_x*s+pred_y*c
 return np.stack([x*c-y*s,x*s+y*c],axis=-1)+orig[:,None,:]

def map_to_target(raw,Q):
 return np.einsum('ntij,ntj->nti',Q,np.concatenate([raw,np.ones((*raw.shape[:2],1))],axis=-1))

def predict(model,x,batch=256):
 model.eval();out=[]
 with torch.no_grad():
  for s in range(0,len(x),batch):out.append(model(torch.tensor(x[s:s+batch],dtype=torch.float32)).numpy())
 return np.concatenate(out,axis=0)

def macro_track(ade,meta):return float(pd.DataFrame({'k':meta.track_key.to_numpy(),'g':meta.source_group.to_numpy(),'a':ade}).groupby(['g','k']).a.mean().mean())

def metrics(pred,target,meta):
 err=np.linalg.norm(pred-target,axis=-1);ade=err.mean(1);fde=err[:,-1]
 return dict(ADE=float(ade.mean()),FDE=float(fde.mean()),track_macro_ADE=macro_track(ade,meta),track_macro_FDE=macro_track(fde,meta),ADE_samples=ade,FDE_samples=fde)

def train_pair(root,out,task,seed,max_epochs,patience):
 train=np.load(root/f'R8_Paired_{task}_trainval.npz');test=np.load(root/f'R8_Paired_{task}_test.npz')
 mtr=pd.read_csv(root/f'R8_Paired_{task}_trainval_metadata.csv',dtype={'source_group':str,'track_key':str});mte=pd.read_csv(root/f'R8_Paired_{task}_test_metadata.csv',dtype={'source_group':str,'track_key':str})
 ixtr=train['train_indices'];ixval=train['validation_indices'];weights=train['train_sample_weights']
 pair_rows=[]; results=[]
 for branch in ('raw','corrected'):
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
  # maintain exactly same train/val indices, model capacity, seed, optimizer, budget
  htr=train[f'history_{branch}'];ftr=train[f'future_{branch}'];hte=test[f'history_{branch}'];fte=test[f'future_{branch}']
  floor=float(np.quantile(path_len(htr[ixtr]),.05))
  xc,yc,org,ang,scale=canon(htr,ftr,floor)
  xte,yte,ote,ate,ste=canon(hte,fte,floor)
  if branch=='corrected':
   # Must match previous original canonicalization within approximate floating tolerance; original used same train-only 5th percentile.
   print('corrected scale floor',floor,flush=True)
  dl=DataLoader(TD(xc[ixtr],yc[ixtr],weights),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
  model=Transformer(yc.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
  sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=10,min_lr=1e-5)
  best=float('inf');bestep=None;beststate=None;since=0
  for epoch in range(1,max_epochs+1):
   model.train(); losses=[]
   for xx,yy,ww in dl:
    opt.zero_grad();o=model(xx);loss=nn.functional.smooth_l1_loss(o,yy,reduction='none').mean((1,2));loss=(loss*ww).sum()/ww.sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach()))
   pred_can=predict(model,xc[ixval]);pred_native=inv(pred_can,org[ixval],ang[ixval],scale[ixval])
   pred_world=map_to_target(pred_native,train['Q_future'][ixval]) if branch=='raw' else pred_native
   mm=metrics(pred_world,train['future_corrected'][ixval],mtr.iloc[ixval])
   score=mm['track_macro_ADE'];sch.step(score)
   pair_rows.append({'task':task,'seed':seed,'branch':branch,'epoch':epoch,'train_loss':np.mean(losses),'val_ADE_common':mm['ADE'],'val_track_macro_ADE_common':score})
   if score<best-1e-5:best=score;bestep=epoch;since=0;beststate=copy.deepcopy(model.state_dict())
   else:since+=1
   if epoch==1 or epoch%20==0:print(task,seed,branch,'epoch',epoch,'val macro',round(score,6),'best',round(best,6),flush=True)
   if since>=patience:break
  model.load_state_dict(beststate)
  native=inv(predict(model,xte),ote,ate,ste)
  pred=map_to_target(native,test['Q_future']) if branch=='raw' else native
  res=metrics(pred,test['future_corrected'],mte)
  samples=mte[['source_index','source_group','track_key']].copy();samples['task']=task;samples['seed']=seed;samples['branch']=branch;samples['ADE']=res['ADE_samples'];samples['FDE']=res['FDE_samples'];samples.to_csv(out/f'R8_Learned_{task}_{branch}_seed{seed}_samples.csv',index=False)
  torch.save({'state_dict':beststate,'task':task,'branch':branch,'seed':seed,'epoch':bestep,'validation_track_macro_ADE':best,'model':'2-layer 64-width Transformer'},out/f'R8_Transformer_{task}_{branch}_seed{seed}.pt')
  results.append({'task':task,'branch':branch,'seed':seed,'train_n':len(ixtr),'validation_n':len(ixval),'test_n':len(mte),'best_epoch':bestep,'validation_track_macro_ADE':best,'test_ADE':res['ADE'],'test_FDE':res['FDE'],'test_track_macro_ADE':res['track_macro_ADE'],'test_track_macro_FDE':res['track_macro_FDE'],'raw_scale_floor':floor})
  print('FINAL',results[-1],flush=True)
 pd.DataFrame(pair_rows).to_csv(out/f'R8_Training_History_{task}_seed{seed}.csv',index=False)
 return results

def main():
 a=argparse.ArgumentParser();a.add_argument('--paired-root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--tasks',nargs='+',default=['2s_to_2s','2s_to_5s']);a.add_argument('--seeds',type=int,nargs='+',default=[42,52,62]);a.add_argument('--max-epochs',type=int,default=300);a.add_argument('--patience',type=int,default=30);z=a.parse_args();z.out.mkdir(exist_ok=True,parents=True)
 rows=[]
 for task in z.tasks:
  for seed in z.seeds:
   rows.extend(train_pair(z.paired_root,z.out,task,seed,z.max_epochs,z.patience))
   pd.DataFrame(rows).to_csv(z.out/'R8_Learned_Test_Summary.csv',index=False)
 print('Saved',z.out,flush=True)
if __name__=='__main__':main()
