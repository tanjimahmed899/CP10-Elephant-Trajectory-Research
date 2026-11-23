#!/usr/bin/env python3
"""Exact Phase 7I motion feature implementation, observation history only."""
import numpy as np

def motion_features(hist):
    """Input: (N,20,2) from observation window only. Units normalized image/fr."""
    if hist.ndim!=3 or hist.shape[1:]!=(20,2):raise ValueError('Require N×20×2')
    v=np.diff(hist,axis=1)
    step=np.linalg.norm(v,axis=2)
    path=step.sum(axis=1)
    disp=np.linalg.norm(hist[:,-1]-hist[:,0],axis=1)
    straightness=np.divide(disp,path,out=np.full_like(disp,np.nan),where=path>1e-12)
    accel=np.diff(v,axis=1)
    accel_rms=np.sqrt(np.mean(np.sum(accel**2,axis=2),axis=1))
    t=np.arange(20,dtype=float);tc=t-t.mean();sxx=np.sum(tc**2)
    velocity=np.einsum('t,ntc->nc',tc,hist)/sxx
    line=hist.mean(axis=1)[:,None,:]+tc[None,:,None]*velocity[:,None,:]
    residual_rms=np.sqrt(np.mean(np.sum((hist-line)**2,axis=-1),axis=1))
    speed=path/19
    # Turning is unstable at tiny steps; only compare adjacent displacements with
    # both magnitudes >= max(5e-5, 0.25*track median step).
    eps=np.maximum(5e-5,0.25*np.median(step,axis=1))
    valid=(step[:,:-1]>=eps[:,None])&(step[:,1:]>=eps[:,None])
    cross=v[:,:-1,0]*v[:,1:,1]-v[:,:-1,1]*v[:,1:,0]
    dot=np.sum(v[:,:-1,:]*v[:,1:,:],axis=-1)
    theta=np.abs(np.arctan2(cross,dot))
    cnt=valid.sum(axis=1)
    turning=np.divide((theta*valid).sum(axis=1),cnt,out=np.full(len(hist),np.nan),where=cnt>0)
    return {'path_norm':path,'mean_step_norm_per_frame':speed,'net_displacement_norm':disp,
            'straightness':straightness,'ols_line_residual_norm':residual_rms,
            'acceleration_rms_norm_per_frame2':accel_rms,'mean_abs_turn_rad':turning,
            'turn_pairs_used':cnt, 'ols_velocity_norm_per_frame':np.linalg.norm(velocity,axis=1)}

