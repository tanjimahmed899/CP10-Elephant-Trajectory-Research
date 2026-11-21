import tempfile,subprocess,sys,pathlib,pandas as pd,json
root=pathlib.Path(__file__).resolve().parent.parent
lock=root/'data/EXP3_new_video_holdout_LOCKED.csv'
with tempfile.TemporaryDirectory() as tmp:
 p=pathlib.Path(tmp); lk=pd.read_csv(lock)
 rows=[]
 for t in ('2s_to_2s','2s_to_5s'):
  for i,x in lk.iterrows():
   rows.append({'task':t,'source_group':x.basename.removesuffix('.zip'),'samples':0 if i==0 else 10,'ADE':float('nan') if i==0 else .02,'FDE':float('nan') if i==0 else .05})
 pd.DataFrame(rows).to_csv(p/'fresh_holdout_per_source.csv',index=False)
 subprocess.run([sys.executable,str(root/'scripts/exp3_postprocess.py'),'--results',str(p),'--lock',str(lock)],check=True)
 table=pd.read_csv(p/'EXP3_ALL_LOCKED_VIDEO_ACCOUNTING.csv')
 assert len(table)==12 and table.status.value_counts().get('ZERO_WINDOWS',0)==2
 assert table.loc[table.status=='ZERO_WINDOWS','ADE'].isna().all()
 report=json.loads((p/'EXP3_SOURCE_COMPLETENESS_STATUS.json').read_text())
 assert not report['all_six_videos_with_windows_per_horizon']
 print('PASS synthetic zero-window accounting test. THESE ARE NOT RESEARCH RESULTS.')
