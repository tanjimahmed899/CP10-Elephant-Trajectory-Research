#!/usr/bin/env python3
"""Render source JSON box annotations on supplied source JPEGs (does not modify source)."""
from pathlib import Path
import argparse,json,zipfile,re
from PIL import Image,ImageDraw,ImageFont

P=re.compile(r'^frame_(\d+)\.(?:jpg|jpeg|png)$',re.I)
COLORS={"0":"#F9B233","1":"#31AFE8","2":"#EF5795","3":"#61BD75"}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--zip',required=True,type=Path);a.add_argument('--frames-dir',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
    args=a.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted([p for p in args.frames_dir.iterdir() if P.match(p.name)],key=lambda p:int(P.match(p.name).group(1)))
    if not files: raise SystemExit('No frame_N.jpg found in --frames-dir')
    with zipfile.ZipFile(args.zip) as z:
        for p in files:
            idx=int(P.match(p.name).group(1));jsons=[v for v in z.namelist() if v.endswith(f'/frame_{idx}.json') or v==f'frame_{idx}.json']
            if len(jsons)!=1:
                print(f'SKIP {p.name}: expected exactly one JSON matching frame index, got {len(jsons)}');continue
            j=json.loads(z.read(jsons[0]));im=Image.open(p).convert('RGB');sx=im.width/j['imageWidth'];sy=im.height/j['imageHeight']
            draw=ImageDraw.Draw(im);width=max(3,round(im.width/650))
            for s in j['shapes']:
                if str(s.get('label','')).lower()!='elephant':continue
                tid=str(s['group_id']);x=[k[0]*sx for k in s['points']];y=[k[1]*sy for k in s['points']]
                bb=[min(x),min(y),max(x),max(y)];color=COLORS.get(tid,'#FFFFFF')
                draw.rectangle(bb,outline=color,width=width)
                text=f'ID {tid}'
                font=ImageFont.load_default()
                px,py=bb[0],max(0,bb[1]-18)
                draw.rectangle((px,py,px+len(text)*8+8,py+17),fill=color)
                draw.text((px+4,py+2),text,fill='black',font=font)
            out=args.output/p.name
            im.save(out,quality=94)
            print(f'{p.name}: {im.width}x{im.height} scale=({sx:.6f},{sy:.6f}) -> {out}')

if __name__=='__main__':main()
