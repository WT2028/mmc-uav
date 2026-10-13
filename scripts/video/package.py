#!/usr/bin/env python3
"""Package recorded-motion clips, source manifests and short GIF previews."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
ACTUATORS=['actuator_'+x for x in ['single','pair','dual','constraints']]
MULTIBODY=['multibody_'+x['case'] for x in json.loads((ROOT/'data/multibody/index.json').read_text())]


def run(*args):
    subprocess.run(list(map(str,args)),check=True)


def concatenate(names, clips, output, work):
    chapters=[]; elapsed=0.
    for name in names:
        meta=json.loads((work/name/'metadata.json').read_text())
        duration=meta['frames']/meta['fps']
        chapters.append({'name':name,'start_video_s':elapsed,'duration_s':duration,'record':meta})
        elapsed+=duration
    listing=work/(output.stem+'_concat.txt')
    # FFmpeg concat file syntax: escape quote and backslash in file paths.
    def escape(p):return str(p.resolve()).replace("'", "'\\''")
    listing.write_text(''.join("file '"+escape(clips/(name+'.mp4'))+"'\n" for name in names))
    run('ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',listing,'-c','copy','-movflags','+faststart',output)
    return {'file':output.name,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'chapters':chapters}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips',type=Path,default=ROOT/'results/videos',help='Completed closed-loop videos')
    parser.add_argument('--work',type=Path,default=ROOT/'results/video_render')
    parser.add_argument('--output',type=Path,default=ROOT/'results/video_package')
    parser.add_argument('--previews',type=Path)
    parser.add_argument('--skip-render',action='store_true',help='Use previously completed isolated clips')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    parts=args.work/'isolated_clips';parts.mkdir(parents=True,exist_ok=True)
    if not args.skip_render:
        run(sys.executable,Path(__file__).with_name('record.py'),*ACTUATORS,*MULTIBODY,'--output',parts,'--work',args.work)
    videos=[]
    profiles=json.loads((ROOT/'experiments/index.json').read_text())
    for row in profiles:
        name=row['name'];p=args.clips/(name+'.mp4')
        meta=json.loads((args.work/name/'metadata.json').read_text())
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if sha!=meta['video_sha256']:raise ValueError('Clip checksum mismatch: '+name)
        # Keep all full-flight clips together when using a separate output folder.
        target=args.output/p.name
        if target.resolve()!=p.resolve():
            import shutil
            shutil.copyfile(p,target)
        videos.append({'file':p.name,'sha256':sha,'record':meta})
    for name,group in [('isolated_actuators',ACTUATORS),('multibody_checks',MULTIBODY)]:
        videos.append(concatenate(group,parts,args.output/(name+'.mp4'),args.work))
    previews=args.previews or args.output/'previews';previews.mkdir(parents=True,exist_ok=True)
    excerpts={'coordinated_motion':(7.,12.),'wind_cmpc_ndo':(7.,12.),'pressure_cmpc':(7.,12.),'pressure_umpc':(7.,12.),'isolated_actuators':(39.,10.),'multibody_checks':(7.05,7.)}
    preview_records=[]
    for name,(start,duration) in excerpts.items():
        p=previews/(name+'.gif')
        run('ffmpeg','-y','-v','error','-threads','2','-ss',start,'-t',duration,'-i',args.output/(name+'.mp4'),'-filter_complex_threads','1','-vf','fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3','-loop','0',p)
        preview_records.append({'file':p.name,'source_video':name+'.mp4','start_video_s':start,'duration_s':duration,'fps':8,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'format_version':1,'kind':'Recorded simulation playback','renderer':'Gazebo Rendering 8 / Ogre2',
              'closed_loop_selection':'All 17 profiles, recorded repetition 1; complete controller history through 70 s',
              'videos':videos,'previews':preview_records}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'videos':len(videos),'previews':len(preview_records),'manifest':str(args.output/'manifest.json')}))


if __name__=='__main__':main()
