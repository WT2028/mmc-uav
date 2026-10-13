#!/usr/bin/env python3
"""Render the supplied simulation histories; never execute a controller or plant.

Requires Gazebo Rendering 8, its Ogre2 plugin, FFmpeg, NumPy, pandas, SciPy and
Pillow. Run from the repository root. See videos/README.md for the source map.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
COLORS = ['blue', 'red', 'purple', 'green']
TITLES = {
    'coordinated_motion': ('Translation and yaw coordination', 'Calm air | 4 s transfer | 40 deg yaw'),
    'wind_pid': ('Crosswind | PID', '5 m/s wind | 6 s transfer | NDO off'),
    'wind_lqr': ('Crosswind | LQR', '5 m/s wind | 6 s transfer | NDO off'),
    'wind_cmpc': ('Crosswind | CMPC', '5 m/s wind | 6 s transfer | NDO off'),
    'wind_cmpc_ndo': ('Crosswind | CMPC + NDO', '5 m/s wind | 6 s transfer'),
    'pressure_cmpc': ('Actuator constraints | CMPC', 'Calm air | 1.5 m / 2.95 s transfer | NDO off'),
    'pressure_umpc': ('Actuator constraints | UMPC', 'Calm air | 1.5 m / 2.95 s transfer | NDO off'),
    'scan_nominal': ('Nominal mass and bandwidth', 'M = 1.45 kg | m = 0.05 kg | 20 rad/s'),
    'scan_mu_low': ('Mass ratio | low', 'Fixed total mass | mu = 0.0252 | NDO off'),
    'scan_mu_high': ('Mass ratio | high', 'Fixed total mass | mu = 0.0484 | NDO off'),
    'scan_M_low': ('Total mass | low', 'Fixed mass ratio | M = 1.39 kg | NDO off'),
    'scan_M_high': ('Total mass | high', 'Fixed mass ratio | M = 1.55 kg | NDO off'),
    'scan_omega10': ('Actuator bandwidth | 10 rad/s', 'Calm air | fixed effective controller weights | NDO off'),
    'scan_omega30': ('Actuator bandwidth | 30 rad/s', 'Calm air | fixed effective controller weights | NDO off'),
    'efficiency_k75': ('Rotor efficiency | 0.75', 'Controller coefficient = 0.85 | 5 m/s wind | NDO on'),
    'efficiency_k85': ('Rotor efficiency | 0.85', 'Controller coefficient = 0.85 | 5 m/s wind | NDO on'),
    'efficiency_k95': ('Rotor efficiency | 0.95', 'Controller coefficient = 0.85 | 5 m/s wind | NDO on'),
}


def read_frame(path, time):
    return pd.read_csv(path).sort_values(time).drop_duplicates(time, keep='last')


def interp(frame, time, cols, at):
    return np.column_stack([np.interp(at, frame[time], frame[c]) for c in cols])


def rotations(frame, time, cols, at, euler=False):
    values = frame[cols].to_numpy()
    r = Rotation.from_euler('xyz', values, degrees=True) if euler else Rotation.from_quat(values)
    return Slerp(frame[time].to_numpy(), r)(np.clip(at, frame[time].iloc[0], frame[time].iloc[-1]))


def ticks(lo, hi, fps, speed):
    return np.arange(math.ceil(lo * fps / speed), math.floor(hi * fps / speed) + 1) * speed / fps


def closed_loop(profile, repeat, fps, duration):
    config = json.loads((ROOT/'experiments/configs'/f'{profile}.json').read_text())
    index = json.loads((ROOT/'data/closed_loop/index.json').read_text())
    record = next(x for x in index if x['profile'] == profile and x['repeat'] == repeat)
    file = ROOT/record['flight']; frame = read_frame(file, 'Time')
    t = ticks(max(0., frame.Time.min()), min(duration or 70., frame.Time.max()), fps, 1.)
    pos = interp(frame, 'Time', ['X','Y','Z'], t)
    rot = rotations(frame, 'Time', ['Roll_deg','Pitch_deg','Yaw_deg'], t, True)
    rails = interp(frame, 'Time', ['Slider_Y_actual','Slider_Y_actual','Slider_X_actual','Slider_X_actual'], t)
    rails[:,2:] *= -1
    sources = [file, ROOT/config['model']]
    rail_basis = 'Measured pair displacements (X joint signs follow the URDF)'
    if 'servo' in record:
        servo = pd.read_csv(ROOT/record['servo'])
        offset = record.get('native_clock_offset_s')
        if offset is None:
            lo, hi = record['window_controller_s']
            selected = frame[(frame.Time >= lo) & (frame.Time <= hi)]
            offset = float(np.median(selected.Joint_feedback_clock_s - selected.Time))
        for i, color in enumerate(COLORS):
            joint = servo[servo.joint_name == 'joint_slider_' + color].sort_values('sim_time_s').drop_duplicates('sim_time_s')
            rails[:,i] = np.interp(t + offset, joint.sim_time_s, joint.position)
        sources.append(ROOT/record['servo']); rail_basis = 'Four independent recorded rail positions'
    angles = []
    # Rotor phase is display-only: integrate measured physical speed with the
    # same 10:1 visual slowdown as the model's Gazebo motor plugin.
    for column, direction in [('Upper_rotor_speed_actual',1), ('Lower_rotor_speed_actual',-1)]:
        phase = cumulative_trapezoid(frame[column].to_numpy()/10 * direction, frame.Time.to_numpy(), initial=0.)
        angles.append(np.interp(t, frame.Time, phase))
    if profile == 'coordinated_motion':
        raw = pd.read_csv(ROOT/'data/coordination/raw_joint_states.csv.gz')
        # Clock alignment is measured from the same recorded controller history.
        offset = float(np.median(frame.Joint_feedback_clock_s - frame.Time))
        for i, name in enumerate(['rotor_upper','rotor_lower']):
            joint=raw[raw.joint_name=='joint_'+name].sort_values('stamp_s').drop_duplicates('stamp_s')
            angles[i]=np.interp(t+offset,joint.stamp_s,joint.position)
        for i,color in enumerate(COLORS):
            joint=raw[raw.joint_name=='joint_slider_'+color].sort_values('stamp_s').drop_duplicates('stamp_s')
            rails[:,i]=np.interp(t+offset,joint.stamp_s,joint.position)
        sources.append(ROOT/'data/coordination/raw_joint_states.csv.gz'); rail_basis='Four recorded rails and recorded rotor joint angles'
    return dict(name=profile, title=TITLES[profile][0], subtitle=TITLES[profile][1], section=config['paper_section'],
                t=t, pos=pos, rot=rot, rails=rails, rotor=np.column_stack(angles), speed=1.,
                model=ROOT/config['model'], floor=0., sources=sources, repeat=repeat,
                reference=np.array([config['protocol']['target_A_m'],config['protocol']['target_B_m']]),
                rail_basis=rail_basis, rotor_basis='Measured angular speed integrated with 10:1 visual slowdown; absolute phase is illustrative' if profile!='coordinated_motion' else 'Recorded joint angles')


def actuator(case, fps, duration):
    folder=ROOT/'data/actuator'/case
    joints=read_frame(folder/'joint_raw.csv.gz','sim_time_s'); body=read_frame(folder/'odometry_raw.csv.gz','sim_time_s')
    # Full recorded single/pair/dual tests and full constrained command sequence.
    speed=.5; t=ticks(max(joints.sim_time_s.min(),body.sim_time_s.min()),min(duration or 100.,joints.sim_time_s.max(),body.sim_time_s.max()),fps,speed)
    return dict(name='actuator_'+case,title='Isolated actuator | '+case,subtitle='Recorded joint and free-body motion | half-speed playback',section='V.A',
                t=t,pos=interp(body,'sim_time_s',['base_x_m','base_y_m','base_z_m'],t),rot=rotations(body,'sim_time_s',['base_qx','base_qy','base_qz','base_qw'],t),
                rails=interp(joints,'sim_time_s',[c+'_position_m' for c in COLORS],t),rotor=interp(joints,'sim_time_s',['rotor_upper_position_rad','rotor_lower_position_rad'],t),
                speed=speed,model=folder/'source_urdf.urdf',floor=-.5,sources=[folder/'joint_raw.csv.gz',folder/'odometry_raw.csv.gz',folder/'source_urdf.urdf'],reference=None,
                rail_basis='Four recorded rail positions',rotor_basis='Recorded joint angles')


def multibody(case, fps, duration):
    folder=ROOT/'data/multibody'/case
    body=read_frame(folder/'raw_physical_body.csv','stamp_s'); joints=pd.read_csv(folder/'raw_joint_states.csv')
    # Initialization at the first sample, then the complete 0.35 s comparison.
    start=max(body.stamp_s.min(),joints.stamp_s.min()); speed=.05
    t=np.arange(round(.35/speed*fps)+1)*speed/fps+start
    if t[-1]>body.stamp_s.max()+1e-10: raise ValueError('Incomplete multibody history')
    values=[]
    for name in ['slider_'+c for c in COLORS]+['rotor_upper','rotor_lower']:
        joint=joints[joints.joint_name=='joint_'+name].sort_values('stamp_s').drop_duplicates('stamp_s')
        values.append(np.interp(t,joint.stamp_s,joint.position))
    return dict(name='multibody_'+case,title='Multibody consistency | '+case.removeprefix('map_').replace('_',' '),subtitle='0.35 s recorded test | 20x slow motion | fixed rotor joints',section='V.A',
                t=t,pos=interp(body,'stamp_s',['x','y','z'],t),rot=rotations(body,'stamp_s',['qx','qy','qz','qw'],t),rails=np.array(values[:4]).T,rotor=np.array(values[4:]).T,
                speed=speed,model=folder/'aligned_source.urdf',floor=8.5,sources=[folder/'raw_physical_body.csv',folder/'raw_joint_states.csv',folder/'aligned_source.urdf'],reference=None,
                rail_basis='Four recorded rail positions',rotor_basis='Recorded fixed rotor joint angles')


def prepare_scene(item, out, width, height):
    urdf=ET.parse(item['model']).getroot(); links=urdf.findall('link')
    joints={j.find('child').get('link'):j for j in urdf.findall('joint')}
    poses=[]; assets=[]; base=item['rot']; pos=item['pos']; n=len(pos)
    for link in links:
        name=link.get('name'); visual=link.find('visual'); origin=visual.find('origin')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' '); rv=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' '))
        if name in joints:
            joint=joints[name]; jorigin=joint.find('origin')
            qxyz=np.fromstring(jorigin.get('xyz','0 0 0'),sep=' '); qr=Rotation.from_euler('xyz',np.fromstring(jorigin.get('rpy','0 0 0'),sep=' '))
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            if joint.get('type')=='prismatic':
                color=name.split('_')[1]; q=item['rails'][:,COLORS.index(color)]
                local_pos=qxyz+qr.apply(q[:,None]*axis+xyz); rotation=base*qr*rv
            else:
                q=item['rotor'][:,0 if 'upper' in name else 1]; moving=Rotation.from_rotvec(q[:,None]*axis)
                local_pos=qxyz+qr.apply(moving.apply(xyz)); rotation=base*qr*moving*rv
        else:local_pos=np.broadcast_to(xyz,(n,3));rotation=base*rv
        world_pos=pos+base.apply(local_pos); quaternion=rotation.as_quat()[:,[3,0,1,2]]
        poses.append(np.column_stack([world_pos,quaternion]))
        mesh=visual.find('geometry/mesh').get('filename')
        mesh=ROOT/'src/mmc_uav_description/meshes'/Path(mesh).name
        rgb=np.fromstring(visual.find('material/color').get('rgba'),sep=' ')[:3]
        assets.append(str(mesh)+' '+' '.join(map(str,rgb)))
    offset=np.array([.95,-1.20,.85]); direction=-offset
    yaw=math.atan2(direction[1],direction[0]); pitch=math.atan2(-direction[2],np.hypot(*direction[:2]))
    cr=Rotation.from_euler('xyz',[0,pitch,yaw]).as_quat()[[3,0,1,2]]
    camera=np.column_stack([pos+offset,np.broadcast_to(cr,(n,4))])
    np.stack([camera,*poses],axis=1).astype('<f8').tofile(out/'poses.bin')
    (out/'scene.txt').write_text(f'{width} {height} {len(links)} {n} {item["floor"]}\n'+'\n'.join(assets)+'\n')
    return poses


def draw_map(item, index, size=(380,350)):
    im=Image.new('RGB',size,'#ffffff');d=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    pos=item['pos'][:,:2]; ref=item['reference']; data=np.vstack([pos,ref[:,:2]]) if ref is not None else pos
    center=(data.min(axis=0)+data.max(axis=0))/2; span=max(np.ptp(data,axis=0).max(),.2 if ref is None else 1.)*1.2
    # Same overview range for all wind/scan/efficiency cases; pressure includes the full UMPC excursion.
    if item['name'].startswith(('wind_','scan_','efficiency_')):center=np.array([0.,1.5]);span=4.2
    if item['name'].startswith('pressure_'):center=np.array([.7,0.]);span=16.
    scale=270/span
    def screen(p):return (int(size[0]/2+(p[0]-center[0])*scale),int(175-(p[1]-center[1])*scale))
    d.text((18,10),'XY',font=font,fill='#26364a')
    if ref is not None:
        a,b=screen(ref[0]);c,e=screen(ref[1]);d.line((a,b,c,e),fill='#a6aeb8',width=3)
        d.ellipse((a-6,b-6,a+6,b+6),fill='#66778b')
        d.line((c-8,e-8,c+8,e+8),fill='#ca684d',width=4);d.line((c-8,e+8,c+8,e-8),fill='#ca684d',width=4)
    path=[screen(p) for p in pos[:index+1:2]]
    if len(path)>1:d.line(path,fill='#197f87',width=4)
    x,y=screen(pos[index]);d.ellipse((x-8,y-8,x+8,y+8),fill='#126a72')
    if ref is not None:
        # Opposite sides keep the nearby pressure-test targets distinguishable.
        label_y=b-32 if a<160 and b>260 else b+8
        d.text((a-30,label_y),'A',font=font,fill='#66778b')
        d.text((c+12,e-32),'B',font=font,fill='#a65038')
    # A scale bar replaces the verbose viewport description; +X is right, +Y up.
    length=10**math.floor(math.log10(span/4))
    length*=next(v for v in [5,2,1] if v*length<=span/4)
    end=22+round(length*scale)
    d.line((22,size[1]-22,end,size[1]-22),fill='#66778b',width=3)
    for x in [22,end]:d.line((x,size[1]-28,x,size[1]-16),fill='#66778b',width=3)
    d.text((end+12,size[1]-39),f'{length:g} m',font=font,fill='#66778b')
    return im


def annotate(im, item, index, attitude, fonts):
    d=ImageDraw.Draw(im); color='#20334a'
    d.rectangle((0,0,1280,82),fill='#f8fafc')
    title=item['title'].replace('Translation and yaw coordination','Translation + yaw').replace('Multibody consistency','Multibody').replace('Isolated actuator','Actuator')
    d.text((30,21),title,font=fonts[0],fill=color)
    time=f't = {item["t"][index]:.2f} s'
    if item['speed']!=1.:time+=f'  |  {item["speed"]:g}×'
    d.text((1250,24),time,font=fonts[1],fill=color,anchor='ra')
    im.paste(draw_map(item,index),(876,280))
    d=ImageDraw.Draw(im)
    d.rectangle((0,642,1280,720),fill='#f8fafc')
    def values(array):return '  '.join(f'{0. if abs(v)<.005 else v:.2f}' for v in array)
    d.text((30,663),'XYZ (m)  '+values(item['pos'][index]),font=fonts[1],fill=color)
    d.text((654,663),'RPY (°)  '+values(attitude[index]),font=fonts[1],fill=color)
    return im


def render(item, args, binary):
    width,height=1280,720; out=args.work/item['name'];out.mkdir(parents=True,exist_ok=True)
    destination=args.output/(item['name']+'.mp4');destination.parent.mkdir(parents=True,exist_ok=True)
    prepare_scene(item,out,width,height)
    fonts=[ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',36),
           ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',32)]
    readfd,writefd=os.pipe()
    with (out/'render.log').open('wb') as log:
        renderer=subprocess.Popen([str(binary),str(out/'scene.txt'),str(out/'poses.bin'),str(writefd)],pass_fds=[writefd],stdout=log,stderr=log)
        os.close(writefd)
        ffmpeg=subprocess.Popen(['ffmpeg','-y','-v','error','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(args.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],stdin=subprocess.PIPE,stdout=log,stderr=log)
        try:
            with os.fdopen(readfd,'rb') as stream:
                attitude=item['rot'].as_euler('xyz',degrees=True)
                for i,t in enumerate(item['t']):
                    raw=stream.read(width*height*3)
                    if len(raw)!=width*height*3:raise RuntimeError(f'Renderer stopped at frame {i}; see {out}/render.log')
                    im=annotate(Image.frombytes('RGB',(width,height),raw),item,i,attitude,fonts)
                    ffmpeg.stdin.write(im.tobytes())
                    if i==min(len(item['t'])-1,int(10*args.fps/item['speed'])):im.save(out/'sample.png')
            ffmpeg.stdin.close()
            if ffmpeg.wait()!=0 or renderer.wait()!=0:raise RuntimeError('Rendering/encoding failed')
        finally:
            for process in [renderer,ffmpeg]:
                if process.poll() is None:process.terminate();process.wait()
    meta={k:item[k] for k in ['name','title','subtitle','section','speed','rail_basis','rotor_basis']}
    meta.update(width=width,height=height,fps=args.fps,frames=len(item['t']),time_range_s=[float(item['t'][0]),float(item['t'][-1])],repeat=item.get('repeat'),
                playback='Gazebo Rendering 8 / Ogre2 visualization of recorded measurements; no physics rerun',
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in item['sources']},video_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'video':str(destination),'frames':len(item['t'])}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles',nargs='*',help='Profile names, actuator_CASE, or multibody_CASE')
    parser.add_argument('--all',action='store_true');parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--fps',type=int,default=20);parser.add_argument('--duration',type=float)
    parser.add_argument('--output',type=Path,default=ROOT/'results/videos')
    parser.add_argument('--work',type=Path,default=ROOT/'results/video_render')
    args=parser.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    if args.fps<1:parser.error('fps must be positive')
    binary=args.work/'render'
    flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','gz-rendering8','gz-common5-graphics'],text=True))
    subprocess.run(['g++','-O2','-o',str(binary),str(Path(__file__).with_name('render.cc')),*flags],check=True)
    names=list(TITLES) if args.all else args.profiles
    if not names:parser.error('Select profiles or --all')
    for name in names:
        if name.startswith('actuator_'):item=actuator(name.removeprefix('actuator_'),args.fps,args.duration)
        elif name.startswith('multibody_'):item=multibody(name.removeprefix('multibody_'),args.fps,args.duration)
        else:item=closed_loop(name,args.repeat,args.fps,args.duration)
        render(item,args,binary)


if __name__=='__main__':main()
