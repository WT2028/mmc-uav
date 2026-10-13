#!/usr/bin/env python3
"""Passive ROS/Gazebo sensor capture for an experiment's isolated partition."""
import csv
import gzip
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Vector3
from gz.transport13 import Node as GzNode
from gz.msgs10.double_v_pb2 import Double_V


def main():
    out = Path(sys.argv[1])
    handles, writers, counts, errors = {}, {}, {}, []
    lock = threading.RLock()
    stopped = threading.Event()
    def writer(name, fields):
        handles[name] = gzip.open(out/(name+'.csv.gz'), 'wt', newline='', compresslevel=1)
        writers[name] = csv.writer(handles[name]); writers[name].writerow(fields)
        counts[name] = 0
    def save(name, row):
        with lock:
            writers[name].writerow(row); counts[name] += 1
    def stamp(value):
        return value.sec + value.nanosec*1e-9
    writer('raw_servo_state', ['sim_time_s','joint_name','position','velocity','requested','target',
                               'force_N','target_acceleration','load_acceleration'])
    writer('raw_joint_states', ['sim_time_s','stamp_s','joint_name','position','velocity','effort'])
    writer('raw_odometry', ['sim_time_s','stamp_s','frame_id','child_frame_id',
                            'x','y','z','qx','qy','qz','qw','vx','vy','vz','p','q','r'])
    writer('raw_imu', ['sim_time_s','stamp_s','qx','qy','qz','qw','p','q','r','ax','ay','az'])
    writer('raw_wind', ['sim_time_s','vx','vy','vz'])
    rclpy.init()
    node, gz_node = rclpy.create_node('mmc_sensor_recorder'), GzNode()
    clock, callbacks, topics = [0.], [], []
    def servo_callback(color):
        def callback(message):
            values = list(message.data)
            if len(values) != 8 or not all(math.isfinite(v) for v in values):
                errors.append('Invalid servo telemetry: '+color)
                return
            save('raw_servo_state', [values[0],'joint_slider_'+color,*values[1:]])
        return callback
    for color in ('blue','red','green','purple'):
        callback = servo_callback(color)
        topic = '/model/mmc_uav/joint/joint_slider_'+color+'/cmd_pos/servo_state'
        if not gz_node.subscribe(Double_V,topic,callback):
            raise RuntimeError('Cannot subscribe '+topic)
        callbacks.append(callback); topics.append(topic)
    def clock_cb(msg):
        clock[0] = stamp(msg.clock)
    def joints_cb(msg):
        for i,name in enumerate(msg.name):
            save('raw_joint_states',[clock[0],stamp(msg.header.stamp),name,
                 msg.position[i] if i<len(msg.position) else math.nan,
                 msg.velocity[i] if i<len(msg.velocity) else math.nan,
                 msg.effort[i] if i<len(msg.effort) else ''])
    def odometry_cb(msg):
        p,q,v,w=msg.pose.pose.position,msg.pose.pose.orientation,msg.twist.twist.linear,msg.twist.twist.angular
        save('raw_odometry',[clock[0],stamp(msg.header.stamp),msg.header.frame_id,msg.child_frame_id,
                             p.x,p.y,p.z,q.x,q.y,q.z,q.w,v.x,v.y,v.z,w.x,w.y,w.z])
    def imu_cb(msg):
        q,w,a=msg.orientation,msg.angular_velocity,msg.linear_acceleration
        save('raw_imu',[clock[0],stamp(msg.header.stamp),q.x,q.y,q.z,q.w,w.x,w.y,w.z,a.x,a.y,a.z])
    def wind_cb(msg):
        save('raw_wind',[clock[0],msg.x,msg.y,msg.z])
    subscriptions=[node.create_subscription(kind,topic,callback,qos_profile_sensor_data)
                   for kind,topic,callback in [(Clock,'/clock',clock_cb),
                       (JointState,'/joint_states',joints_cb),(Odometry,'/model/mmc_uav/odometry',odometry_cb),
                       (Imu,'/imu',imu_cb),(Vector3,'/mmc/wind/velocity',wind_cb)]]
    signal.signal(signal.SIGINT,lambda *_:stopped.set())
    signal.signal(signal.SIGTERM,lambda *_:stopped.set())
    (out/'recorder_ready.json').write_text(json.dumps({'ready':True})+'\n')
    try:
        while rclpy.ok() and not stopped.is_set():
            rclpy.spin_once(node,timeout_sec=.1)
    finally:
        for topic in topics:
            gz_node.unsubscribe(topic)
        with lock:
            for handle in handles.values():handle.close()
        node.destroy_node(); rclpy.try_shutdown()
        (out/'sensor_capture.json').write_text(json.dumps({'counts':counts,'errors':errors,
                                                          'last_sim_time_s':clock[0]},indent=2)+'\n')


if __name__=='__main__':main()
