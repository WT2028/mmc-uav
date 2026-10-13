"""Prepare a launch-local actuator configuration without editing source geometry."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare_actuator_model(source, destination, wn, zeta, velocity_limit, command_limit):
    """Share servo parameters with MPC; retain masses, inertias and rail stops."""
    values = (wn, zeta, velocity_limit, command_limit)
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
        raise ValueError('Actuator parameters must be finite and positive')
    if wn < 1.0 or zeta < 0.05:
        raise ValueError('Actuator bandwidth/damping outside supported controller range')
    if command_limit < 1e-4:
        raise ValueError('Command limit below supported MPC minimum of 1e-4 m')
    tree = ET.parse(source)
    root = tree.getroot()
    joints = {joint.attrib['name']: joint for joint in root.findall('joint')}
    links = {link.attrib['name']: link for link in root.findall('link')}
    plugins = [p for p in root.findall('gazebo/plugin') if p.attrib.get('name') == 'mmc::SliderServo']
    if len(plugins) != 4:
        raise ValueError('Model must contain four physical SliderServo plugins')
    for plugin in plugins:
        joint = joints[plugin.findtext('joint_name')]
        limit = joint.find('limit')
        if limit is None:
            raise ValueError('Slider rail limits missing')
        lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
        if not lower < -command_limit < command_limit < upper:
            raise ValueError('Command limit must remain inside mechanical rail stops')
        mass = float(links[joint.find('child').attrib['link']].find('inertial/mass').attrib['value'])
        updates = {'wn': wn, 'zeta': zeta, 'moving_mass': mass,
                   'velocity_limit': velocity_limit, 'command_limit': command_limit,
                   'mechanical_limit': min(-lower, upper), 'effort_limit': float(limit.attrib['effort'])}
        for key, value in updates.items():
            element = plugin.find(key)
            if element is None:
                element = ET.SubElement(plugin, key)
            element.text = str(float(value))
        limit.set('velocity', str(float(velocity_limit)))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    return str(destination)
