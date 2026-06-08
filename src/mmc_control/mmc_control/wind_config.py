import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class WindConfigSummary:
    world_name: str = ""
    config_valid: bool = False
    enable_wind: bool = False
    wind_vx_world: float = 0.0
    wind_vy_world: float = 0.0
    wind_vz_world: float = 0.0
    wind_speed_world: float = 0.0
    wind_force_scaling: float = 0.0
    wind_mag_rise_time: float = 0.0
    wind_mag_sin_amp_pct: float = 0.0
    wind_mag_noise_stddev: float = 0.0
    wind_dir_rise_time: float = 0.0
    wind_dir_sin_amp_deg: float = 0.0
    wind_dir_noise_stddev: float = 0.0
    wind_vertical_noise_stddev: float = 0.0
    activation_mode: str = "immediate"
    hover_target_z: float = 1.5
    hover_z_tol: float = 0.15
    hover_speed_tol: float = 0.15
    hover_hold_time: float = 0.0

    def as_log_row(self):
        return [
            float(self.config_valid),
            float(self.enable_wind),
            self.wind_vx_world,
            self.wind_vy_world,
            self.wind_vz_world,
            self.wind_speed_world,
            self.wind_force_scaling,
            self.wind_mag_rise_time,
            self.wind_mag_sin_amp_pct,
            self.wind_mag_noise_stddev,
            self.wind_dir_rise_time,
            self.wind_dir_sin_amp_deg,
            self.wind_dir_noise_stddev,
            self.wind_vertical_noise_stddev,
            self.activation_mode,
            self.hover_target_z,
            self.hover_z_tol,
            self.hover_speed_tol,
            self.hover_hold_time,
        ]


def _zero_wind_config(valid: bool = False) -> WindConfigSummary:
    return WindConfigSummary(config_valid=valid, enable_wind=False)


def _parse_float(element: Optional[ET.Element], path: str, default: float = 0.0) -> float:
    if element is None:
        return default
    target = element.find(path)
    if target is None or target.text is None:
        return default
    return float(target.text.strip())


def _parse_text(element: Optional[ET.Element], path: str, default: str = "") -> str:
    if element is None:
        return default
    target = element.find(path)
    if target is None or target.text is None:
        return default
    return target.text.strip()


def _parse_vector3(text: str) -> tuple[float, float, float]:
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got: {text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _find_world(root: ET.Element) -> Optional[ET.Element]:
    if root.tag == "world":
        return root
    return root.find("world")


def _find_wind_plugin(world: ET.Element) -> Optional[ET.Element]:
    for plugin in world.findall("plugin"):
        filename = (plugin.get("filename") or "").strip()
        name = (plugin.get("name") or "").strip()
        if name == "gz::sim::systems::WindEffects" or "wind-effects-system" in filename:
            return plugin
    return None


def parse_world_wind_config(world_sdf_path: str | Path | None) -> WindConfigSummary:
    if not world_sdf_path:
        return _zero_wind_config(valid=False)

    try:
        path = Path(world_sdf_path).expanduser()
    except TypeError:
        return _zero_wind_config(valid=False)

    if not path.is_file():
        return _zero_wind_config(valid=False)

    try:
        root = ET.parse(path).getroot()
        world = _find_world(root)
        if world is None:
            return _zero_wind_config(valid=False)

        world_name = (world.get("name") or "").strip()
        wind = world.find("wind")
        initial_vx = initial_vy = initial_vz = 0.0
        if wind is not None:
            linear_velocity = wind.find("linear_velocity")
            if linear_velocity is not None and linear_velocity.text:
                initial_vx, initial_vy, initial_vz = _parse_vector3(linear_velocity.text.strip())

        plugin = _find_wind_plugin(world)
        force_scaling = _parse_float(plugin, "force_approximation_scaling_factor", 0.0)
        mag_rise_time = _parse_float(plugin, "horizontal/magnitude/time_for_rise", 0.0)
        mag_sin_amp_pct = _parse_float(plugin, "horizontal/magnitude/sin/amplitude_percent", 0.0)
        mag_noise_stddev = _parse_float(plugin, "horizontal/magnitude/noise/stddev", 0.0)
        dir_rise_time = _parse_float(plugin, "horizontal/direction/time_for_rise", 0.0)
        dir_sin_amp_deg = _parse_float(plugin, "horizontal/direction/sin/amplitude", 0.0)
        dir_noise_stddev = _parse_float(plugin, "horizontal/direction/noise/stddev", 0.0)
        vertical_noise_stddev = _parse_float(plugin, "vertical/noise/stddev", 0.0)

        schedule = plugin.find("mmc_wind_schedule") if plugin is not None else None
        activation_mode = (_parse_text(schedule, "activation_mode", "immediate") or "immediate").strip()
        target_velocity_text = _parse_text(schedule, "target_linear_velocity", "")
        vx, vy, vz = initial_vx, initial_vy, initial_vz
        if target_velocity_text:
            vx, vy, vz = _parse_vector3(target_velocity_text)

        default_hold_time = 3.0 if activation_mode == "hover_hold" else 0.0
        hover_target_z = _parse_float(schedule, "hover_target_z", 1.5)
        hover_z_tol = max(0.0, _parse_float(schedule, "hover_z_tol", 0.15))
        hover_speed_tol = max(0.0, _parse_float(schedule, "hover_speed_tol", 0.15))
        hover_hold_time = max(0.0, _parse_float(schedule, "hover_hold_time", default_hold_time))

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        return WindConfigSummary(
            world_name=world_name,
            config_valid=True,
            enable_wind=speed > 1e-9,
            wind_vx_world=vx,
            wind_vy_world=vy,
            wind_vz_world=vz,
            wind_speed_world=speed,
            wind_force_scaling=force_scaling,
            wind_mag_rise_time=mag_rise_time,
            wind_mag_sin_amp_pct=mag_sin_amp_pct,
            wind_mag_noise_stddev=mag_noise_stddev,
            wind_dir_rise_time=dir_rise_time,
            wind_dir_sin_amp_deg=dir_sin_amp_deg,
            wind_dir_noise_stddev=dir_noise_stddev,
            wind_vertical_noise_stddev=vertical_noise_stddev,
            activation_mode=activation_mode,
            hover_target_z=hover_target_z,
            hover_z_tol=hover_z_tol,
            hover_speed_tol=hover_speed_tol,
            hover_hold_time=hover_hold_time,
        )
    except (ET.ParseError, OSError, ValueError):
        return _zero_wind_config(valid=False)
