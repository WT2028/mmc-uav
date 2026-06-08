import os
import time

# Gazebo Harmonic's Python bindings on this workstation use generated
# protobuf modules that are incompatible with the newer Python protobuf C++
# runtime pulled in from the user site-packages.  The pure-Python runtime is
# slower but perfectly adequate for the low-rate wind command bridge, and it
# keeps M4 hover-hold wind activation reproducible without changing global
# Python packages.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gz.transport13
import rclpy
from geometry_msgs.msg import Vector3
from gz.msgs10 import wind_pb2
from mmc_interfaces.msg import WindCommand, WindStatus
from rclpy.node import Node

GZ_CONNECTION_TIMEOUT_SEC = 1.0
GZ_CONNECTION_POLL_SEC = 0.05


def gz_wind_topic_for_world(world_name: str) -> str:
    return f"/world/{str(world_name).strip()}/wind/"


def _copy_vector3(source, target) -> None:
    target.x = float(source.x)
    target.y = float(source.y)
    target.z = float(source.z)


def build_wind_pb_message(command: WindCommand) -> wind_pb2.Wind:
    message = wind_pb2.Wind()
    _copy_vector3(command.linear_velocity_world, message.linear_velocity)
    message.enable_wind = bool(command.enable_wind)
    return message


def build_status_message(command: WindCommand, publish_ok: bool, detail: str) -> WindStatus:
    status = WindStatus()
    status.command_seq = command.command_seq
    status.stamp = command.stamp
    status.world_name = command.world_name
    status.publish_ok = bool(publish_ok)
    status.wind_active = bool(publish_ok and command.enable_wind)
    if not hasattr(status, "linear_velocity_world") or status.linear_velocity_world is None:
        status.linear_velocity_world = Vector3()
    _copy_vector3(command.linear_velocity_world, status.linear_velocity_world)
    status.source = command.source
    status.detail = detail
    return status


def wait_for_gz_connections(
    publisher,
    *,
    timeout_sec: float = GZ_CONNECTION_TIMEOUT_SEC,
    poll_sec: float = GZ_CONNECTION_POLL_SEC,
) -> bool:
    has_connections = getattr(publisher, "has_connections", None)
    if not callable(has_connections):
        return True

    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        if has_connections():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.0, float(poll_sec)))


class WindBridgeNode(Node):
    def __init__(self, gz_node=None):
        super().__init__("wind_bridge_node")
        self.gz_node = gz_node or gz.transport13.Node()
        self._publisher_cache = {}
        self.command_sub = self.create_subscription(
            WindCommand,
            "/mmc/wind/command",
            self._on_command,
            10,
        )
        self.status_pub = self.create_publisher(WindStatus, "/mmc/wind/status", 10)

    def _publisher_for_world(self, world_name: str):
        publisher = self._publisher_cache.get(world_name)
        if publisher is None:
            publisher = self.gz_node.advertise(gz_wind_topic_for_world(world_name), wind_pb2.Wind)
            self._publisher_cache[world_name] = publisher
        return publisher

    def _on_command(self, command: WindCommand) -> None:
        world_name = str(command.world_name).strip()
        if not world_name:
            self.status_pub.publish(
                build_status_message(command, publish_ok=False, detail="world_name is required")
            )
            return

        try:
            publisher = self._publisher_for_world(world_name)
            topic = gz_wind_topic_for_world(world_name)
            if not wait_for_gz_connections(publisher):
                publish_ok = False
                detail = f"no Gazebo wind subscribers connected on {topic}"
            else:
                publish_ok = bool(publisher.publish(build_wind_pb_message(command)))
                detail = "published wind command" if publish_ok else "publisher returned False"
        except Exception as exc:  # pragma: no cover - exercised through integration paths
            publish_ok = False
            detail = f"failed to publish wind command: {exc}"

        self.status_pub.publish(build_status_message(command, publish_ok=publish_ok, detail=detail))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WindBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, "destroy_node"):
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass
