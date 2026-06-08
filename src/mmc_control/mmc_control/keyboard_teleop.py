import os
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - depends on system package
    tk = None

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:  # pragma: no cover - optional runtime fallback
    pynput_keyboard = None


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("mmc_keyboard_teleop")

        self.manual_xy_topic = self.declare_parameter("manual_xy_topic", "/mmc/manual_xy_cmd").value
        self.input_backend = str(self.declare_parameter("input_backend", "auto").value).lower()
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 30.0).value)

        self.pub_cmd = self.create_publisher(Twist, self.manual_xy_topic, 30)

        self._lock = threading.Lock()
        self._pressed_keys = set()
        self._listener = None
        self._backend = "none"
        self._running = True
        self._root = None
        self._status_var = None

        self._start_input_backend()

        if self._backend != "tk":
            self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._publish_current_command)

        self.get_logger().info(
            "Keyboard teleop node started. Use arrow keys to command horizontal motion after hover unlock."
        )

    @property
    def uses_tk(self) -> bool:
        return self._backend == "tk"

    def _start_input_backend(self):
        if self.input_backend in ("auto", "tk"):
            if tk is not None and os.environ.get("DISPLAY"):
                self._start_tk_backend()
                return
            if self.input_backend == "tk":
                self.get_logger().error("Requested tkinter backend, but Tk/Display is unavailable.")
                return

        if self.input_backend in ("auto", "pynput") and pynput_keyboard is not None:
            try:
                self._listener = pynput_keyboard.Listener(
                    on_press=self._on_pynput_press,
                    on_release=self._on_pynput_release,
                )
                self._listener.start()
                self._backend = "pynput"
                self.get_logger().warning(
                    "Keyboard teleop backend: pynput fallback. If key events are unstable, switch back to tkinter."
                )
                return
            except Exception as exc:  # pragma: no cover - runtime environment dependent
                self.get_logger().warning(f"Failed to start pynput keyboard backend: {exc}")

        self._backend = "none"
        self.get_logger().error("No usable keyboard backend is available for the teleop node.")

    def _start_tk_backend(self):
        self._root = tk.Tk()
        self._root.title("MMC Keyboard Teleop")
        self._root.geometry("420x170")
        self._root.resizable(False, False)

        title = tk.Label(
            self._root,
            text="MMC Keyboard Teleop",
            font=("Ubuntu", 14, "bold"),
            pady=10,
        )
        title.pack()

        info = tk.Label(
            self._root,
            text="Focus this window, then use the arrow keys.\nW A S D are also supported.",
            justify="center",
        )
        info.pack()

        self._status_var = tk.StringVar(value="Active keys: none")
        status = tk.Label(self._root, textvariable=self._status_var, pady=12)
        status.pack()

        tip = tk.Label(
            self._root,
            text="Close this window to stop keyboard teleop.",
            fg="#666666",
        )
        tip.pack()

        self._root.bind("<KeyPress>", self._on_tk_press)
        self._root.bind("<KeyRelease>", self._on_tk_release)
        self._root.bind("<FocusOut>", self._on_tk_focus_out)
        self._root.protocol("WM_DELETE_WINDOW", self._on_tk_close)
        self._root.after(200, self._focus_tk_window)
        self._backend = "tk"
        self.get_logger().info("Keyboard teleop backend: tkinter key window.")

    def _focus_tk_window(self):
        if self._root is None:
            return
        try:
            self._root.focus_force()
        except Exception:
            pass

    @staticmethod
    def _normalize_key_name(name):
        if not name:
            return None

        mapping = {
            "Up": "up",
            "Down": "down",
            "Left": "left",
            "Right": "right",
            "w": "up",
            "W": "up",
            "s": "down",
            "S": "down",
            "a": "left",
            "A": "left",
            "d": "right",
            "D": "right",
        }
        return mapping.get(name)

    def _update_status_label(self):
        if self._status_var is None:
            return
        with self._lock:
            keys = sorted(self._pressed_keys)
        text = "Active keys: none" if not keys else f"Active keys: {', '.join(keys)}"
        self._status_var.set(text)

    def _on_tk_press(self, event):
        key_name = self._normalize_key_name(event.keysym)
        if key_name is None:
            return
        with self._lock:
            self._pressed_keys.add(key_name)
        self._update_status_label()

    def _on_tk_release(self, event):
        key_name = self._normalize_key_name(event.keysym)
        if key_name is None:
            return
        with self._lock:
            self._pressed_keys.discard(key_name)
        self._update_status_label()

    def _on_tk_focus_out(self, _event):
        with self._lock:
            self._pressed_keys.clear()
        self._update_status_label()

    def _on_tk_close(self):
        self._running = False
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def _on_pynput_press(self, key):
        key_name = self._normalize_pynput_key(key)
        if key_name is None:
            return
        with self._lock:
            self._pressed_keys.add(key_name)

    def _on_pynput_release(self, key):
        key_name = self._normalize_pynput_key(key)
        if key_name is None:
            return
        with self._lock:
            self._pressed_keys.discard(key_name)

    @staticmethod
    def _normalize_pynput_key(key):
        if pynput_keyboard is None:
            return None

        arrow_map = {
            pynput_keyboard.Key.up: "up",
            pynput_keyboard.Key.down: "down",
            pynput_keyboard.Key.left: "left",
            pynput_keyboard.Key.right: "right",
        }
        if key in arrow_map:
            return arrow_map[key]

        char = getattr(key, "char", None)
        return KeyboardTeleopNode._normalize_key_name(char)

    def _active_keys_snapshot(self):
        with self._lock:
            return set(self._pressed_keys)

    def _publish_current_command(self):
        active_keys = self._active_keys_snapshot()

        forward = 0.0
        lateral = 0.0
        if "up" in active_keys:
            forward += 1.0
        if "down" in active_keys:
            forward -= 1.0
        if "right" in active_keys:
            lateral += 1.0
        if "left" in active_keys:
            lateral -= 1.0

        msg = Twist()
        msg.linear.x = forward
        msg.linear.y = lateral
        self.pub_cmd.publish(msg)

    def run_tk_loop(self):
        if self._root is None:
            return

        period_ms = max(10, int(1000.0 / max(self.publish_rate_hz, 1.0)))

        def tick():
            if not self._running or self._root is None:
                return
            self._publish_current_command()
            self._root.after(period_ms, tick)

        self._root.after(period_ms, tick)
        self._root.mainloop()

    def destroy_node(self):
        self._running = False

        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        if node.uses_tk:
            node.run_tk_loop()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
