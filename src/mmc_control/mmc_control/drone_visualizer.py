#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机数据可视化脚本
用于处理CSV格式的无人机数据并生成可视化图表（折线图）

作者: AI Assistant
版本: 2.1.0
依赖: pandas, matplotlib
"""

import argparse
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from mmc_control.project_paths import (
        DEFAULT_LOG_PATTERN,
        get_default_output_dir,
        get_default_log_dir,
        list_csv_files,
        resolve_package_path,
        select_latest_csv_file,
    )
except ImportError:  # pragma: no cover - fallback for direct source execution
    from project_paths import (  # type: ignore
        DEFAULT_LOG_PATTERN,
        get_default_output_dir,
        get_default_log_dir,
        list_csv_files,
        resolve_package_path,
        select_latest_csv_file,
    )

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)
RPM_TO_RAD_S = 1.0 / RAD_S_TO_RPM

ROTOR_SPEED_RAD_S_COLUMNS = {
    "Upper_rotor_cmd",
    "Lower_rotor_cmd",
    "Upper_rotor_actual",
    "Lower_rotor_actual",
    "Upper_rotor_speed_cmd",
    "Lower_rotor_speed_cmd",
    "Upper_rotor_speed_actual",
    "Lower_rotor_speed_actual",
}

ROTOR_SPEED_RPM_COLUMNS = {
    "Upper_rotor_speed_cmd_rpm",
    "Lower_rotor_speed_cmd_rpm",
    "Upper_rotor_speed_actual_rpm",
    "Lower_rotor_speed_actual_rpm",
}

ROTOR_SPEED_RAD_S_TO_RPM_COLUMN = {
    "Upper_rotor_speed_cmd": "Upper_rotor_speed_cmd_rpm",
    "Lower_rotor_speed_cmd": "Lower_rotor_speed_cmd_rpm",
    "Upper_rotor_speed_actual": "Upper_rotor_speed_actual_rpm",
    "Lower_rotor_speed_actual": "Lower_rotor_speed_actual_rpm",
}

ROTOR_SPEED_RPM_TO_RAD_S_COLUMN = {
    rpm_col: rad_s_col for rad_s_col, rpm_col in ROTOR_SPEED_RAD_S_TO_RPM_COLUMN.items()
}


class CSVVisualizer:
    """通用CSV数据可视化类（折线图 + 3D轨迹）"""

    def __init__(self, base_dir: str | Path | None = None, output_dir: str | Path | None = None):
        self.base_dir = resolve_package_path(base_dir)
        self.output_dir = self._resolve_output_dir(output_dir)
        self._create_output_directory()

    def _resolve_output_dir(self, output_dir: str | Path | None) -> Path:
        if output_dir is None:
            return get_default_output_dir()

        candidate = Path(output_dir).expanduser()
        if candidate.is_absolute():
            return candidate
        return self.base_dir / candidate

    def _create_output_directory(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_csv_files(self, directory: str | Path | None = None) -> List[Path]:
        search_dir = resolve_package_path(directory if directory is not None else get_default_log_dir())
        csv_files = list_csv_files(search_dir, DEFAULT_LOG_PATTERN)
        if csv_files:
            return csv_files

        if not search_dir.exists() or not search_dir.is_dir():
            return []

        fallback = [path for path in search_dir.glob("*.csv") if path.is_file()]
        return sorted(fallback, key=lambda path: path.stat().st_mtime, reverse=True)

    def load_csv(self, file_path: str | Path) -> Optional[pd.DataFrame]:
        csv_path = resolve_package_path(file_path)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"  成功加载文件: {csv_path.name}")
            print(f"  数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
            return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='gbk')
                print(f"  成功加载文件(GBK编码): {csv_path.name}")
                return df
            except Exception as e:
                print(f"  错误: 无法读取文件 {csv_path}: {e}")
                return None
        except Exception as e:
            print(f"  错误: 读取文件失败 {csv_path}: {e}")
            return None
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
    
    def suggest_default_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
        analysis = self.analyze_dataframe(df)
        numeric_cols = analysis['numeric_columns']
        
        if not numeric_cols:
            return None, []
        
        x_col = None
        time_keywords = ['time', 'timestamp', 'date', 'sec', 'ms', 'frame']
        
        for col in numeric_cols[:5]:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                x_col = col
                break
        
        if x_col is None and len(numeric_cols) > 0:
            x_col = numeric_cols[0]
            
        y_cols = [col for col in numeric_cols if col != x_col]
        
        return x_col, y_cols
    
    def generate_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def prepare_plot_dataframe(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        rotor_speed_unit: str = "rpm",
    ) -> Tuple[pd.DataFrame, List[str]]:
        plot_df = pd.DataFrame(index=df.index)
        plot_df[x_col] = df[x_col]
        plot_y_cols: List[str] = []

        for y_col in y_cols:
            if y_col not in df.columns:
                continue

            target_col = y_col
            series = df[y_col]

            if rotor_speed_unit == "rpm":
                if y_col in ROTOR_SPEED_RPM_COLUMNS:
                    target_col = f"{y_col.removesuffix('_rpm')} [RPM]"
                elif y_col in ROTOR_SPEED_RAD_S_COLUMNS:
                    rpm_col = ROTOR_SPEED_RAD_S_TO_RPM_COLUMN.get(y_col)
                    if rpm_col and rpm_col in df.columns:
                        series = df[rpm_col]
                    else:
                        series = df[y_col] * RAD_S_TO_RPM
                    target_col = f"{y_col} [RPM]"
            elif rotor_speed_unit == "rad_s":
                if y_col in ROTOR_SPEED_RAD_S_COLUMNS:
                    target_col = f"{y_col} [rad/s]"
                elif y_col in ROTOR_SPEED_RPM_COLUMNS:
                    rad_s_col = ROTOR_SPEED_RPM_TO_RAD_S_COLUMN.get(y_col)
                    if rad_s_col and rad_s_col in df.columns:
                        series = df[rad_s_col]
                    else:
                        series = df[y_col] * RPM_TO_RAD_S
                    target_col = f"{y_col.removesuffix('_rpm')} [rad/s]"

            plot_df[target_col] = series
            plot_y_cols.append(target_col)

        return plot_df, plot_y_cols

    def prepare_trajectory_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str = "Time",
    ) -> Tuple[Optional[pd.DataFrame], bool]:
        required_cols = [time_col, "X", "Y", "Z"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  跳过 3D 轨迹图: 缺少必要列 {missing_cols}")
            return None, False

        has_reference = all(col in df.columns for col in ("X_ref", "Y_ref", "Z_ref"))
        selected_cols = required_cols + (["X_ref", "Y_ref", "Z_ref"] if has_reference else [])
        trajectory_df = df[selected_cols].dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)
        if trajectory_df.empty:
            print("  跳过 3D 轨迹图: X/Y/Z 轨迹数据为空")
            return None, False

        return trajectory_df, has_reference

    def _compute_trajectory_limits(
        self,
        trajectory_df: pd.DataFrame,
        has_reference: bool,
    ) -> Dict[str, Tuple[float, float]]:
        axes_data = {
            "x": trajectory_df["X"].tolist(),
            "y": trajectory_df["Y"].tolist(),
            "z": trajectory_df["Z"].tolist(),
        }
        if has_reference:
            axes_data["x"].extend(trajectory_df["X_ref"].tolist())
            axes_data["y"].extend(trajectory_df["Y_ref"].tolist())
            axes_data["z"].extend(trajectory_df["Z_ref"].tolist())

        raw_limits: Dict[str, Tuple[float, float]] = {}
        max_span = 1e-3
        centers: Dict[str, float] = {}
        for axis_name, values in axes_data.items():
            axis_min = min(values)
            axis_max = max(values)
            span = max(axis_max - axis_min, 1e-3)
            max_span = max(max_span, span)
            centers[axis_name] = 0.5 * (axis_min + axis_max)
            raw_limits[axis_name] = (axis_min, axis_max)

        padding = max(0.08 * max_span, 0.05)
        half_span = 0.5 * max_span + padding
        return {
            axis_name: (centers[axis_name] - half_span, centers[axis_name] + half_span)
            for axis_name in raw_limits
        }

    def _apply_trajectory_axes_style(
        self,
        ax,
        limits: Dict[str, Tuple[float, float]],
        title: str,
    ) -> None:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["y"])
        ax.set_zlim(*limits["z"])
        ax.view_init(elev=24.0, azim=-58.0)
        ax.grid(True, alpha=0.25)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(
                (
                    limits["x"][1] - limits["x"][0],
                    limits["y"][1] - limits["y"][0],
                    limits["z"][1] - limits["z"][0],
                )
            )

    def build_animation_frame_indices(
        self,
        point_count: int,
        frame_step: int = 5,
        max_frames: int = 400,
    ) -> List[int]:
        if point_count <= 0:
            return []

        effective_step = max(1, int(frame_step))
        if max_frames > 0:
            effective_step = max(effective_step, math.ceil(point_count / max_frames))

        indices = list(range(0, point_count, effective_step))
        if indices[-1] != point_count - 1:
            indices.append(point_count - 1)
        return indices

    def plot_trajectory_3d_static(
        self,
        trajectory_df: pd.DataFrame,
        has_reference: bool,
        title: str,
        filename: str,
    ) -> bool:
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            limits = self._compute_trajectory_limits(trajectory_df, has_reference)

            if has_reference:
                ax.plot(
                    trajectory_df["X_ref"],
                    trajectory_df["Y_ref"],
                    trajectory_df["Z_ref"],
                    linestyle='--',
                    linewidth=1.5,
                    color='tab:orange',
                    alpha=0.85,
                    label='Reference trajectory',
                )

            ax.plot(
                trajectory_df["X"],
                trajectory_df["Y"],
                trajectory_df["Z"],
                linewidth=2.0,
                color='tab:blue',
                label='Actual trajectory',
            )

            ax.scatter(
                [trajectory_df.iloc[0]["X"]],
                [trajectory_df.iloc[0]["Y"]],
                [trajectory_df.iloc[0]["Z"]],
                color='tab:green',
                s=42,
                label='Start',
            )
            ax.scatter(
                [trajectory_df.iloc[-1]["X"]],
                [trajectory_df.iloc[-1]["Y"]],
                [trajectory_df.iloc[-1]["Z"]],
                color='tab:red',
                s=42,
                label='End',
            )

            self._apply_trajectory_axes_style(ax, limits, title)
            ax.legend(loc='upper left', fontsize=9)
            plt.tight_layout()

            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close()

            print(f"    3D静态轨迹图已保存: {save_path}")
            return True
        except Exception as e:
            print(f"    错误: 生成 3D 静态轨迹图失败: {e}")
            plt.close()
            return False

    def plot_trajectory_3d_animation(
        self,
        trajectory_df: pd.DataFrame,
        has_reference: bool,
        title: str,
        filename: str,
        frame_step: int = 5,
        max_frames: int = 400,
        fps: int = 20,
    ) -> bool:
        try:
            frame_indices = self.build_animation_frame_indices(
                len(trajectory_df),
                frame_step=frame_step,
                max_frames=max_frames,
            )
            if not frame_indices:
                print("    跳过 3D 动态轨迹图: 无有效动画帧")
                return False

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            limits = self._compute_trajectory_limits(trajectory_df, has_reference)
            self._apply_trajectory_axes_style(ax, limits, title)

            if has_reference:
                ax.plot(
                    trajectory_df["X_ref"],
                    trajectory_df["Y_ref"],
                    trajectory_df["Z_ref"],
                    linestyle='--',
                    linewidth=1.2,
                    color='tab:orange',
                    alpha=0.55,
                    label='Reference trajectory',
                )

            actual_line, = ax.plot([], [], [], linewidth=2.2, color='tab:blue', label='Actual trajectory')
            current_point = ax.scatter([], [], [], color='tab:red', s=48, label='Current position')
            start_point = ax.scatter(
                [trajectory_df.iloc[0]["X"]],
                [trajectory_df.iloc[0]["Y"]],
                [trajectory_df.iloc[0]["Z"]],
                color='tab:green',
                s=42,
                label='Start',
            )
            end_point = ax.scatter(
                [trajectory_df.iloc[-1]["X"]],
                [trajectory_df.iloc[-1]["Y"]],
                [trajectory_df.iloc[-1]["Z"]],
                color='tab:purple',
                s=36,
                alpha=0.4,
                label='Final target',
            )
            _ = (start_point, end_point)
            time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes, fontsize=10)
            ax.legend(loc='upper left', fontsize=9)

            def _set_scatter_point(scatter, x_val: float, y_val: float, z_val: float) -> None:
                scatter._offsets3d = ([x_val], [y_val], [z_val])

            def init():
                actual_line.set_data([], [])
                actual_line.set_3d_properties([])
                _set_scatter_point(
                    current_point,
                    float(trajectory_df.iloc[0]["X"]),
                    float(trajectory_df.iloc[0]["Y"]),
                    float(trajectory_df.iloc[0]["Z"]),
                )
                time_text.set_text("")
                return actual_line, current_point, time_text

            def update(frame_idx: int):
                end_idx = frame_indices[frame_idx]
                path_slice = trajectory_df.iloc[: end_idx + 1]
                actual_line.set_data(path_slice["X"], path_slice["Y"])
                actual_line.set_3d_properties(path_slice["Z"])
                _set_scatter_point(
                    current_point,
                    float(path_slice.iloc[-1]["X"]),
                    float(path_slice.iloc[-1]["Y"]),
                    float(path_slice.iloc[-1]["Z"]),
                )
                time_text.set_text(f"t = {float(path_slice.iloc[-1]['Time']):.2f} s")
                return actual_line, current_point, time_text

            animation = FuncAnimation(
                fig,
                update,
                frames=len(frame_indices),
                init_func=init,
                interval=max(1, int(1000 / max(1, fps))),
                blit=False,
                repeat=False,
            )

            save_path = self.output_dir / filename
            animation.save(save_path, writer=PillowWriter(fps=max(1, fps)))
            plt.close()

            print(f"    3D动态轨迹图已保存: {save_path}")
            return True
        except Exception as e:
            print(f"    错误: 生成 3D 动态轨迹图失败: {e}")
            plt.close()
            return False
    
    def plot_line_chart(self, df: pd.DataFrame, x_col: str, y_cols: List[str],
                        title: str, filename: str) -> bool:
        try:
            num_y = len(y_cols)
            rows = (num_y + 2) // 3
            cols = min(3, num_y)
            
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            if num_y == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if rows > 1 else axes
            
            colors = plt.cm.tab10.colors
            
            for idx, y_col in enumerate(y_cols):
                valid_data = df[[x_col, y_col]].dropna()
                axes[idx].plot(valid_data[x_col], valid_data[y_col], 
                              label=y_col, linewidth=1.5, alpha=0.8, color=colors[idx % len(colors)])
                axes[idx].set_xlabel(x_col, fontsize=10)
                axes[idx].set_ylabel(y_col, fontsize=10)
                axes[idx].set_title(f'{y_col}', fontsize=11, fontweight='bold')
                axes[idx].legend(loc='best', fontsize=9)
                axes[idx].grid(True, alpha=0.3)
            
            for idx in range(num_y, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    折线图已保存: {save_path}")
            return True
            
        except Exception as e:
            print(f"    错误: 生成折线图失败: {e}")
            plt.close()
            return False
    
    def process_csv_file(self, file_path: str | Path,
                         x_col: Optional[str] = None,
                         y_cols: Optional[List[str]] = None,
                         rotor_speed_unit: str = "rpm",
                         generate_trajectory_3d: bool = False,
                         skip_line_chart: bool = False,
                         trajectory_frame_step: int = 5,
                         trajectory_max_frames: int = 400,
                         trajectory_fps: int = 20) -> Dict:
        csv_path = resolve_package_path(file_path)
        csv_filename = csv_path.name
        print(f"\n{'='*60}")
        print(f"处理文件: {csv_filename}")
        print(f"  输入路径: {csv_path}")
        print('='*60)

        df = self.load_csv(csv_path)
        if df is None or df.empty:
            print(f"  错误: 无法加载数据或数据为空")
            return {'success': False, 'file': csv_filename, 'charts': []}
        
        analysis = self.analyze_dataframe(df)
        print(f"  数值列: {len(analysis['numeric_columns'])} 个")
        print(f"  数值列名: {', '.join(analysis['numeric_columns'][:5])}")
        
        if x_col is None or y_cols is None:
            suggested_x, suggested_y = self.suggest_default_columns(df)
            if x_col is None:
                x_col = suggested_x
            if y_cols is None:
                y_cols = suggested_y
        
        timestamp = self.generate_timestamp()
        base_name = csv_path.stem
        results = {'success': False, 'charts': []}

        if not skip_line_chart:
            if not y_cols:
                print("  提示: 未指定有效 Y 轴列，跳过折线图")
            else:
                plot_df, plot_y_cols = self.prepare_plot_dataframe(df, x_col, y_cols, rotor_speed_unit=rotor_speed_unit)
                if not plot_y_cols:
                    print("  提示: 选择的列不可用于折线图，已跳过")
                else:
                    line_filename = f"{base_name}_line_{timestamp}.png"
                    if self.plot_line_chart(
                        plot_df,
                        x_col,
                        plot_y_cols,
                        f"Line Chart: {base_name}",
                        line_filename,
                    ):
                        results['success'] = True
                        results['charts'].append(('line', line_filename))

        if generate_trajectory_3d:
            trajectory_df, has_reference = self.prepare_trajectory_dataframe(df, time_col=x_col or "Time")
            if trajectory_df is not None:
                static_filename = f"{base_name}_trajectory3d_{timestamp}.png"
                if self.plot_trajectory_3d_static(
                    trajectory_df,
                    has_reference,
                    f"3D Trajectory: {base_name}",
                    static_filename,
                ):
                    results['success'] = True
                    results['charts'].append(('trajectory3d_static', static_filename))

                animation_filename = f"{base_name}_trajectory3d_{timestamp}.gif"
                if self.plot_trajectory_3d_animation(
                    trajectory_df,
                    has_reference,
                    f"3D Trajectory Animation: {base_name}",
                    animation_filename,
                    frame_step=trajectory_frame_step,
                    max_frames=trajectory_max_frames,
                    fps=trajectory_fps,
                ):
                    results['success'] = True
                    results['charts'].append(('trajectory3d_animation', animation_filename))
        
        return results
    
    def process_batch(self, csv_files: List[str | Path],
                      x_col: Optional[str] = None,
                      y_cols: Optional[List[str]] = None,
                      rotor_speed_unit: str = "rpm",
                      generate_trajectory_3d: bool = False,
                      skip_line_chart: bool = False,
                      trajectory_frame_step: int = 5,
                      trajectory_max_frames: int = 400,
                      trajectory_fps: int = 20) -> Dict:
        summary = {
            'total_files': len(csv_files),
            'successful': 0,
            'failed': 0,
            'total_charts': 0,
            'details': []
        }
        
        for file_path in csv_files:
            result = self.process_csv_file(
                file_path,
                x_col,
                y_cols,
                rotor_speed_unit=rotor_speed_unit,
                generate_trajectory_3d=generate_trajectory_3d,
                skip_line_chart=skip_line_chart,
                trajectory_frame_step=trajectory_frame_step,
                trajectory_max_frames=trajectory_max_frames,
                trajectory_fps=trajectory_fps,
            )
            
            if result['success']:
                summary['successful'] += 1
                summary['total_charts'] += len(result['charts'])
            else:
                summary['failed'] += 1
            
            summary['details'].append(result)
        
        return summary
    
    def print_summary(self, summary: Dict) -> None:
        print("\n" + "="*60)
        print("处理完成 - 结果汇总")
        print("="*60)
        print(f"总文件数: {summary['total_files']}")
        print(f"成功处理: {summary['successful']}")
        print(f"处理失败: {summary['failed']}")
        print(f"生成图表总数: {summary['total_charts']}")
        print(f"输出目录: {self.output_dir}")
        print("="*60)
    
    def interactive_select(self, df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
        analysis = self.analyze_dataframe(df)
        numeric_cols = analysis['numeric_columns']
        
        if not numeric_cols:
            print("错误: 未找到数值列")
            return None, []
        
        print("\n" + "="*50)
        print("可用数值列:")
        print("="*50)
        for i, col in enumerate(numeric_cols, 1):
            print(f"  {i}. {col}")
        
        x_col_idx = input("\n请选择X轴列号 (直接回车使用第一列): ").strip()
        if x_col_idx:
            try:
                x_col = numeric_cols[int(x_col_idx) - 1]
            except (ValueError, IndexError):
                x_col = numeric_cols[0]
        else:
            x_col = numeric_cols[0]
        
        y_cols_input = input("请输入Y轴列号(用逗号分隔，如1,2,3)，直接回车选择所有列: ").strip()
        if y_cols_input:
            try:
                y_cols = [numeric_cols[int(i) - 1] for i in y_cols_input.split(',')]
            except (ValueError, IndexError):
                y_cols = numeric_cols[1:] if len(numeric_cols) > 1 else numeric_cols
        else:
            y_cols = [col for col in numeric_cols if col != x_col]
        
        print(f"\n选择的配置:")
        print(f"  X轴: {x_col}")
        print(f"  Y轴: {', '.join(y_cols)}")
        
        return x_col, y_cols


def parse_arguments(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description='无人机数据可视化工具 - 折线图 / 3D轨迹图生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 -m mmc_control.drone_visualizer                    # 处理 fly_data 下最新CSV
  python3 -m mmc_control.drone_visualizer -f data.csv        # 处理指定CSV文件
  python3 -m mmc_control.drone_visualizer -d ./fly_data       # 指定目录并自动找最新CSV
  python3 -m mmc_control.drone_visualizer --batch            # 批量处理目录下所有CSV
  python3 -m mmc_control.drone_visualizer -o my_output       # 指定输出目录
  python3 -m mmc_control.drone_visualizer -i                 # 交互模式选择列
  python3 -m mmc_control.drone_visualizer -x Time -y X Y Z   # 指定X轴和Y轴列
  python3 -m mmc_control.drone_visualizer --rotor-speeds     # 直接输出上下旋翼转速图（默认 RPM）
  python3 -m mmc_control.drone_visualizer --rotor-speed-unit rad_s --rotor-speeds
  python3 -m mmc_control.drone_visualizer -f fly_data/log.csv --trajectory-3d --no-line-chart
        """
    )
    
    parser.add_argument('-f', '--file', type=str, default=None,
                       help='指定要处理的CSV文件路径')
    parser.add_argument('-d', '--directory', type=str, default=str(get_default_log_dir()),
                       help='指定要处理的目录路径 (默认: fly_data)')
    parser.add_argument('-o', '--output', type=str, default='output_picture',
                       help='指定输出目录 (默认: output_picture)')
    parser.add_argument('-x', '--xaxis', type=str, default=None,
                       help='指定X轴数据列')
    parser.add_argument('-y', '--yaxis', nargs='+', default=None,
                       help='指定Y轴数据列(可多个)')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理目录下所有CSV文件（默认仅处理最新日志）')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='启用交互模式选择数据列')
    parser.add_argument('--rotor-speeds', action='store_true',
                       help='快捷输出上下旋翼转速列（优先使用 *_speed_*，缺失时回退到原有 signed cmd/actual 列）')
    parser.add_argument('--rotor-speed-unit', choices=('rpm', 'rad_s'), default='rpm',
                       help='绘制旋翼转速列时使用的单位（默认: rpm）')
    parser.add_argument('--trajectory-3d', action='store_true',
                       help='额外生成空间 3D 静态轨迹图和 3D 动态轨迹 GIF（依赖 X/Y/Z，若存在 X_ref/Y_ref/Z_ref 会一并叠加参考轨迹）')
    parser.add_argument('--no-line-chart', action='store_true',
                       help='跳过默认折线图输出；常与 --trajectory-3d 搭配使用')
    parser.add_argument('--trajectory-3d-frame-step', type=int, default=5,
                       help='3D 动态轨迹图的抽帧步长（默认: 5）')
    parser.add_argument('--trajectory-3d-max-frames', type=int, default=400,
                       help='3D 动态轨迹图最大帧数，超过后自动增大抽帧步长（默认: 400）')
    parser.add_argument('--trajectory-3d-fps', type=int, default=20,
                       help='3D 动态轨迹 GIF 的帧率（默认: 20）')
    
    return parser.parse_args(list(argv) if argv is not None else None)


def main():
    print("="*60)
    print("无人机数据可视化工具 v2.2.0 (折线图 + 3D轨迹版)")
    print("="*60)
    
    args = parse_arguments()
    
    visualizer = CSVVisualizer(output_dir=args.output)
    search_dir = resolve_package_path(args.directory)
    csv_files: List[Path] = []
    used_search_dir = search_dir
    
    if args.file:
        file_path = resolve_package_path(args.file)
        if file_path.exists():
            csv_files = [file_path]
            print(f"\n指定CSV文件: {file_path}")
        else:
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
    else:
        csv_files = visualizer.find_csv_files(search_dir)
        if not csv_files and search_dir == get_default_log_dir():
            legacy_dir = visualizer.base_dir
            legacy_csv_files = visualizer.find_csv_files(legacy_dir)
            if legacy_csv_files:
                csv_files = legacy_csv_files
                used_search_dir = legacy_dir
        if not csv_files:
            print(f"错误: 在目录 {search_dir} 中未找到CSV文件")
            sys.exit(1)

        if args.batch:
            print(f"\n在目录 {used_search_dir} 中找到 {len(csv_files)} 个CSV文件（批量模式）:")
            for csv_file in csv_files:
                print(f"  - {csv_file.name}")
        else:
            latest_csv = select_latest_csv_file(used_search_dir) or csv_files[0]
            print(f"\n在目录 {used_search_dir} 中找到 {len(csv_files)} 个CSV文件，自动选择最新文件:")
            print(f"  - {latest_csv.name}")
            csv_files = [latest_csv]
    
    if not csv_files:
        print(f"错误: 未找到可处理的CSV文件")
        sys.exit(1)

    x_col = args.xaxis
    y_cols = args.yaxis

    if args.rotor_speeds:
        rotor_speed_rpm_cols = [
            "Upper_rotor_speed_cmd_rpm",
            "Lower_rotor_speed_cmd_rpm",
            "Upper_rotor_speed_actual_rpm",
            "Lower_rotor_speed_actual_rpm",
        ]
        rotor_speed_cols = [
            "Upper_rotor_speed_cmd",
            "Lower_rotor_speed_cmd",
            "Upper_rotor_speed_actual",
            "Lower_rotor_speed_actual",
        ]
        legacy_rotor_cols = [
            "Upper_rotor_cmd",
            "Lower_rotor_cmd",
            "Upper_rotor_actual",
            "Lower_rotor_actual",
        ]
        df_preview = visualizer.load_csv(csv_files[0])
        if df_preview is not None:
            available_cols = set(df_preview.columns)
            selected = []
            if args.rotor_speed_unit == "rpm":
                selected = [col for col in rotor_speed_rpm_cols if col in available_cols]
            if not selected:
                selected = [col for col in rotor_speed_cols if col in available_cols]
            if not selected:
                selected = [col for col in legacy_rotor_cols if col in available_cols]
            y_cols = selected
            x_col = x_col or "Time"

    if args.interactive and len(csv_files) == 1:
        df = visualizer.load_csv(csv_files[0])
        if df is not None:
            x_col, y_cols = visualizer.interactive_select(df)
    
    if len(csv_files) == 1:
        result = visualizer.process_csv_file(
            csv_files[0],
            x_col,
            y_cols,
            rotor_speed_unit=args.rotor_speed_unit,
            generate_trajectory_3d=args.trajectory_3d,
            skip_line_chart=args.no_line_chart,
            trajectory_frame_step=args.trajectory_3d_frame_step,
            trajectory_max_frames=args.trajectory_3d_max_frames,
            trajectory_fps=args.trajectory_3d_fps,
        )
        summary = {
            'total_files': 1,
            'successful': 1 if result['success'] else 0,
            'failed': 0 if result['success'] else 1,
            'total_charts': len(result['charts']),
            'details': [result]
        }
        visualizer.print_summary(summary)
    else:
        summary = visualizer.process_batch(
            csv_files,
            x_col,
            y_cols,
            rotor_speed_unit=args.rotor_speed_unit,
            generate_trajectory_3d=args.trajectory_3d,
            skip_line_chart=args.no_line_chart,
            trajectory_frame_step=args.trajectory_3d_frame_step,
            trajectory_max_frames=args.trajectory_3d_max_frames,
            trajectory_fps=args.trajectory_3d_fps,
        )
        visualizer.print_summary(summary)
    
    print("\n可视化完成！")


if __name__ == "__main__":
    main()
