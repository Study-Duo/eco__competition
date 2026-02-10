"""
TEG温差发电混合分析系统
版本：2.0
功能：实时数据采集、处理、分析和可视化
数据格式兼容性：支持Arduino RAW模式输出
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import serial.tools.list_ports
import threading
import time
import os
from datetime import datetime
from scipy import stats
from collections import deque
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据类定义 ====================
@dataclass
class SensorConfig:
    """传感器配置类"""
    sample_rate: float = 1.0  # 采样率 (Hz)
    buffer_size: int = 1000   # 缓冲区大小
    load_resistance: float = 10.0  # 负载电阻 (Ω)
    teg_resistance: float = 4.0    # TEG内阻 (Ω)
    seebeck_coefficient: float = 40.0  # 塞贝克系数 (mV/°C)
    
@dataclass
class TEGDataPoint:
    """单个数据点类"""
    timestamp: float          # 时间戳 (ms)
    hot_temp: float          # 热端温度 (°C)
    cold_temp: float         # 冷端温度 (°C)
    voltage_raw: float       # 原始电压 (mV)
    ch1_raw: int             # 通道1原始值
    ch2_raw: int             # 通道2原始值
    ch3_raw: int             # 通道3原始值
    
    # 计算属性
    @property
    def temp_diff(self) -> float:
        """温差 (°C)"""
        return self.hot_temp - self.cold_temp if self.hot_temp != -999 and self.cold_temp != -999 else 0.0
    
    @property
    def voltage_v(self) -> float:
        """电压 (V)"""
        return self.voltage_raw / 1000.0
    
    @property
    def current_ma(self) -> float:
        """电流 (mA) - 使用负载电阻计算"""
        if self.voltage_v > 0 and config.load_resistance > 0:
            return (self.voltage_v / config.load_resistance) * 1000
        return 0.0
    
    @property
    def power_mw(self) -> float:
        """功率 (mW)"""
        return self.voltage_v * self.current_ma
    
    @property
    def seebeck_mv_per_c(self) -> float:
        """塞贝克系数 (mV/°C)"""
        if abs(self.temp_diff) > 0.1:
            return (self.voltage_raw / self.temp_diff) if self.temp_diff != 0 else 0.0
        return 0.0
    
    @property
    def efficiency(self) -> float:
        """效率 (%) - 简化计算"""
        if abs(self.temp_diff) > 0.1 and self.hot_temp > 0:
            carnot = 1 - (self.cold_temp + 273.15) / (self.hot_temp + 273.15)
            return (self.power_mw / 1000) / (carnot * 100) * 100 if carnot > 0 else 0.0
        return 0.0

# ==================== 全局配置 ====================
config = SensorConfig()
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== TEG分析器类 ====================
class TEGHybridAnalyzer:
    """混合模式TEG分析器"""
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        """
        初始化分析器
        
        参数:
        port: 串口号
        baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.is_running = False
        
        # 数据管理
        self.data_buffer: List[TEGDataPoint] = []
        self.raw_data: List[List] = []  # 原始CSV数据
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'data_points': 0,
            'start_time': None,
            'sample_rate': 0.0,
            'errors': 0
        }
        
        # 线程控制
        self.serial_thread = None
        self.stop_event = threading.Event()
        
        # 数据保存
        self.save_enabled = False
        self.csv_file = None
        self.csv_writer = None
        self.json_config_file = None
        
        # 实时绘图
        self.fig = None
        self.animation = None
        self.plot_objects = {}
        
        print("TEG混合分析器初始化完成")
        print(f"配置: 采样率={config.sample_rate}Hz, 缓冲区={config.buffer_size}")
    
    # ==================== 串口通信 ====================
    def auto_detect_port(self) -> Optional[str]:
        """自动检测Arduino串口"""
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            print("未检测到串口设备")
            return None
        
        # 优先选择Arduino设备
        for port in ports:
            desc = port.description.lower()
            if any(keyword in desc for keyword in ['arduino', 'ch340', 'cp210', 'usb serial']):
                print(f"检测到Arduino: {port.device} - {port.description}")
                return port.device
        
        # 返回第一个可用端口
        print(f"使用端口: {ports[0].device}")
        return ports[0].device
    
    def connect(self) -> bool:
        """连接串口设备"""
        if self.is_connected:
            return True
        
        port = self.port or self.auto_detect_port()
        if not port:
            return False
        
        try:
            print(f"连接串口: {port} @ {self.baudrate}bps")
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            time.sleep(2)  # 等待Arduino重启
            self.serial_conn.reset_input_buffer()
            
            # 验证连接
            self._send_command("status")
            time.sleep(0.5)
            
            self.is_connected = True
            self.stats['start_time'] = time.time()
            print("串口连接成功")
            return True
            
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self.stop()
        
        if self.serial_conn and self.is_connected:
            try:
                self.serial_conn.close()
            except:
                pass
        
        self.is_connected = False
        print("串口已断开")
    
    def _send_command(self, command: str):
        """发送命令到Arduino"""
        if self.serial_conn and self.is_connected:
            try:
                self.serial_conn.write(f"{command}\n".encode('utf-8'))
            except:
                pass
    
    # ==================== 数据采集 ====================
    def start(self, save_data: bool = False):
        """开始数据采集"""
        if not self.is_connected:
            if not self.connect():
                return False
        
        self.save_enabled = save_data
        if save_data:
            self._setup_data_saving()
        
        self.is_running = True
        self.stop_event.clear()
        
        # 启动数据采集线程
        self.serial_thread = threading.Thread(
            target=self._data_acquisition_thread,
            daemon=True
        )
        self.serial_thread.start()
        
        print("数据采集已启动")
        return True
    
    def stop(self):
        """停止数据采集"""
        print("停止数据采集...")
        self.is_running = False
        self.stop_event.set()
        
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2.0)
        
        if self.save_enabled:
            self._finish_data_saving()
        
        print("数据采集已停止")
    
    def _data_acquisition_thread(self):
        """数据采集线程"""
        print("数据采集线程启动")
        line_buffer = ""
        
        while self.is_running and not self.stop_event.is_set():
            try:
                if not self.serial_conn or not self.is_connected:
                    break
                
                # 读取数据
                if self.serial_conn.in_waiting > 0:
                    raw_data = self.serial_conn.read(self.serial_conn.in_waiting)
                    line_buffer += raw_data.decode('utf-8', errors='ignore')
                    
                    # 处理完整行
                    while '\n' in line_buffer:
                        line, line_buffer = line_buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line and not line.startswith('#'):
                            self._process_data_line(line)
                
                time.sleep(0.01)
                
            except Exception as e:
                self.stats['errors'] += 1
                if self.stats['errors'] % 10 == 0:
                    print(f"数据采集错误: {e}")
                time.sleep(0.1)
        
        print("数据采集线程结束")
    
    def _process_data_line(self, line: str):
        """处理单行数据"""
        try:
            # 跳过标题行和非数据行
            if any(keyword in line.lower() for keyword in ['timestamp', 'time', 'version']):
                return
            
            # 解析CSV数据
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) >= 7:  # 最少需要7个字段
                # 解析数据
                data_point = TEGDataPoint(
                    timestamp=float(parts[0]),
                    hot_temp=float(parts[1]),
                    cold_temp=float(parts[2]),
                    voltage_raw=float(parts[3]),
                    ch1_raw=int(parts[4]),
                    ch2_raw=int(parts[5]),
                    ch3_raw=int(parts[6])
                )
                
                # 添加到缓冲区
                with self.data_lock:
                    self.data_buffer.append(data_point)
                    self.raw_data.append(parts)
                    
                    # 限制缓冲区大小
                    if len(self.data_buffer) > config.buffer_size:
                        self.data_buffer.pop(0)
                        self.raw_data.pop(0)
                
                self.stats['data_points'] += 1
                
                # 计算采样率
                if self.stats['data_points'] % 10 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    self.stats['sample_rate'] = self.stats['data_points'] / elapsed
                
                # 保存数据
                if self.save_enabled and self.csv_writer:
                    self.csv_writer.writerow([
                        data_point.timestamp,
                        data_point.hot_temp,
                        data_point.cold_temp,
                        data_point.voltage_raw,
                        data_point.ch1_raw,
                        data_point.ch2_raw,
                        data_point.ch3_raw,
                        data_point.temp_diff,
                        data_point.voltage_v,
                        data_point.current_ma,
                        data_point.power_mw,
                        data_point.seebeck_mv_per_c,
                        data_point.efficiency
                    ])
                    
                    if self.stats['data_points'] % 20 == 0:
                        self.csv_file.flush()
                        
        except Exception as e:
            self.stats['errors'] += 1
    
    # ==================== 数据保存 ====================
    def _setup_data_saving(self):
        """设置数据保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = "teg_data"
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # CSV文件
        csv_path = os.path.join(data_dir, f"teg_{timestamp}.csv")
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入标题
        headers = [
            'timestamp_ms', 'hot_temp_C', 'cold_temp_C', 'voltage_raw_mV',
            'ch1_raw', 'ch2_raw', 'ch3_raw', 'temp_diff_C', 'voltage_V',
            'current_mA', 'power_mW', 'seebeck_mV_per_C', 'efficiency_percent'
        ]
        self.csv_writer.writerow(headers)
        
        # 配置文件
        config_path = os.path.join(data_dir, f"config_{timestamp}.json")
        config_data = {
            'timestamp': timestamp,
            'sample_rate': config.sample_rate,
            'load_resistance': config.load_resistance,
            'teg_resistance': config.teg_resistance,
            'seebeck_coefficient': config.seebeck_coefficient,
            'data_columns': headers
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"数据保存到: {csv_path}")
        print(f"配置保存到: {config_path}")
    
    def _finish_data_saving(self):
        """完成数据保存"""
        if self.csv_file:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                print("数据文件已关闭")
            except:
                pass
            finally:
                self.csv_file = None
                self.csv_writer = None
    
    # ==================== 数据分析 ====================
    def get_latest_data(self, n_points: int = None) -> List[TEGDataPoint]:
        """获取最新数据点"""
        with self.data_lock:
            if not self.data_buffer:
                return []
            
            if n_points is None or n_points >= len(self.data_buffer):
                return self.data_buffer.copy()
            else:
                return self.data_buffer[-n_points:]
    
    def get_dataframe(self) -> pd.DataFrame:
        """获取DataFrame格式的数据"""
        data = self.get_latest_data()
        
        if not data:
            return pd.DataFrame()
        
        df_data = []
        for point in data:
            df_data.append({
                'timestamp': point.timestamp,
                'hot_temp': point.hot_temp,
                'cold_temp': point.cold_temp,
                'voltage_raw': point.voltage_raw,
                'voltage': point.voltage_v,
                'temp_diff': point.temp_diff,
                'current': point.current_ma,
                'power': point.power_mw,
                'seebeck': point.seebeck_mv_per_c,
                'efficiency': point.efficiency
            })
        
        return pd.DataFrame(df_data)
    
    def calculate_statistics(self) -> Dict:
        """计算统计信息"""
        df = self.get_dataframe()
        
        if df.empty:
            return {}
        
        stats = {
            'data_points': len(df),
            'duration_seconds': (df['timestamp'].max() - df['timestamp'].min()) / 1000 if len(df) > 1 else 0,
            'sample_rate_hz': self.stats['sample_rate'],
            'hot_temp': {
                'mean': df['hot_temp'].mean(),
                'min': df['hot_temp'].min(),
                'max': df['hot_temp'].max(),
                'std': df['hot_temp'].std()
            },
            'cold_temp': {
                'mean': df['cold_temp'].mean(),
                'min': df['cold_temp'].min(),
                'max': df['cold_temp'].max(),
                'std': df['cold_temp'].std()
            },
            'temp_diff': {
                'mean': df['temp_diff'].mean(),
                'min': df['temp_diff'].min(),
                'max': df['temp_diff'].max()
            },
            'voltage': {
                'mean': df['voltage'].mean(),
                'min': df['voltage'].min(),
                'max': df['voltage'].max()
            },
            'power': {
                'mean': df['power'].mean(),
                'min': df['power'].min(),
                'max': df['power'].max()
            },
            'seebeck': {
                'mean': df['seebeck'][df['seebeck'].abs() < 1000].mean() if len(df) > 0 else 0
            }
        }
        
        # 线性回归分析
        if len(df) > 5:
            valid_mask = (df['temp_diff'].abs() > 0.1) & (df['temp_diff'].abs() < 100)
            if valid_mask.sum() > 5:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df['temp_diff'][valid_mask], 
                        df['voltage'][valid_mask]
                    )
                    
                    stats['regression'] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'seebeck_calculated': slope * 1000
                    }
                except:
                    pass
        
        return stats
    
    # ==================== 实时可视化 ====================
    def create_real_time_dashboard(self):
        """创建实时仪表板"""
        print("创建实时仪表板...")
        
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('TEG温差发电实时监测系统 - 混合模式', fontsize=16, fontweight='bold')
        
        # 创建子图
        gs = self.fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. 温差-电压特性
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.plot_objects['scatter'] = ax1.scatter([], [], c=[], cmap='plasma', alpha=0.7, s=30)
        ax1.set_xlabel('温差 (°C)')
        ax1.set_ylabel('开路电压 (V)')
        ax1.set_title('温差-电压特性')
        ax1.grid(True, alpha=0.3)
        
        # 2. 温度时间序列
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.plot_objects['temp_hot'], = ax2.plot([], [], 'r-', label='热端', linewidth=2)
        self.plot_objects['temp_cold'], = ax2.plot([], [], 'b-', label='冷端', linewidth=2)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('温度 (°C)')
        ax2.set_title('温度变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 电压时间序列
        ax3 = self.fig.add_subplot(gs[0, 2])
        self.plot_objects['voltage'], = ax3.plot([], [], 'g-', linewidth=2)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('电压 (V)')
        ax3.set_title('电压输出')
        ax3.grid(True, alpha=0.3)
        
        # 4. 功率时间序列
        ax4 = self.fig.add_subplot(gs[1, 0])
        self.plot_objects['power'], = ax4.plot([], [], 'orange', linewidth=2)
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('功率 (mW)')
        ax4.set_title('输出功率')
        ax4.grid(True, alpha=0.3)
        
        # 5. 塞贝克系数
        ax5 = self.fig.add_subplot(gs[1, 1])
        self.plot_objects['seebeck'], = ax5.plot([], [], 'purple', linewidth=2)
        ax5.set_xlabel('时间 (s)')
        ax5.set_ylabel('塞贝克系数 (mV/°C)')
        ax5.set_title('塞贝克系数')
        ax5.grid(True, alpha=0.3)
        
        # 6. 效率
        ax6 = self.fig.add_subplot(gs[1, 2])
        self.plot_objects['efficiency'], = ax6.plot([], [], 'brown', linewidth=2)
        ax6.set_xlabel('时间 (s)')
        ax6.set_ylabel('效率 (%)')
        ax6.set_title('转换效率')
        ax6.grid(True, alpha=0.3)
        
        # 7. 原始数据
        ax7 = self.fig.add_subplot(gs[2, 0])
        self.plot_objects['raw_voltage'], = ax7.plot([], [], 'gray', linewidth=1, alpha=0.7)
        ax7.set_xlabel('时间 (s)')
        ax7.set_ylabel('原始电压 (mV)')
        ax7.set_title('原始电压信号')
        ax7.grid(True, alpha=0.3)
        
        # 8. 统计信息
        ax8 = self.fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        self.plot_objects['stats_text'] = ax8.text(
            0.02, 0.98, '正在初始化...',
            transform=ax8.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )
        
        # 创建动画
        self.animation = FuncAnimation(
            self.fig, self._update_dashboard,
            interval=500,  # 每500ms更新
            blit=False,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # 窗口关闭时停止
        self.stop()
    
    def _update_dashboard(self, frame):
        """更新仪表板"""
        try:
            data = self.get_latest_data(200)  # 获取最近200个点
            
            if not data:
                return []
            
            # 提取数据
            timestamps = [d.timestamp for d in data]
            time_rel = [(t - timestamps[0]) / 1000.0 for t in timestamps]
            
            hot_temps = [d.hot_temp for d in data]
            cold_temps = [d.cold_temp for d in data]
            voltages = [d.voltage_v for d in data]
            powers = [d.power_mw for d in data]
            seebecks = [d.seebeck_mv_per_c for d in data]
            efficiencies = [d.efficiency for d in data]
            raw_voltages = [d.voltage_raw for d in data]
            temp_diffs = [d.temp_diff for d in data]
            
            # 更新温差-电压图
            if len(temp_diffs) > 5:
                self.plot_objects['scatter'].set_offsets(np.column_stack([temp_diffs[-100:], voltages[-100:]]))
                self.plot_objects['scatter'].set_array(np.array(hot_temps[-100:]))
                
                if len(temp_diffs) > 1:
                    ax = self.plot_objects['scatter'].axes
                    x_min, x_max = min(temp_diffs[-100:]), max(temp_diffs[-100:])
                    y_min, y_max = min(voltages[-100:]), max(voltages[-100:])
                    if x_max - x_min > 0.1:
                        ax.set_xlim(x_min - 0.1, x_max + 0.1)
                    if y_max - y_min > 0.01:
                        ax.set_ylim(y_min - 0.01, y_max + 0.01)
            
            # 更新温度图
            if len(time_rel) > 1:
                self.plot_objects['temp_hot'].set_data(time_rel, hot_temps)
                self.plot_objects['temp_cold'].set_data(time_rel, cold_temps)
                ax = self.plot_objects['temp_hot'].axes
                if time_rel:
                    ax.set_xlim(max(0, time_rel[0]), time_rel[-1])
            
            # 更新电压图
            self.plot_objects['voltage'].set_data(time_rel, voltages)
            
            # 更新功率图
            self.plot_objects['power'].set_data(time_rel, powers)
            
            # 更新塞贝克系数图
            seebecks_filtered = [s for s in seebecks if abs(s) < 1000]
            if seebecks_filtered:
                self.plot_objects['seebeck'].set_data(time_rel[-len(seebecks_filtered):], seebecks_filtered)
            
            # 更新效率图
            self.plot_objects['efficiency'].set_data(time_rel, efficiencies)
            
            # 更新原始电压图
            self.plot_objects['raw_voltage'].set_data(time_rel, raw_voltages)
            
            # 更新统计信息
            stats = self.calculate_statistics()
            stats_text = self._format_stats_text(stats)
            self.plot_objects['stats_text'].set_text(stats_text)
            
        except Exception as e:
            pass
        
        return []
    
    def _format_stats_text(self, stats: Dict) -> str:
        """格式化统计信息文本"""
        if not stats:
            return "等待数据..."
        
        text = "=== 实时统计 ===\n\n"
        text += f"数据点数: {stats.get('data_points', 0)}\n"
        text += f"采样率: {stats.get('sample_rate_hz', 0):.1f} Hz\n"
        text += f"持续时间: {stats.get('duration_seconds', 0):.1f} s\n\n"
        
        if 'hot_temp' in stats:
            ht = stats['hot_temp']
            text += f"热端温度: {ht['mean']:.1f}°C\n"
            text += f"  (范围: {ht['min']:.1f} - {ht['max']:.1f})\n"
        
        if 'cold_temp' in stats:
            ct = stats['cold_temp']
            text += f"冷端温度: {ct['mean']:.1f}°C\n"
            text += f"  (范围: {ct['min']:.1f} - {ct['max']:.1f})\n"
        
        if 'temp_diff' in stats:
            td = stats['temp_diff']
            text += f"温差: {td['mean']:.1f}°C\n"
            text += f"  (范围: {td['min']:.1f} - {td['max']:.1f})\n"
        
        if 'voltage' in stats:
            v = stats['voltage']
            text += f"电压: {v['mean']:.3f} V\n"
            text += f"  (范围: {v['min']:.3f} - {v['max']:.3f})\n"
        
        if 'power' in stats:
            p = stats['power']
            text += f"功率: {p['mean']:.2f} mW\n"
            text += f"  (最大: {p['max']:.2f} mW)\n"
        
        if 'regression' in stats:
            reg = stats['regression']
            text += f"\n塞贝克系数: {reg['seebeck_calculated']:.2f} mV/°C\n"
            text += f"拟合优度 R²: {reg['r_squared']:.4f}\n"
        
        return text
    
    # ==================== 批量数据分析 ====================
    @staticmethod
    def analyze_saved_file(filepath: str):
        """分析已保存的数据文件"""
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            print(f"文件: {os.path.basename(filepath)}")
            print(f"数据点数: {len(df)}")
            print(f"数据列: {list(df.columns)}")
            
            # 创建分析图表
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f'TEG数据分析 - {os.path.basename(filepath)}', fontsize=16)
            
            # 子图布局
            axes = []
            for i in range(6):
                axes.append(fig.add_subplot(2, 3, i+1))
            
            # 1. 温差-电压特性
            axes[0].scatter(df['temp_diff_C'], df['voltage_V'], 
                           c=df['hot_temp_C'], cmap='plasma', alpha=0.6, s=20)
            axes[0].set_xlabel('温差 (°C)')
            axes[0].set_ylabel('电压 (V)')
            axes[0].set_title('温差-电压特性')
            axes[0].grid(True, alpha=0.3)
            
            # 2. 温度变化
            if 'timestamp_ms' in df.columns:
                time_rel = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000
                axes[1].plot(time_rel, df['hot_temp_C'], 'r-', label='热端', linewidth=2)
                axes[1].plot(time_rel, df['cold_temp_C'], 'b-', label='冷端', linewidth=2)
                axes[1].set_xlabel('时间 (s)')
            else:
                axes[1].plot(df.index, df['hot_temp_C'], 'r-', label='热端', linewidth=2)
                axes[1].plot(df.index, df['cold_temp_C'], 'b-', label='冷端', linewidth=2)
                axes[1].set_xlabel('数据点')
            axes[1].set_ylabel('温度 (°C)')
            axes[1].set_title('温度变化')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # 3. 功率变化
            if 'power_mW' in df.columns:
                axes[2].plot(time_rel if 'timestamp_ms' in df.columns else df.index, 
                           df['power_mW'], 'g-', linewidth=2)
                axes[2].set_xlabel('时间 (s)' if 'timestamp_ms' in df.columns else '数据点')
                axes[2].set_ylabel('功率 (mW)')
                axes[2].set_title('输出功率')
                axes[2].grid(True, alpha=0.3)
            
            # 4. 塞贝克系数分布
            if 'seebeck_mV_per_C' in df.columns:
                seebeck = df['seebeck_mV_per_C'].dropna()
                seebeck = seebeck[(seebeck.abs() < 1000) & (seebeck.abs() > 0.1)]
                axes[3].hist(seebeck, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[3].axvline(seebeck.mean(), color='red', linestyle='--', 
                               linewidth=2, label=f'平均: {seebeck.mean():.1f} mV/°C')
                axes[3].set_xlabel('塞贝克系数 (mV/°C)')
                axes[3].set_ylabel('频数')
                axes[3].set_title('塞贝克系数分布')
                axes[3].grid(True, alpha=0.3)
                axes[3].legend()
            
            # 5. 电压-功率关系
            if 'voltage_V' in df.columns and 'power_mW' in df.columns:
                axes[4].scatter(df['voltage_V'], df['power_mW'], 
                               c=df['temp_diff_C'], cmap='viridis', alpha=0.6, s=20)
                axes[4].set_xlabel('电压 (V)')
                axes[4].set_ylabel('功率 (mW)')
                axes[4].set_title('电压-功率关系')
                axes[4].grid(True, alpha=0.3)
            
            # 6. 统计信息
            axes[5].axis('off')
            stats_text = "=== 数据统计 ===\n\n"
            stats_text += f"数据点数: {len(df)}\n"
            stats_text += f"热端温度范围: {df['hot_temp_C'].min():.1f} - {df['hot_temp_C'].max():.1f} °C\n"
            stats_text += f"冷端温度范围: {df['cold_temp_C'].min():.1f} - {df['cold_temp_C'].max():.1f} °C\n"
            stats_text += f"温差范围: {df['temp_diff_C'].min():.1f} - {df['temp_diff_C'].max():.1f} °C\n"
            stats_text += f"平均温差: {df['temp_diff_C'].mean():.1f} °C\n"
            stats_text += f"电压范围: {df['voltage_V'].min():.3f} - {df['voltage_V'].max():.3f} V\n"
            stats_text += f"平均电压: {df['voltage_V'].mean():.3f} V\n"
            
            if 'power_mW' in df.columns:
                stats_text += f"最大功率: {df['power_mW'].max():.2f} mW\n"
                stats_text += f"平均功率: {df['power_mW'].mean():.2f} mW\n"
            
            if 'seebeck_mV_per_C' in df.columns:
                stats_text += f"平均塞贝克系数: {seebeck.mean():.1f} mV/°C\n"
            
            axes[5].text(0.02, 0.98, stats_text, transform=axes[5].transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            return df
            
        except Exception as e:
            print(f"分析文件时出错: {e}")
            return None

# ==================== 主程序 ====================
def main():
    """主程序"""
    print("=" * 60)
    print("TEG温差发电混合分析系统")
    print("版本: 2.0")
    print("=" * 60)
    print()
    
    while True:
        print("请选择操作:")
        print("1. 实时监测模式")
        print("2. 分析已保存的数据")
        print("3. 配置系统参数")
        print("4. 退出程序")
        print()
        
        choice = input("请输入选项 (1-4): ").strip()
        
        if choice == '1':
            # 实时监测模式
            analyzer = TEGHybridAnalyzer()
            
            save_data = input("保存数据到文件? (y/n): ").lower() == 'y'
            
            if analyzer.start(save_data=save_data):
                try:
                    analyzer.create_real_time_dashboard()
                except KeyboardInterrupt:
                    print("\n程序被用户中断")
                finally:
                    analyzer.stop()
                    analyzer.disconnect()
        
        elif choice == '2':
            # 分析模式
            data_dir = "teg_data"
            if not os.path.exists(data_dir):
                print(f"数据目录 '{data_dir}' 不存在")
                continue
            
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if not files:
                print("没有找到数据文件")
                continue
            
            print("\n可用数据文件:")
            for i, f in enumerate(files, 1):
                print(f"{i}. {f}")
            
            try:
                file_idx = int(input("\n选择文件序号: ")) - 1
                if 0 <= file_idx < len(files):
                    filepath = os.path.join(data_dir, files[file_idx])
                    TEGHybridAnalyzer.analyze_saved_file(filepath)
            except (ValueError, IndexError):
                print("无效的选择")
        
        elif choice == '3':
            # 配置模式
            print("\n当前配置:")
            print(f"  采样率: {config.sample_rate} Hz")
            print(f"  负载电阻: {config.load_resistance} Ω")
            print(f"  TEG内阻: {config.teg_resistance} Ω")
            print(f"  塞贝克系数: {config.seebeck_coefficient} mV/°C")
            
            change = input("\n修改配置? (y/n): ").lower()
            if change == 'y':
                try:
                    config.sample_rate = float(input("采样率 (Hz): ") or config.sample_rate)
                    config.load_resistance = float(input("负载电阻 (Ω): ") or config.load_resistance)
                    config.teg_resistance = float(input("TEG内阻 (Ω): ") or config.teg_resistance)
                    config.seebeck_coefficient = float(input("塞贝克系数 (mV/°C): ") or config.seebeck_coefficient)
                    print("配置已更新")
                except ValueError:
                    print("输入无效，使用默认值")
        
        elif choice == '4':
            print("退出程序")
            break
        
        else:
            print("无效的选项")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    # 创建必要目录
    os.makedirs("teg_data", exist_ok=True)
    
    # 运行主程序
    main()