#!/usr/bin/env python3
"""
节能减排竞赛 - DHT11数据可视化分析系统
功能：实时接收DHT11串口数据，保存为CSV，生成专业可视化图表
"""

import serial
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import threading
import queue
import time
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
class Config:
    # 串口配置 (根据实际情况修改)
    # Windows: COM3, COM4... | Linux/Mac: /dev/ttyUSB0, /dev/ttyACM0...
    SERIAL_PORT = 'COM4'
    BAUD_RATE = 9600
    SERIAL_TIMEOUT = 2
    
    # 文件配置
    CSV_FILENAME = 'environment_monitor_data.csv'
    FIGURE_SAVE_PATH = 'competition_charts/'
    
    # 可视化配置
    UPDATE_INTERVAL_MS = 2000        # 图表更新间隔（毫秒）
    MAX_DISPLAY_POINTS = 100         # 图表显示的最大数据点数
    FIGURE_SIZE = (16, 12)           # 图表尺寸
    
    # 颜色配置
    COLOR_TEMP = '#FF6B6B'           # 温度颜色（红色）
    COLOR_HUMID = '#4ECDC4'          # 湿度颜色（青色）
    COLOR_HEAT_INDEX = '#FFA726'     # 体感温度颜色（橙色）
    COLOR_SUCCESS = '#66BB6A'        # 成功状态（绿色）
    COLOR_ERROR = '#EF5350'          # 错误状态（红色）
    
    # 系统配置
    ENABLE_REALTIME_PLOT = True      # 启用实时图表
    AUTO_SAVE_INTERVAL_MIN = 5       # 自动保存间隔（分钟）

# ==================== 数据管理类 ====================
class DataManager:
    def __init__(self, filename):
        self.filename = filename
        self.initialize_csv()
        
    def initialize_csv(self):
        """初始化CSV文件，写入表头"""
        try:
            # 如果目录不存在则创建
            os.makedirs(os.path.dirname(self.filename) if os.path.dirname(self.filename) else '.', exist_ok=True)
            
            # 检查文件是否存在，不存在则创建并写入表头
            if not os.path.exists(self.filename):
                with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Timestamp(ms)', 'Local_Time', 'Temperature(°C)', 
                        'Humidity(%)', 'Heat_Index(°C)', 'Read_Status',
                        'Data_Quality', 'Hour_of_Day'
                    ])
                print(f"✓ 已创建新的数据文件: {self.filename}")
            else:
                print(f"✓ 使用现有数据文件: {self.filename}")
                
        except Exception as e:
            print(f"✗ 初始化CSV文件失败: {e}")
    
    def save_data(self, data_dict):
        """保存单行数据到CSV"""
        try:
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    data_dict['timestamp'],
                    data_dict['local_time'],
                    data_dict['temperature'],
                    data_dict['humidity'],
                    data_dict['heat_index'],
                    data_dict['read_status'],
                    data_dict['data_quality'],
                    data_dict['hour_of_day']
                ])
            return True
        except Exception as e:
            print(f"✗ 保存数据时出错: {e}")
            return False
    
    def load_recent_data(self, limit=100):
        """加载最近的N条数据"""
        try:
            df = pd.read_csv(self.filename)
            if len(df) > limit:
                df = df.tail(limit)
            return df
        except Exception as e:
            print(f"✗ 加载数据失败: {e}")
            return pd.DataFrame()

# ==================== 串口通信类 ====================
class SerialReader:
    def __init__(self, port, baud_rate, timeout):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.data_queue = queue.Queue()
        self.is_running = False
        
    def start(self):
        """启动串口连接"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            time.sleep(2)  # 等待串口稳定
            print(f"✓ 串口连接成功: {self.port}")
            self.is_running = True
            return True
        except Exception as e:
            print(f"✗ 串口连接失败: {e}")
            print("请检查:")
            print("1. 串口号是否正确 (Windows: COM3, COM4...)")
            print("2. Arduino是否已连接")
            print("3. 其他程序是否占用了串口")
            return False
    
    def read_data(self):
        """读取串口数据（主线程调用）"""
        if not self.serial_conn or not self.is_running:
            return None
        
        try:
            # 读取一行数据
            line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            
            if line and not line.startswith("时间:") and not line.startswith("==="):
                # 解析数据：时间戳,温度,湿度,体感温度,状态码
                parts = line.split(',')
                if len(parts) >= 5:
                    return {
                        'raw': line,
                        'timestamp': int(parts[0]),
                        'temperature': float(parts[1]),
                        'humidity': float(parts[2]),
                        'heat_index': float(parts[3]),
                        'read_status': int(parts[4])
                    }
            return None
        except Exception as e:
            print(f"✗ 读取串口数据出错: {e}")
            return None
    
    def stop(self):
        """停止串口连接"""
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("✓ 串口连接已关闭")

# ==================== 数据处理与可视化类 ====================
class DataVisualizer:
    def __init__(self, data_manager, serial_reader):
        self.data_manager = data_manager
        self.serial_reader = serial_reader
        self.data_history = []
        self.fig = None
        self.axs = None
        self.start_time = time.time()
        
    def create_dashboard(self):
        """创建监控仪表盘"""
        self.fig = plt.figure(figsize=Config.FIGURE_SIZE, facecolor='#f5f5f5')
        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 创建子图
        self.ax1 = self.fig.add_subplot(gs[0, :])  # 温度时间序列
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # 湿度时间序列
        self.ax3 = self.fig.add_subplot(gs[1, 1])  # 体感温度时间序列
        self.ax4 = self.fig.add_subplot(gs[2, 0])  # 温湿度散点图
        self.ax5 = self.fig.add_subplot(gs[2, 1])  # 状态统计
        
        # 设置图表样式
        self.setup_plot_styles()
        
        return self.fig
    
    def setup_plot_styles(self):
        """设置图表样式"""
        # 统一设置字体
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'Microsoft YaHei' if sys.platform == 'win32' else 'DejaVu Sans'
        })
        
        # 设置标题
        self.fig.suptitle('环境监测系统 - 实时数据仪表盘\n(DHT11温湿度传感器)', 
                        fontsize=16, fontweight='bold', y=0.98)
    
    def update_dashboard(self, frame):
        """更新仪表盘数据"""
        # 读取最新数据
        data = self.serial_reader.read_data()
        
        if data:
            # 添加本地时间信息
            local_time = datetime.now()
            data['local_time'] = local_time
            data['hour_of_day'] = local_time.hour
            data['data_quality'] = 'Good' if data['read_status'] == 0 else 'Poor'
            
            # 保存到CSV
            self.data_manager.save_data(data)
            
            # 添加到历史数据
            self.data_history.append(data)
            
            # 限制历史数据长度
            if len(self.data_history) > Config.MAX_DISPLAY_POINTS:
                self.data_history.pop(0)
            
            # 更新所有图表
            self.update_all_plots()
        
        return []
    
    def update_all_plots(self):
        """更新所有子图"""
        if not self.data_history:
            return
        
        # 准备数据
        timestamps = [d['timestamp']/1000 for d in self.data_history]  # 转换为秒
        temps = [d['temperature'] for d in self.data_history]
        humids = [d['humidity'] for d in self.data_history]
        heat_indices = [d['heat_index'] for d in self.data_history]
        statuses = [d['read_status'] for d in self.data_history]
        
        # 清空所有图表
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.clear()
        
        # 1. 温度时间序列图
        self.ax1.plot(timestamps, temps, color=Config.COLOR_TEMP, linewidth=2, marker='o', markersize=4)
        self.ax1.fill_between(timestamps, temps, alpha=0.3, color=Config.COLOR_TEMP)
        self.ax1.set_xlabel('时间 (秒)', fontsize=11)
        self.ax1.set_ylabel('温度 (°C)', fontsize=11, color=Config.COLOR_TEMP)
        self.ax1.set_title('温度变化趋势', fontsize=13, fontweight='bold')
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        self.ax1.tick_params(axis='y', labelcolor=Config.COLOR_TEMP)
        
        # 添加最新值标注
        if temps:
            self.ax1.annotate(f'{temps[-1]:.1f}°C', 
                                xy=(timestamps[-1], temps[-1]),
                                xytext=(10, 0), textcoords='offset points',
                                fontsize=12, fontweight='bold',
                                color=Config.COLOR_TEMP,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 2. 湿度时间序列图
        self.ax2.plot(timestamps, humids, color=Config.COLOR_HUMID, linewidth=2, marker='s', markersize=4)
        self.ax2.set_xlabel('时间 (秒)', fontsize=11)
        self.ax2.set_ylabel('湿度 (%)', fontsize=11, color=Config.COLOR_HUMID)
        self.ax2.set_title('湿度变化趋势', fontsize=13, fontweight='bold')
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        self.ax2.tick_params(axis='y', labelcolor=Config.COLOR_HUMID)
        
        # 添加舒适度区域
        self.ax2.axhspan(40, 60, alpha=0.2, color='green', label='舒适范围')
        
        # 3. 体感温度时间序列图
        self.ax3.plot(timestamps, heat_indices, color=Config.COLOR_HEAT_INDEX, linewidth=2, marker='^', markersize=4)
        self.ax3.set_xlabel('时间 (秒)', fontsize=11)
        self.ax3.set_ylabel('体感温度 (°C)', fontsize=11, color=Config.COLOR_HEAT_INDEX)
        self.ax3.set_title('体感温度变化', fontsize=13, fontweight='bold')
        self.ax3.grid(True, alpha=0.3, linestyle='--')
        self.ax3.tick_params(axis='y', labelcolor=Config.COLOR_HEAT_INDEX)
        
        # 4. 温湿度散点图
        scatter = self.ax4.scatter(temps, humids, c=heat_indices, 
                                    cmap='coolwarm', s=50, alpha=0.7, 
                                    edgecolors='black', linewidth=0.5)
        self.ax4.set_xlabel('温度 (°C)', fontsize=11)
        self.ax4.set_ylabel('湿度 (%)', fontsize=11)
        self.ax4.set_title('温湿度相关性分析', fontsize=13, fontweight='bold')
        self.ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=self.ax4)
        cbar.set_label('体感温度 (°C)', fontsize=11)
        
        # 5. 数据质量统计图
        success_count = statuses.count(0)
        error_count = statuses.count(1)
        total_count = len(statuses)
        
        if total_count > 0:
            labels = ['成功', '失败']
            sizes = [success_count, error_count]
            colors = [Config.COLOR_SUCCESS, Config.COLOR_ERROR]
            
            self.ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, shadow=True, explode=(0.05, 0))
            self.ax5.set_title(f'数据采集质量\n(总计: {total_count}次)', fontsize=13, fontweight='bold')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("节能减排竞赛 - DHT11环境监测数据可视化系统")
    print("=" * 60)
    
    # 创建数据保存目录
    if not os.path.exists(Config.FIGURE_SAVE_PATH):
        os.makedirs(Config.FIGURE_SAVE_PATH)
    
    # 初始化数据管理器
    data_manager = DataManager(Config.CSV_FILENAME)
    
    # 初始化串口读取器
    serial_reader = SerialReader(
        port=Config.SERIAL_PORT,
        baud_rate=Config.BAUD_RATE,
        timeout=Config.SERIAL_TIMEOUT
    )
    
    # 尝试连接串口
    if not serial_reader.start():
        print("程序退出。")
        return
    
    # 初始化可视化器
    visualizer = DataVisualizer(data_manager, serial_reader)
    
    try:
        if Config.ENABLE_REALTIME_PLOT:
            # 创建实时仪表盘
            fig = visualizer.create_dashboard()
            
            # 创建动画
            ani = FuncAnimation(
                fig, 
                visualizer.update_dashboard,
                interval=Config.UPDATE_INTERVAL_MS,
                blit=False,
                cache_frame_data=False
            )
            
            print("\n" + "=" * 60)
            print("实时监控仪表盘已启动")
            print("正在接收数据并保存到:", Config.CSV_FILENAME)
            print("关闭图表窗口将停止程序")
            print("=" * 60 + "\n")
            
            plt.show()
            
        else:
            # 仅记录数据模式
            print("\n开始记录数据... (按Ctrl+C停止)")
            record_count = 0
            
            try:
                while True:
                    data = serial_reader.read_data()
                    if data:
                        local_time = datetime.now()
                        data['local_time'] = local_time
                        data['hour_of_day'] = local_time.hour
                        data['data_quality'] = 'Good' if data['read_status'] == 0 else 'Poor'
                        
                        if data_manager.save_data(data):
                            record_count += 1
                            if record_count % 10 == 0:
                                print(f"✓ 已记录 {record_count} 条数据...")
                    
                    time.sleep(2)  # 与Arduino读取间隔同步
                    
            except KeyboardInterrupt:
                print(f"\n数据记录完成，总计 {record_count} 条数据")
                
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 清理资源
        serial_reader.stop()
        
        # 保存最终图表
        if Config.ENABLE_REALTIME_PLOT and visualizer.data_history:
            final_chart_path = os.path.join(Config.FIGURE_SAVE_PATH, 'final_analysis.png')
            visualizer.fig.savefig(final_chart_path, dpi=300, bbox_inches='tight')
            print(f"✓ 最终图表已保存到: {final_chart_path}")
        
        print("\n程序正常退出")
        print("生成的数据文件可用于:")
        print("1. 竞赛报告中的数据支撑")
        print("2. 系统稳定性分析")
        print("3. 环境变化规律研究")

# ==================== 运行主程序 ====================
if __name__ == "__main__":
    main()