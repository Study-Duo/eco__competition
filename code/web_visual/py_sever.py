"""
TEG温差发电混合模式服务器
版本：2.0
功能：串口数据读取、数据处理、WebSocket服务器、REST API
"""

import serial
import serial.tools.list_ports
import time
import json
import threading
import queue
import os
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# ==================== 配置类 ====================
@dataclass
class ServerConfig:
    """服务器配置"""
    port: str = "COM3"           # 串口端口
    baudrate: int = 115200       # 波特率
    buffer_size: int = 1000      # 数据缓冲区大小
    sample_interval: int = 1000  # 采样间隔(ms)
    load_resistance: float = 10.0  # 负载电阻(Ω)
    teg_resistance: float = 4.0    # TEG内阻(Ω)
    seebeck_coefficient: float = 40.0  # 塞贝克系数(mV/°C)

# ==================== 数据类 ====================
@dataclass
class TEGData:
    """TEG数据点"""
    timestamp: float          # 时间戳(ms)
    hot_temp: float          # 热端温度(°C)
    cold_temp: float         # 冷端温度(°C)
    voltage_raw: float       # 原始电压(mV)
    ch1_raw: int             # 通道1原始值
    ch2_raw: int             # 通道2原始值
    ch3_raw: int             # 通道3原始值
    
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
        """电流 (mA)"""
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
        """效率 (%)"""
        if abs(self.temp_diff) > 0.1 and self.hot_temp > 0:
            carnot = 1 - (self.cold_temp + 273.15) / (self.hot_temp + 273.15)
            return (self.power_mw / 1000) / (carnot * 100) * 100 if carnot > 0 else 0.0
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "hot_temp": self.hot_temp,
            "cold_temp": self.cold_temp,
            "temp_diff": self.temp_diff,
            "voltage_raw": self.voltage_raw,
            "voltage": self.voltage_v,
            "current": self.current_ma,
            "power": self.power_mw,
            "seebeck": self.seebeck_mv_per_c,
            "efficiency": self.efficiency,
            "ch1_raw": self.ch1_raw,
            "ch2_raw": self.ch2_raw,
            "ch3_raw": self.ch3_raw
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())

# ==================== 全局配置 ====================
config = ServerConfig()
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==================== 数据管理 ====================
class DataManager:
    """数据管理器"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffer: List[TEGData] = []
        self.raw_data: List[List] = []
        self.lock = threading.Lock()
        self.stats = {
            "data_points": 0,
            "start_time": time.time(),
            "errors": 0,
            "connected": False
        }
        
        # 数据保存
        self.save_enabled = False
        self.csv_file = None
        self.csv_writer = None
        
        # 串口连接
        self.serial_conn = None
        self.serial_thread = None
        self.running = False
    
    # ==================== 串口管理 ====================
    def get_available_ports(self) -> List[Dict[str, str]]:
        """获取可用串口列表"""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "port": port.device,
                "description": port.description,
                "hwid": port.hwid
            })
        return ports
    
    def connect_serial(self, port: str = None, baudrate: int = 115200) -> bool:
        """连接串口"""
        if self.stats["connected"]:
            return True
        
        try:
            port = port or config.port
            print(f"正在连接串口: {port} @ {baudrate}bps")
            
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            time.sleep(2)  # 等待Arduino重启
            self.serial_conn.reset_input_buffer()
            
            # 更新状态
            self.stats["connected"] = True
            self.stats["start_time"] = time.time()
            
            print(f"串口连接成功: {port}")
            return True
            
        except Exception as e:
            print(f"串口连接失败: {e}")
            return False
    
    def disconnect_serial(self):
        """断开串口连接"""
        self.running = False
        
        if self.serial_conn and self.stats["connected"]:
            try:
                self.serial_conn.close()
            except:
                pass
        
        self.stats["connected"] = False
        print("串口已断开")
    
    def start_data_acquisition(self):
        """开始数据采集"""
        if not self.stats["connected"]:
            if not self.connect_serial():
                return False
        
        self.running = True
        
        # 启动数据采集线程
        self.serial_thread = threading.Thread(
            target=self._serial_reader_thread,
            daemon=True
        )
        self.serial_thread.start()
        
        print("数据采集已启动")
        return True
    
    def stop_data_acquisition(self):
        """停止数据采集"""
        print("停止数据采集...")
        self.running = False
        
        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2.0)
        
        self.disconnect_serial()
        print("数据采集已停止")
    
    def _serial_reader_thread(self):
        """串口读取线程"""
        print("串口读取线程启动")
        
        while self.running and self.stats["connected"]:
            try:
                if not self.serial_conn or not self.serial_conn.is_open:
                    break
                
                # 读取数据
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        self._process_data_line(line)
                
                time.sleep(0.01)
                
            except Exception as e:
                self.stats["errors"] += 1
                if self.stats["errors"] % 10 == 0:
                    print(f"串口读取错误: {e}")
                time.sleep(0.1)
        
        print("串口读取线程结束")
    
    def _process_data_line(self, line: str):
        """处理单行数据"""
        try:
            # 跳过标题行和非数据行
            if any(keyword in line.lower() for keyword in ['timestamp', 'time', 'version', '格式']):
                return
            
            # 解析CSV数据
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) >= 7:  # 最少需要7个字段
                # 解析数据
                data_point = TEGData(
                    timestamp=float(parts[0]),
                    hot_temp=float(parts[1]),
                    cold_temp=float(parts[2]),
                    voltage_raw=float(parts[3]),
                    ch1_raw=int(parts[4]),
                    ch2_raw=int(parts[5]),
                    ch3_raw=int(parts[6])
                )
                
                # 添加到缓冲区
                with self.lock:
                    self.data_buffer.append(data_point)
                    self.raw_data.append(parts)
                    
                    # 限制缓冲区大小
                    if len(self.data_buffer) > self.buffer_size:
                        self.data_buffer.pop(0)
                        self.raw_data.pop(0)
                
                self.stats["data_points"] += 1
                
                # 通过WebSocket广播数据
                socketio.emit('new_data', data_point.to_dict())
                
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
                    
                    if self.stats["data_points"] % 20 == 0:
                        self.csv_file.flush()
                        
        except Exception as e:
            self.stats["errors"] += 1
            print(f"数据处理错误: {e}")
    
    # ==================== 数据获取 ====================
    def get_latest_data(self, n_points: int = None) -> List[Dict]:
        """获取最新数据"""
        with self.lock:
            if not self.data_buffer:
                return []
            
            if n_points is None or n_points >= len(self.data_buffer):
                data = self.data_buffer.copy()
            else:
                data = self.data_buffer[-n_points:]
        
        return [d.to_dict() for d in data]
    
    def get_all_data(self) -> List[Dict]:
        """获取所有数据"""
        return self.get_latest_data()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        data = self.get_latest_data()
        
        if not data:
            return {"status": "no_data"}
        
        # 提取数据列
        timestamps = [d["timestamp"] for d in data]
        hot_temps = [d["hot_temp"] for d in data]
        cold_temps = [d["cold_temp"] for d in data]
        temp_diffs = [d["temp_diff"] for d in data]
        voltages = [d["voltage"] for d in data]
        powers = [d["power"] for d in data]
        seebecks = [d["seebeck"] for d in data]
        
        stats = {
            "status": "ok",
            "data_points": len(data),
            "duration": (timestamps[-1] - timestamps[0]) / 1000 if len(timestamps) > 1 else 0,
            "sample_rate": len(data) / max(1, (time.time() - self.stats["start_time"])),
            "hot_temp": {
                "mean": np.mean(hot_temps) if hot_temps else 0,
                "min": np.min(hot_temps) if hot_temps else 0,
                "max": np.max(hot_temps) if hot_temps else 0,
                "std": np.std(hot_temps) if hot_temps else 0
            },
            "cold_temp": {
                "mean": np.mean(cold_temps) if cold_temps else 0,
                "min": np.min(cold_temps) if cold_temps else 0,
                "max": np.max(cold_temps) if cold_temps else 0,
                "std": np.std(cold_temps) if cold_temps else 0
            },
            "temp_diff": {
                "mean": np.mean(temp_diffs) if temp_diffs else 0,
                "min": np.min(temp_diffs) if temp_diffs else 0,
                "max": np.max(temp_diffs) if temp_diffs else 0
            },
            "voltage": {
                "mean": np.mean(voltages) if voltages else 0,
                "min": np.min(voltages) if voltages else 0,
                "max": np.max(voltages) if voltages else 0
            },
            "power": {
                "mean": np.mean(powers) if powers else 0,
                "min": np.min(powers) if powers else 0,
                "max": np.max(powers) if powers else 0
            },
            "seebeck": {
                "mean": np.mean([s for s in seebecks if abs(s) < 1000]) if seebecks else 0
            }
        }
        
        return stats
    
    # ==================== 数据保存 ====================
    def start_data_saving(self, filename: str = None):
        """开始保存数据"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"teg_data_{timestamp}.csv"
        
        # 确保目录存在
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            import csv
            self.csv_file = open(filepath, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # 写入标题
            headers = [
                'timestamp_ms', 'hot_temp_C', 'cold_temp_C', 'voltage_raw_mV',
                'ch1_raw', 'ch2_raw', 'ch3_raw', 'temp_diff_C', 'voltage_V',
                'current_mA', 'power_mW', 'seebeck_mV_per_C', 'efficiency_percent'
            ]
            self.csv_writer.writerow(headers)
            
            self.save_enabled = True
            print(f"数据保存到: {filepath}")
            
        except Exception as e:
            print(f"创建数据文件失败: {e}")
            self.save_enabled = False
    
    def stop_data_saving(self):
        """停止保存数据"""
        self.save_enabled = False
        
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

# ==================== 初始化数据管理器 ====================
data_manager = DataManager(buffer_size=config.buffer_size)

# ==================== WebSocket事件 ====================
@socketio.on('connect')
def handle_connect():
    """客户端连接事件"""
    print(f"客户端连接: {request.sid}")
    emit('connected', {'status': 'connected', 'timestamp': time.time()})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开事件"""
    print(f"客户端断开: {request.sid}")

@socketio.on('request_data')
def handle_request_data(data):
    """处理数据请求"""
    n_points = data.get('n_points', 100)
    data_points = data_manager.get_latest_data(n_points)
    emit('data_response', {'data': data_points})

@socketio.on('start_acquisition')
def handle_start_acquisition():
    """开始数据采集"""
    data_manager.start_data_acquisition()
    emit('acquisition_status', {'status': 'started'})

@socketio.on('stop_acquisition')
def handle_stop_acquisition():
    """停止数据采集"""
    data_manager.stop_data_acquisition()
    emit('acquisition_status', {'status': 'stopped'})

@socketio.on('start_recording')
def handle_start_recording(data):
    """开始记录数据"""
    filename = data.get('filename')
    data_manager.start_data_saving(filename)
    emit('recording_status', {'status': 'started'})

@socketio.on('stop_recording')
def handle_stop_recording():
    """停止记录数据"""
    data_manager.stop_data_saving()
    emit('recording_status', {'status': 'stopped'})

@socketio.on('update_config')
def handle_update_config(data):
    """更新配置"""
    global config
    
    if 'load_resistance' in data:
        config.load_resistance = float(data['load_resistance'])
    
    if 'teg_resistance' in data:
        config.teg_resistance = float(data['teg_resistance'])
    
    if 'seebeck_coefficient' in data:
        config.seebeck_coefficient = float(data['seebeck_coefficient'])
    
    emit('config_updated', {'status': 'success'})

# ==================== REST API ====================
@app.route('/')
def serve_index():
    """服务主页"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """服务静态文件"""
    return send_from_directory('.', filename)

@app.route('/api/status')
def api_status():
    """API状态"""
    return jsonify({
        "status": "running",
        "version": "2.0",
        "timestamp": time.time()
    })

@app.route('/api/data/latest')
def api_data_latest():
    """获取最新数据"""
    n_points = request.args.get('n', 100, type=int)
    data = data_manager.get_latest_data(n_points)
    return jsonify(data)

@app.route('/api/data/all')
def api_data_all():
    """获取所有数据"""
    data = data_manager.get_all_data()
    return jsonify(data)

@app.route('/api/statistics')
def api_statistics():
    """获取统计信息"""
    stats = data_manager.get_statistics()
    return jsonify(stats)

@app.route('/api/ports')
def api_ports():
    """获取可用串口列表"""
    ports = data_manager.get_available_ports()
    return jsonify(ports)

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """连接串口"""
    data = request.json
    port = data.get('port', config.port)
    baudrate = data.get('baudrate', config.baudrate)
    
    success = data_manager.connect_serial(port, baudrate)
    
    return jsonify({
        "status": "success" if success else "error",
        "port": port,
        "baudrate": baudrate
    })

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    """断开串口"""
    data_manager.disconnect_serial()
    
    return jsonify({
        "status": "success"
    })

@app.route('/api/start', methods=['POST'])
def api_start():
    """开始数据采集"""
    success = data_manager.start_data_acquisition()
    
    return jsonify({
        "status": "success" if success else "error"
    })

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """停止数据采集"""
    data_manager.stop_data_acquisition()
    
    return jsonify({
        "status": "success"
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """获取或更新配置"""
    global config
    
    if request.method == 'POST':
        data = request.json
        
        if 'load_resistance' in data:
            config.load_resistance = float(data['load_resistance'])
        
        if 'teg_resistance' in data:
            config.teg_resistance = float(data['teg_resistance'])
        
        if 'seebeck_coefficient' in data:
            config.seebeck_coefficient = float(data['seebeck_coefficient'])
        
        if 'port' in data:
            config.port = data['port']
        
        if 'baudrate' in data:
            config.baudrate = int(data['baudrate'])
    
    return jsonify({
        "load_resistance": config.load_resistance,
        "teg_resistance": config.teg_resistance,
        "seebeck_coefficient": config.seebeck_coefficient,
        "port": config.port,
        "baudrate": config.baudrate
    })

@app.route('/api/saved_files')
def api_saved_files():
    """获取已保存的文件列表"""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        return jsonify([])
    
    files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            stat = os.stat(filepath)
            files.append({
                "name": filename,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "path": filepath
            })
    
    return jsonify(files)

@app.route('/api/export/<filename>')
def api_export(filename):
    """导出数据文件"""
    data_dir = "data"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_from_directory(data_dir, filename, as_attachment=True)

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("TEG温差发电混合模式服务器")
    print("版本: 2.0")
    print("=" * 60)
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    # 自动连接串口
    data_manager.connect_serial()
    
    # 启动服务器
    print(f"服务器启动，访问: http://localhost:5000")
    print("API接口:")
    print("  GET  /api/status          - 服务器状态")
    print("  GET  /api/data/latest     - 最新数据")
    print("  GET  /api/statistics      - 统计信息")
    print("  POST /api/start           - 开始采集")
    print("  POST /api/stop            - 停止采集")
    print("  GET  /api/config          - 获取配置")
    print("  POST /api/config          - 更新配置")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()