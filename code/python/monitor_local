"""
TEG温差发电数据可视化分析程序
功能：
1. 读取CSV格式的TEG测量数据
2. 绘制温差-电压关系图（带温度信息）
3. 绘制时间序列图
4. 统计分析TEG特性
5. 计算塞贝克系数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class TEGVisualizer:
    def __init__(self, data_file=None):
        """
        初始化TEG可视化器
        
        参数:
        data_file: 数据文件路径（CSV格式）
        """
        self.data_file = data_file
        self.df = None
        self.fig = None
        self.ax = None
        self.color_map = cm.get_cmap('viridis')
        
        # 数据统计信息
        self.stats = {}
        
    def load_data(self, data_file=None):
        """
        加载数据文件
        
        参数:
        data_file: 数据文件路径，如果为None则使用self.data_file
        """
        if data_file is not None:
            self.data_file = data_file
            
        if self.data_file is None:
            print("错误: 未指定数据文件")
            return False
            
        try:
            # 读取CSV文件
            self.df = pd.read_csv(self.data_file)
            
            # 检查必要的列
            required_columns = ['热端温度(°C)', '冷端温度(°C)', '温差(°C)', '开路电压(V)']
            for col in required_columns:
                if col not in self.df.columns:
                    print(f"错误: 缺少必要列 '{col}'")
                    return False
            
            # 添加时间列（如果不存在）
            if '时间(ms)' in self.df.columns:
                # 将时间转换为秒
                if self.df['时间(ms)'].iloc[0] > 1e6:  # 可能是微秒
                    self.df['时间(s)'] = (self.df['时间(ms)'] - self.df['时间(ms)'].iloc[0]) / 1000000.0
                else:
                    self.df['时间(s)'] = (self.df['时间(ms)'] - self.df['时间(ms)'].iloc[0]) / 1000.0
            else:
                # 生成时间索引
                self.df['时间(s)'] = np.arange(len(self.df))
            
            # 计算功率（假设负载电阻为10Ω）
            self.df['功率(mW)'] = (self.df['开路电压(V)'] ** 2 / 10) * 1000
            
            # 计算塞贝克系数（瞬时）
            self.df['塞贝克系数(mV/°C)'] = self.df['开路电压(V)'] * 1000 / self.df['温差(°C)'].replace(0, np.nan)
            
            print(f"数据加载成功: {len(self.df)} 个数据点")
            print("数据列:", list(self.df.columns))
            
            # 计算基本统计信息
            self.calculate_statistics()
            
            return True
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
    
    def calculate_statistics(self):
        """计算数据统计信息"""
        if self.df is None or len(self.df) == 0:
            return
        
        self.stats = {
            '数据点数': len(self.df),
            '热端温度范围': (self.df['热端温度(°C)'].min(), self.df['热端温度(°C)'].max()),
            '冷端温度范围': (self.df['冷端温度(°C)'].min(), self.df['冷端温度(°C)'].max()),
            '温差范围': (self.df['温差(°C)'].min(), self.df['温差(°C)'].max()),
            '平均温差': self.df['温差(°C)'].mean(),
            '电压范围': (self.df['开路电压(V)'].min(), self.df['开路电压(V)'].max()),
            '平均电压': self.df['开路电压(V)'].mean(),
            '最大功率': self.df['功率(mW)'].max(),
            '平均功率': self.df['功率(mW)'].mean()
        }
        
        # 线性回归分析（温差-电压）
        temp_diff = self.df['温差(°C)'].values
        voltage = self.df['开路电压(V)'].values
        
        # 移除温差接近0的点（避免除以0）
        mask = np.abs(temp_diff) > 0.1
        if np.sum(mask) > 5:
            slope, intercept, r_value, p_value, std_err = stats.linregress(temp_diff[mask], voltage[mask])
            self.stats.update({
                '塞贝克系数(V/°C)': slope,
                '塞贝克系数(mV/°C)': slope * 1000,
                '截距(V)': intercept,
                '相关系数R': r_value,
                'R平方': r_value**2,
                'P值': p_value,
                '标准误差': std_err
            })
        
        # 打印统计信息
        print("\n=== 数据统计信息 ===")
        for key, value in self.stats.items():
            if isinstance(value, tuple):
                print(f"{key}: {value[0]:.2f} - {value[1]:.2f}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    def plot_comprehensive_analysis(self, save_path=None):
        """
        绘制综合可视化分析图
        
        参数:
        save_path: 保存图片的路径，如果为None则不保存
        """
        if self.df is None:
            print("错误: 没有数据可绘制")
            return
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 设置整体标题
        fig.suptitle('TEG温差发电系统综合分析', fontsize=16, fontweight='bold')
        
        # 创建子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 温差-电压散点图（主要分析图）
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_temp_voltage_scatter(ax1)
        
        # 2. 温差-电压关系（按热端温度着色）
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_temp_voltage_colored(ax2)
        
        # 3. 温度分布图
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_temperature_distribution(ax3)
        
        # 4. 时间序列图：温度
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_temperature_time_series(ax4)
        
        # 5. 时间序列图：电压和功率
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_voltage_power_time_series(ax5)
        
        # 6. 统计信息面板
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_statistics_panel(ax6)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")
        
        plt.show()
    
    def _plot_temp_voltage_scatter(self, ax):
        """绘制温差-电压散点图（主要分析图）"""
        temp_diff = self.df['温差(°C)'].values
        voltage = self.df['开路电压(V)'].values
        hot_temp = self.df['热端温度(°C)'].values
        cold_temp = self.df['冷端温度(°C)'].values
        
        # 绘制散点图，颜色表示热端温度
        scatter = ax.scatter(temp_diff, voltage, c=hot_temp, 
                           cmap='plasma', alpha=0.7, s=50, 
                           edgecolors='k', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, label='热端温度 (°C)')
        
        # 线性拟合
        if len(temp_diff) > 2:
            # 移除温差为0的点
            mask = np.abs(temp_diff) > 0.1
            if np.sum(mask) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(temp_diff[mask], voltage[mask])
                
                # 绘制拟合线
                x_fit = np.linspace(temp_diff.min(), temp_diff.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合: y={slope:.4f}x+{intercept:.4f}')
                
                # 添加拟合信息
                ax.text(0.05, 0.95, f'斜率: {slope:.4f} V/°C\n截距: {intercept:.4f} V\nR² = {r_value**2:.4f}',
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('温差 (°C)', fontsize=12)
        ax.set_ylabel('开路电压 (V)', fontsize=12)
        ax.set_title('温差-电压特性分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        ax.legend(loc='lower right')
    
    def _plot_temp_voltage_colored(self, ax):
        """绘制温差-电压关系（按热端温度着色）的增强图"""
        temp_diff = self.df['温差(°C)'].values
        voltage = self.df['开路电压(V)'].values
        hot_temp = self.df['热端温度(°C)'].values
        cold_temp = self.df['冷端温度(°C)'].values
        
        # 计算每个数据点的平均温度（用于着色）
        avg_temp = (hot_temp + cold_temp) / 2
        
        # 绘制散点图，颜色表示平均温度，大小表示温差大小
        sizes = np.abs(temp_diff) * 5 + 10  # 温差越大，点越大
        
        scatter = ax.scatter(temp_diff, voltage, c=avg_temp, 
                           cmap='coolwarm', alpha=0.8, s=sizes,
                           edgecolors='k', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, label='平均温度 (°C)')
        
        # 标记最大功率点
        max_power_idx = self.df['功率(mW)'].idxmax()
        ax.scatter(temp_diff[max_power_idx], voltage[max_power_idx], 
                  color='red', s=200, marker='*', edgecolors='k', 
                  linewidth=2, label='最大功率点')
        
        # 标记最大温差点
        max_diff_idx = self.df['温差(°C)'].idxmax()
        ax.scatter(temp_diff[max_diff_idx], voltage[max_diff_idx], 
                  color='green', s=150, marker='^', edgecolors='k', 
                  linewidth=2, label='最大温差点')
        
        ax.set_xlabel('温差 (°C)', fontsize=12)
        ax.set_ylabel('开路电压 (V)', fontsize=12)
        ax.set_title('温度分布与电压关系', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    def _plot_temperature_distribution(self, ax):
        """绘制温度分布图"""
        hot_temp = self.df['热端温度(°C)'].values
        cold_temp = self.df['冷端温度(°C)'].values
        
        # 绘制箱线图
        data_to_plot = [hot_temp, cold_temp]
        bp = ax.boxplot(data_to_plot, labels=['热端', '冷端'], patch_artist=True)
        
        # 设置颜色
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加散点图显示数据分布
        for i, data in enumerate(data_to_plot, 1):
            # 添加一些抖动
            y = data
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, color=colors[i-1], s=20)
        
        ax.set_ylabel('温度 (°C)', fontsize=12)
        ax.set_title('温度分布统计', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加平均值线
        for i, data in enumerate(data_to_plot, 1):
            ax.axhline(np.mean(data), color=colors[i-1], linestyle='--', alpha=0.7, 
                      xmin=0.1 + (i-1)*0.4, xmax=0.5 + (i-1)*0.4, linewidth=2)
    
    def _plot_temperature_time_series(self, ax):
        """绘制温度时间序列图"""
        if '时间(s)' not in self.df.columns:
            time = np.arange(len(self.df))
        else:
            time = self.df['时间(s)'].values
        
        hot_temp = self.df['热端温度(°C)'].values
        cold_temp = self.df['冷端温度(°C)'].values
        
        # 绘制温度曲线
        ax.plot(time, hot_temp, 'r-', linewidth=2, label='热端温度')
        ax.plot(time, cold_temp, 'b-', linewidth=2, label='冷端温度')
        
        # 填充温差区域
        ax.fill_between(time, cold_temp, hot_temp, alpha=0.2, color='purple', label='温差区域')
        
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('温度 (°C)', fontsize=12)
        ax.set_title('温度时间序列', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 添加温差曲线
        ax_temp = ax.twinx()
        temp_diff = self.df['温差(°C)'].values
        ax_temp.plot(time, temp_diff, 'g--', linewidth=1.5, alpha=0.7, label='温差')
        ax_temp.set_ylabel('温差 (°C)', fontsize=12, color='green')
        ax_temp.tick_params(axis='y', labelcolor='green')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_temp.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    def _plot_voltage_power_time_series(self, ax):
        """绘制电压和功率时间序列图"""
        if '时间(s)' not in self.df.columns:
            time = np.arange(len(self.df))
        else:
            time = self.df['时间(s)'].values
        
        voltage = self.df['开路电压(V)'].values
        
        # 绘制电压曲线
        ax.plot(time, voltage, 'orange', linewidth=2, label='开路电压')
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel('开路电压 (V)', fontsize=12, color='orange')
        ax.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, alpha=0.3)
        
        # 添加功率曲线（次坐标轴）
        ax_power = ax.twinx()
        if '功率(mW)' in self.df.columns:
            power = self.df['功率(mW)'].values
            ax_power.plot(time, power, 'purple', linewidth=2, label='输出功率')
            ax_power.set_ylabel('输出功率 (mW)', fontsize=12, color='purple')
            ax_power.tick_params(axis='y', labelcolor='purple')
        
        # 添加温差曲线（第三个坐标轴）
        ax_diff = ax.twinx()
        ax_diff.spines['right'].set_position(('outward', 60))
        temp_diff = self.df['温差(°C)'].values
        ax_diff.plot(time, temp_diff, 'g--', linewidth=1.5, alpha=0.7, label='温差')
        ax_diff.set_ylabel('温差 (°C)', fontsize=12, color='green')
        ax_diff.tick_params(axis='y', labelcolor='green')
        
        # 合并图例
        lines = []
        labels = []
        for ax_i in [ax, ax_power, ax_diff]:
            line, label = ax_i.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        
        ax.set_title('电压、功率与温差时间序列', fontsize=14, fontweight='bold')
        ax.legend(lines, labels, loc='upper right')
    
    def _plot_statistics_panel(self, ax):
        """绘制统计信息面板"""
        # 清空坐标轴
        ax.axis('off')
        
        # 创建文本内容
        stats_text = "=== TEG系统性能统计 ===\n\n"
        
        # 添加基本信息
        stats_text += f"数据点数: {self.stats.get('数据点数', 'N/A')}\n"
        stats_text += f"平均温差: {self.stats.get('平均温差', 0):.2f} °C\n"
        stats_text += f"平均电压: {self.stats.get('平均电压', 0):.4f} V\n"
        
        # 添加温度范围
        temp_range = self.stats.get('热端温度范围', (0, 0))
        stats_text += f"热端温度范围: {temp_range[0]:.1f} - {temp_range[1]:.1f} °C\n"
        
        temp_range = self.stats.get('冷端温度范围', (0, 0))
        stats_text += f"冷端温度范围: {temp_range[0]:.1f} - {temp_range[1]:.1f} °C\n"
        
        diff_range = self.stats.get('温差范围', (0, 0))
        stats_text += f"温差范围: {diff_range[0]:.2f} - {diff_range[1]:.2f} °C\n"
        
        # 添加电压范围
        volt_range = self.stats.get('电压范围', (0, 0))
        stats_text += f"电压范围: {volt_range[0]:.4f} - {volt_range[1]:.4f} V\n\n"
        
        # 添加功率信息
        stats_text += f"最大输出功率: {self.stats.get('最大功率', 0):.2f} mW\n"
        stats_text += f"平均输出功率: {self.stats.get('平均功率', 0):.2f} mW\n\n"
        
        # 添加塞贝克系数信息
        if '塞贝克系数(mV/°C)' in self.stats:
            stats_text += f"塞贝克系数: {self.stats['塞贝克系数(mV/°C)']:.2f} mV/°C\n"
        
        if 'R平方' in self.stats:
            stats_text += f"拟合优度 (R²): {self.stats['R平方']:.4f}\n"
        
        # 在面板中显示文本
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def plot_3d_temperature_voltage(self, save_path=None):
        """
        绘制三维温度-电压关系图
        
        参数:
        save_path: 保存图片的路径
        """
        if self.df is None:
            print("错误: 没有数据可绘制")
            return
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备数据
        hot_temp = self.df['热端温度(°C)'].values
        cold_temp = self.df['冷端温度(°C)'].values
        temp_diff = self.df['温差(°C)'].values
        voltage = self.df['开路电压(V)'].values
        
        # 绘制3D散点图
        scatter = ax.scatter(hot_temp, cold_temp, voltage, 
                            c=temp_diff, cmap='plasma', 
                            s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('温差 (°C)', fontsize=12)
        
        # 设置标签
        ax.set_xlabel('热端温度 (°C)', fontsize=12, labelpad=10)
        ax.set_ylabel('冷端温度 (°C)', fontsize=12, labelpad=10)
        ax.set_zlabel('开路电压 (V)', fontsize=12, labelpad=10)
        ax.set_title('三维温度-电压关系图', fontsize=16, fontweight='bold', pad=20)
        
        # 调整视角
        ax.view_init(elev=20, azim=45)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D图形已保存到: {save_path}")
        
        plt.show()
    
    def plot_seebeck_analysis(self, save_path=None):
        """
        绘制塞贝克系数分析图
        
        参数:
        save_path: 保存图片的路径
        """
        if self.df is None:
            print("错误: 没有数据可绘制")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('塞贝克系数分析', fontsize=16, fontweight='bold')
        
        # 1. 温差-电压散点图（分区间）
        ax1 = axes[0, 0]
        temp_diff = self.df['温差(°C)'].values
        voltage = self.df['开路电压(V)'].values
        
        # 按温差分区间
        n_bins = 5
        temp_bins = np.linspace(temp_diff.min(), temp_diff.max(), n_bins+1)
        
        for i in range(n_bins):
            mask = (temp_diff >= temp_bins[i]) & (temp_diff < temp_bins[i+1])
            if i == n_bins-1:  # 包括最后一个边界点
                mask = (temp_diff >= temp_bins[i]) & (temp_diff <= temp_bins[i+1])
            
            if np.sum(mask) > 0:
                # 计算每个区间的平均温差和电压
                avg_temp = np.mean(temp_diff[mask])
                avg_voltage = np.mean(voltage[mask])
                
                # 计算每个区间的塞贝克系数
                seebeck = avg_voltage * 1000 / avg_temp if avg_temp != 0 else 0
                
                # 绘制区间点
                ax1.scatter(avg_temp, avg_voltage, s=100, 
                           color=self.color_map(i/n_bins), 
                           edgecolors='k', linewidth=1.5,
                           label=f'区间{i+1}: {seebeck:.1f} mV/°C')
        
        ax1.set_xlabel('温差 (°C)', fontsize=12)
        ax1.set_ylabel('开路电压 (V)', fontsize=12)
        ax1.set_title('分区间塞贝克系数分析', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # 2. 塞贝克系数分布直方图
        ax2 = axes[0, 1]
        if '塞贝克系数(mV/°C)' in self.df.columns:
            seebeck_values = self.df['塞贝克系数(mV/°C)'].dropna().values
            
            # 绘制直方图
            n, bins, patches = ax2.hist(seebeck_values, bins=20, 
                                        alpha=0.7, color='skyblue', 
                                        edgecolor='black')
            
            # 添加平均值线
            mean_seebeck = np.mean(seebeck_values)
            ax2.axvline(mean_seebeck, color='red', linestyle='--', 
                       linewidth=2, label=f'平均值: {mean_seebeck:.2f} mV/°C')
            
            ax2.set_xlabel('塞贝克系数 (mV/°C)', fontsize=12)
            ax2.set_ylabel('频数', fontsize=12)
            ax2.set_title('塞贝克系数分布', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. 塞贝克系数随温差变化
        ax3 = axes[1, 0]
        
        # 使用滑动窗口计算平均塞贝克系数
        window_size = min(20, len(temp_diff) // 10)
        if window_size > 1:
            seebeck_smooth = self._moving_average(seebeck_values, window_size)
            temp_diff_smooth = self._moving_average(temp_diff, window_size)
            
            ax3.plot(temp_diff_smooth, seebeck_smooth, 'b-', linewidth=2)
            ax3.fill_between(temp_diff_smooth, 
                            seebeck_smooth - np.std(seebeck_smooth)/2,
                            seebeck_smooth + np.std(seebeck_smooth)/2,
                            alpha=0.2, color='blue')
        
        ax3.set_xlabel('温差 (°C)', fontsize=12)
        ax3.set_ylabel('塞贝克系数 (mV/°C)', fontsize=12)
        ax3.set_title('塞贝克系数随温差变化', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能指标对比
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 计算性能指标
        performance_metrics = {
            '平均塞贝克系数': f"{self.stats.get('塞贝克系数(mV/°C)', 0):.2f} mV/°C",
            '最大输出功率': f"{self.stats.get('最大功率', 0):.2f} mW",
            '电压温差比': f"{self.stats.get('平均电压', 0) / max(self.stats.get('平均温差', 1), 0.1) * 1000:.2f} mV/°C",
            '线性拟合R²': f"{self.stats.get('R平方', 0):.4f}",
            '温度利用效率': f"{self.stats.get('平均电压', 0) / max(self.stats.get('热端温度范围', (0, 1))[1], 1) * 1000:.2f} mV/°C"
        }
        
        # 显示性能指标
        metrics_text = "=== TEG性能指标 ===\n\n"
        for key, value in performance_metrics.items():
            metrics_text += f"{key}: {value}\n"
        
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"塞贝克分析图已保存到: {save_path}")
        
        plt.show()
    
    def _moving_average(self, data, window_size):
        """计算移动平均值"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def export_analysis_report(self, output_file=None):
        """
        导出分析报告
        
        参数:
        output_file: 输出文件路径，如果为None则生成默认文件名
        """
        if self.df is None:
            print("错误: 没有数据可分析")
            return
        
        if output_file is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"TEG_分析报告_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("TEG温差发电系统分析报告\n")
                f.write("=" * 60 + "\n\n")
                
                # 基本信息
                f.write("一、基本信息\n")
                f.write("-" * 40 + "\n")
                f.write(f"数据文件: {self.data_file}\n")
                f.write(f"数据点数: {len(self.df)}\n")
                f.write(f"数据采集时间: {self.df['时间(s)'].iloc[-1]:.1f} 秒\n\n")
                
                # 温度统计
                f.write("二、温度统计\n")
                f.write("-" * 40 + "\n")
                hot_min, hot_max = self.stats.get('热端温度范围', (0, 0))
                cold_min, cold_max = self.stats.get('冷端温度范围', (0, 0))
                diff_min, diff_max = self.stats.get('温差范围', (0, 0))
                
                f.write(f"热端温度: {hot_min:.2f} - {hot_max:.2f} °C (平均: {self.df['热端温度(°C)'].mean():.2f} °C)\n")
                f.write(f"冷端温度: {cold_min:.2f} - {cold_max:.2f} °C (平均: {self.df['冷端温度(°C)'].mean():.2f} °C)\n")
                f.write(f"温差: {diff_min:.2f} - {diff_max:.2f} °C (平均: {self.stats.get('平均温差', 0):.2f} °C)\n\n")
                
                # 电压统计
                f.write("三、电压与功率统计\n")
                f.write("-" * 40 + "\n")
                volt_min, volt_max = self.stats.get('电压范围', (0, 0))
                
                f.write(f"开路电压: {volt_min:.4f} - {volt_max:.4f} V (平均: {self.stats.get('平均电压', 0):.4f} V)\n")
                f.write(f"最大输出功率: {self.stats.get('最大功率', 0):.2f} mW (负载10Ω)\n")
                f.write(f"平均输出功率: {self.stats.get('平均功率', 0):.2f} mW\n\n")
                
                # 塞贝克系数分析
                f.write("四、塞贝克系数分析\n")
                f.write("-" * 40 + "\n")
                
                if '塞贝克系数(V/°C)' in self.stats:
                    f.write(f"线性回归斜率: {self.stats['塞贝克系数(V/°C)']:.6f} V/°C\n")
                    f.write(f"塞贝克系数: {self.stats['塞贝克系数(mV/°C)']:.2f} mV/°C\n")
                    f.write(f"截距: {self.stats['截距(V)']:.6f} V\n")
                    f.write(f"相关系数R: {self.stats['相关系数R']:.6f}\n")
                    f.write(f"拟合优度R²: {self.stats['R平方']:.6f}\n")
                    f.write(f"P值: {self.stats['P值']:.6f}\n")
                    f.write(f"标准误差: {self.stats['标准误差']:.6f}\n\n")
                
                # 性能评估
                f.write("五、性能评估\n")
                f.write("-" * 40 + "\n")
                
                # 计算温度利用效率
                if hot_max > 0:
                    temp_utilization = self.stats.get('平均电压', 0) / hot_max * 1000
                    f.write(f"温度利用效率: {temp_utilization:.2f} mV/°C (每°C热端温度产生的mV)\n")
                
                # 计算功率密度（假设TEG面积1cm²）
                teg_area = 1.0  # cm²
                power_density = self.stats.get('最大功率', 0) / teg_area
                f.write(f"最大功率密度: {power_density:.2f} mW/cm² (假设TEG面积1cm²)\n")
                
                # 温差稳定性
                temp_diff_std = self.df['温差(°C)'].std()
                f.write(f"温差稳定性: 标准差 {temp_diff_std:.2f} °C (越小越稳定)\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("报告生成完成\n")
                f.write("=" * 60 + "\n")
            
            print(f"分析报告已保存到: {output_file}")
            
        except Exception as e:
            print(f"导出报告时出错: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("TEG温差发电数据可视化分析系统")
    print("=" * 60)
    
    # 创建数据目录
    data_dir = "teg_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 查找数据文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"在 {data_dir} 目录中没有找到CSV文件")
        print("请确保Arduino数据文件已保存到该目录")
        return
    
    print(f"在 {data_dir} 目录中找到以下CSV文件:")
    for i, f in enumerate(csv_files, 1):
        print(f"{i}. {f}")
    
    # 选择文件
    try:
        choice = int(input("\n请选择要分析的文件编号 (1-{}): ".format(len(csv_files))))
        if choice < 1 or choice > len(csv_files):
            print("无效的选择")
            return
    except ValueError:
        print("请输入有效的数字")
        return
    
    selected_file = os.path.join(data_dir, csv_files[choice-1])
    print(f"选择文件: {selected_file}")
    
    # 创建可视化器
    visualizer = TEGVisualizer(selected_file)
    
    # 加载数据
    if not visualizer.load_data():
        print("数据加载失败，请检查文件格式")
        return
    
    # 交互式分析菜单
    while True:
        print("\n" + "=" * 60)
        print("请选择分析功能:")
        print("1. 综合可视化分析")
        print("2. 3D温度-电压关系图")
        print("3. 塞贝克系数分析")
        print("4. 导出分析报告")
        print("5. 重新选择数据文件")
        print("6. 退出")
        
        try:
            option = int(input("请输入选项 (1-6): "))
        except ValueError:
            print("请输入有效的数字")
            continue
        
        if option == 1:
            # 综合可视化分析
            output_file = os.path.join(data_dir, f"综合分析_{csv_files[choice-1].replace('.csv', '.png')}")
            visualizer.plot_comprehensive_analysis(save_path=output_file)
            
        elif option == 2:
            # 3D温度-电压关系图
            output_file = os.path.join(data_dir, f"3D分析_{csv_files[choice-1].replace('.csv', '.png')}")
            visualizer.plot_3d_temperature_voltage(save_path=output_file)
            
        elif option == 3:
            # 塞贝克系数分析
            output_file = os.path.join(data_dir, f"塞贝克分析_{csv_files[choice-1].replace('.csv', '.png')}")
            visualizer.plot_seebeck_analysis(save_path=output_file)
            
        elif option == 4:
            # 导出分析报告
            report_file = os.path.join(data_dir, f"分析报告_{csv_files[choice-1].replace('.csv', '.txt')}")
            visualizer.export_analysis_report(output_file=report_file)
            
        elif option == 5:
            # 重新选择数据文件
            return main()  # 递归调用主函数
            
        elif option == 6:
            print("退出程序")
            break
            
        else:
            print("无效的选项")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")