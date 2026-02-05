#include <DHT.h>
#include <DHT_U.h>

// ==================== 配置区域 ====================
#define DHTPIN 2           // DHT11数据引脚连接到Arduino的数字引脚2
#define DHTTYPE DHT11      // 传感器类型为DHT11

// 系统参数
const unsigned long SENSOR_READ_INTERVAL = 2000;  // 传感器读取间隔(毫秒)，DHT11要求至少2秒
const bool ENABLE_SERIAL_DEBUG = true;            // 串口调试信息开关

// ==================== 全局变量 ====================
DHT dht(DHTPIN, DHTTYPE);
unsigned long lastReadTime = 0;

// 数据结构体
struct SensorData {
  unsigned long timestamp;   // 时间戳（毫秒）
  float temperature;         // 温度（℃）
  float humidity;            // 湿度（%）
  float heat_index;          // 体感温度（℃）
  int read_status;           // 读取状态：0=成功，1=失败
};

// ==================== 初始化 ====================
void setup() {
  // 初始化串口通信
  Serial.begin(9600);
  while (!Serial) {
    ; // 等待串口连接（对于某些Arduino板型需要）
  }
  
  // 初始化DHT传感器
  dht.begin();
  
  // 等待传感器稳定
  delay(2000);
  
  // 打印系统信息
  if (ENABLE_SERIAL_DEBUG) {
    Serial.println(F("========================================"));
    Serial.println(F("环境监测系统 - DHT11数据采集终端"));
    Serial.println(F("数据格式: timestamp_ms,temp_c,humidity_pct,heat_index_c,status"));
    Serial.println(F("状态码: 0=读取成功, 1=读取失败"));
    Serial.println(F("========================================"));
  }
}

// ==================== 主循环 ====================
void loop() {
  // 检查是否到达读取间隔时间
  unsigned long currentTime = millis();
  
  if (currentTime - lastReadTime >= SENSOR_READ_INTERVAL) {
    // 更新最后读取时间
    lastReadTime = currentTime;
    
    // 创建数据对象
    SensorData data;
    data.timestamp = currentTime;
    data.read_status = 0; // 默认状态为成功
    
    // 读取传感器数据
    readDHT11Sensor(data);
    
    // 输出格式化数据
    outputFormattedData(data);
  }
  
  // 处理其他任务（如果有）
  // 这里可以添加低功耗休眠代码
}

// ==================== 函数定义 ====================

/**
 * 读取DHT11传感器数据
 * @param data 传感器数据结构体引用
 */
void readDHT11Sensor(SensorData &data) {
  // 读取湿度
  float h = dht.readHumidity();
  
  // 读取温度（摄氏度）
  float t = dht.readTemperature();
  
  // 检查读取是否成功
  if (isnan(h) || isnan(t)) {
    // 读取失败
    data.read_status = 1;
    
    // 使用默认值或上一次的值（此处使用默认值）
    data.temperature = 0.0;
    data.humidity = 0.0;
    data.heat_index = 0.0;
    
    if (ENABLE_SERIAL_DEBUG) {
      Serial.println(F("[警告] DHT11传感器读取失败"));
    }
  } else {
    // 读取成功
    data.temperature = t;
    data.humidity = h;
    
    // 计算体感温度（heat index）
    // 使用DHT库的计算函数，false表示使用摄氏度
    data.heat_index = dht.computeHeatIndex(t, h, false);
  }
}

/**
 * 输出格式化数据（CSV格式）
 * @param data 传感器数据
 */
void outputFormattedData(const SensorData &data) {
  // 输出格式：时间戳(毫秒),温度(℃),湿度(%),体感温度(℃),状态码
  Serial.print(data.timestamp);
  Serial.print(F(","));
  Serial.print(data.temperature, 1);  // 温度保留1位小数
  Serial.print(F(","));
  Serial.print(data.humidity, 1);     // 湿度保留1位小数
  Serial.print(F(","));
  Serial.print(data.heat_index, 1);   // 体感温度保留1位小数
  Serial.print(F(","));
  Serial.println(data.read_status);
  
  // 可选：同时输出可读格式（用于调试）
  if (ENABLE_SERIAL_DEBUG && data.read_status == 0) {
    Serial.print(F("时间: "));
    Serial.print(data.timestamp / 1000); // 转换为秒
    Serial.print(F("s | 温度: "));
    Serial.print(data.temperature);
    Serial.print(F("°C | 湿度: "));
    Serial.print(data.humidity);
    Serial.print(F("% | 体感: "));
    Serial.print(data.heat_index);
    Serial.println(F("°C"));
  }
}

/**
 * 获取传感器状态信息（用于调试）
 */
void printSensorInfo() {
  if (ENABLE_SERIAL_DEBUG) {
    Serial.println(F("\n=== 传感器信息 ==="));
    Serial.print(F("型号: DHT11"));
    Serial.print(F(" | 温度精度: ±2°C"));
    Serial.print(F(" | 湿度精度: ±5%"));
    Serial.print(F(" | 采样间隔: "));
    Serial.print(SENSOR_READ_INTERVAL / 1000.0);
    Serial.println(F("秒"));
    Serial.println(F("========================\n"));
  }
}