/*接线说明
DS18B20引脚 -> Arduino Nano引脚
1. GND (黑色) -> GND
2. DQ  (黄色) -> D2 (数字引脚2，可自定义)
3. VDD (红色) -> 5V

ADS1115引脚 -> Arduino Nano引脚
VDD  (电源正) -> 5V
GND  (地)     -> GND
SCL  (时钟线) -> A5 (或标记为SCL的引脚)
SDA  (数据线) -> A4 (或标记为SDA的引脚)
ADDR (地址)   -> GND (设置I2C地址为0x48)
A0   (通道0)  -> TEG正极 (通过分压电阻)
A1   (通道1)  -> TEG负极
A2   (通道2)  -> 预留 (可选接其他传感器)
A3   (通道3)  -> 预留 (可选接其他传感器)

标准分压电路：
TEG+ ─── R1 ───┬── A0 (ADS1115)
                │
               R2
                │
TEG- ───────────┴── A1 (ADS1115)

分压比 = R2 / (R1 + R2)，分压系数voltageDividerRatio的倒数
测量电压 = 实际电压 × 分压比
实际电压 = 测量电压 ÷ 分压比
*/
// ==================== 库文件引入 ====================
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ==================== 引脚定义 ====================
// DS18B20引脚
#define ONE_WIRE_BUS 2

// ADS1115预留定义
// 使用I2C通信，引脚为A4(SDA)和A5(SCL)

// ==================== 对象声明 ====================
// DS18B20相关
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ADS1115相关
Adafruit_ADS1115 ads;  // 使用16位ADC

// ==================== 全局变量声明 ====================
// 温度传感器地址
DeviceAddress hotThermometer, coldThermometer;
bool sensorsFound = false;

// TEG测量变量
float tegVoltage = 0.0;          // TEG开路电压 (V)
float temperatureDiff = 0.0;     // 温差 (°C)
float hotTemperature = 0.0;      // 热端温度 (°C)
float coldTemperature = 0.0;     // 冷端温度 (°C)

// ADS1115校准参数
float adsGainFactor = 0.0;       // 增益系数
bool adsInitialized = false;     // ADS1115初始化标志

// 数据滤波
const int SAMPLE_COUNT = 10;     // 移动平均采样点数
float voltageSamples[SAMPLE_COUNT];
int sampleIndex = 0;

// ==================== 设置函数 ====================
void setup() {
  Serial.begin(9600);
  Serial.println("TEG温差与电压测量系统");
  Serial.println("=========================");
  
  // 初始化I2C总线
  Wire.begin();

  // 初始化DS18B20
  initDS18B20();
  
  // 预留ADS1115初始化位置
  initADS1115();

   // 初始化电压采样数组
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    voltageSamples[i] = 0.0;
  }
  
  // 打印系统状态
  printSystemStatus();
  
  Serial.println();
  Serial.println("开始数据采集...");
  Serial.println("格式: 时间(ms),热端温度(°C),冷端温度(°C),温差(°C),电压(V)");
  Serial.println("----------------------------------------");
}

// ==================== 主循环 ====================
void loop() {
   // 记录开始时间
  unsigned long startTime = millis();

  // 读取温度数据
  readTemperatures();
  
  // 读取TEG电压
  readTEGVoltage();
  
  // 计算温差
  calculateTemperatureDiff();

  //数据验证
  if(checkDataValidity()){
    // 显示数据用于调试
    displayData();
    //输出CSV格式的数据
    logDataToSerial();
    }
  
  // 数据记录间隔
  delay(1000);  // 每1秒记录一次
}

// ==================== DS18B20初始化函数 ====================
void initDS18B20() {
  Serial.println("初始化DS18B20温度传感器...");
  
  sensors.begin();

  //等待稳定
  delay(100);
  
  // 检测连接的设备数量
  int deviceCount = sensors.getDeviceCount();
  Serial.print("发现设备数量: ");
  Serial.println(deviceCount);
  
  if (deviceCount >= 2) {
    // 获取第一个传感器地址（热端）
    if (sensors.getAddress(hotThermometer, 0)) {
      Serial.print("热端传感器地址: ");
      printAddress(hotThermometer);

      // 设置分辨率（12位，0.0625°C精度）
      sensors.setResolution(hotThermometer, 12);
    }
    
    // 获取第二个传感器地址（冷端）
    if (sensors.getAddress(coldThermometer, 1)) {
      Serial.print("冷端传感器地址: ");
      printAddress(coldThermometer);

      // 设置分辨率
      sensors.setResolution(coldThermometer, 12);
    }
    
    sensorsFound = true;
    
  } else {
    Serial.println("错误: 需要至少2个DS18B20传感器!");
    sensorsFound = false;
  }
  
  Serial.println("DS18B20初始化完成");
  Serial.println("-------------------------");
}

// ==================== ADS1115初始化函数 ====================
void initADS1115() {
  Serial.print("初始化ADS1115模数转换器...");
  
  // 初始化ADS1115，默认I2C地址0x48
  if (!ads.begin()) {
    Serial.println("  失败! 请检查ADS1115连接");
    adsInitialized = false;
    return;
  }
  
  // 设置增益
  // GAIN_TWOTHIRDS: ±6.144V
  // GAIN_ONE:       ±4.096V
  // GAIN_TWO:       ±2.048V 
  // GAIN_FOUR:      ±1.024V
  // GAIN_EIGHT:     ±0.512V
  // GAIN_SIXTEEN:   ±0.256V
  
  // 根据TEG电压范围选择增益
  ads.setGain(GAIN_TWO);  // ±2.048V，每bit 0.0625mV
  
  // 计算增益系数
  switch (ads.getGain()) {
    case GAIN_TWOTHIRDS: adsGainFactor = 0.1875; break;    // 6.144V / 32767
    case GAIN_ONE:       adsGainFactor = 0.125; break;     // 4.096V / 32767
    case GAIN_TWO:       adsGainFactor = 0.0625; break;    // 2.048V / 32767
    case GAIN_FOUR:      adsGainFactor = 0.03125; break;   // 1.024V / 32767
    case GAIN_EIGHT:     adsGainFactor = 0.015625; break;  // 0.512V / 32767
    case GAIN_SIXTEEN:   adsGainFactor = 0.0078125; break; // 0.256V / 32767
    default:             adsGainFactor = 0.0625; break;    // 默认GAIN_TWO
  }
  
  // 设置数据速率（默认128SPS）128次采样/秒
  // 可选速率: 8, 16, 32, 64, 128, 250, 475, 860
  ads.setDataRate(RATE_ADS1115_128SPS);
  
  adsInitialized = true;
  Serial.println("  成功!");
  Serial.print("  增益设置: GAIN_TWO (±2.048V)");
  Serial.print("  分辨率: ");
  Serial.print(adsGainFactor * 1000, 4);
  Serial.println(" mV/LSB");
}

// ==================== 读取温度函数 ====================
//后期考虑测不同高温同样温差对内阻影响
void readTemperatures() {
  if (!sensorsFound) return;
  
  // 请求所有传感器进行温度转换
  sensors.requestTemperatures();
  
  // 读取热端温度
  hotTemperature = sensors.getTempC(hotThermometer);
  
  // 读取冷端温度
  coldTemperature = sensors.getTempC(coldThermometer);
  
  // 检查传感器是否断开
  if (hotTemperature == DEVICE_DISCONNECTED_C) {
    Serial.println("错误: 热端传感器断开!");
  }
  if (coldTemperature == DEVICE_DISCONNECTED_C) {
    Serial.println("错误: 冷端传感器断开!");
  }
}

// ==================== 读取TEG电压函数 ====================
void readTEGVoltage() {
  if (!adsInitialized) {
    // ADS1115没有接入，使用模拟值
    tegVoltage = 0.5 + (random(-100, 100) / 1000.0);  // 模拟0.4-0.6V电压
    return;
  }
  
  // 读取A0和A1之间的差分电压
  int16_t adcValue = ads.readADC_Differential_0_1();
  
  // 计算实际电压值 (单位: V)
  float rawVoltage = adcValue * adsGainFactor / 1000.0;  // 转换为伏特
  
  // 如果使用分压电路，需要乘以分压系数
  float voltageDividerRatio = 1.0;  // 无分压
  // float voltageDividerRatio = 2.0;  // 100kΩ+100kΩ分压
  
  // 应用分压系数
  rawVoltage *= voltageDividerRatio;
  
  // 移动平均滤波
  voltageSamples[sampleIndex] = rawVoltage;
  sampleIndex = (sampleIndex + 1) % SAMPLE_COUNT;//索引循环更新
  
  // 计算平均值
  float sum = 0.0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    sum += voltageSamples[i];
  }
  
  tegVoltage = sum / SAMPLE_COUNT;
  
  // 可选：添加软件校准偏移
  float voltageOffset = 0.0;  // 校准偏移量 (V)
  tegVoltage += voltageOffset;
}

// ==================== 计算温差函数 ====================
void calculateTemperatureDiff() {
  temperatureDiff = hotTemperature - coldTemperature;
}

// ==================== 数据验证函数 ====================
bool checkDataValidity() {
  bool valid = true;
  
  // 检查温度传感器数据
  if (!sensorsFound) {
    Serial.println("警告: 温度传感器未初始化");
    valid = false;
  }
  
  if (hotTemperature == DEVICE_DISCONNECTED_C || 
      coldTemperature == DEVICE_DISCONNECTED_C) {
    Serial.println("警告: 温度传感器读取失败");
    valid = false;
  }
  
  // 检查ADS1115数据
  if (!adsInitialized) {
    Serial.println("警告: ADS1115未初始化，使用模拟数据");
    // 不标记为无效，允许使用模拟数据
  }
  
  // 检查温差是否合理
  if (abs(temperatureDiff) > 100.0) {  // 假设温差不会超过100°C
    Serial.println("警告: 温差异常");
    valid = false;
  }
  
  // 检查电压是否合理
  if (abs(tegVoltage) > 5.0) {  // 假设电压不会超过5V
    Serial.println("警告: 电压异常");
    valid = false;
  }
  
  return valid;
}

// ==================== 调试时显示数据函数 ====================
void displayData() {
  Serial.print("热端温度: ");
  Serial.print(hotTemperature, 2);
  Serial.println(" °C");
  
  Serial.print("冷端温度: ");
  Serial.print(coldTemperature, 2);
  Serial.println(" °C");

  Serial.println("=========================");
  Serial.print("温差: ");
  Serial.print(temperatureDiff,2);
  Serial.print(" °C, ");
  
  Serial.print("开路电压: ");
  Serial.print(tegVoltage,4);
  Serial.println(" V");

   // 计算塞贝克系数 (V/°C)
  if (abs(temperatureDiff) > 0.1) {  // 避免除以零
    float seebeckCoefficient = tegVoltage / temperatureDiff;
    Serial.print("塞贝克系数: ");
    Serial.print(seebeckCoefficient * 1000, 2);  // 转换为mV/°C
    Serial.println(" mV/°C");
  }

  Serial.println("=========================");
}

// ==================== 输出CSV函数 ====================
void logDataToSerial() {
  //CSV格式: 时间戳(ms),热端温度(°C),冷端温度(°C),温差(°C),电压(V)
  Serial.print(millis() / 1000.0);  // 时间（秒）
  Serial.print(",");
  Serial.print(hotTemperature,2);
  Serial.print(",");
  Serial.print(coldTemperature,2);
  Serial.print(",");
  Serial.print(temperatureDiff,2);
  Serial.print(",");
  Serial.println(tegVoltage,4);
}

// ==================== 系统状态打印函数 ====================
void printSystemStatus() {
  Serial.println("系统状态:");
  Serial.print("  温度传感器: ");
  Serial.println(sensorsFound ? "正常" : "异常");
  
  Serial.print("  ADS1115: ");
  Serial.println(adsInitialized ? "正常" : "异常");
  
  if (adsInitialized) {
    Serial.print("  当前增益: ");
    switch (ads.getGain()) {
      case GAIN_TWOTHIRDS: Serial.println("GAIN_TWOTHIRDS (±6.144V)"); break;
      case GAIN_ONE:       Serial.println("GAIN_ONE (±4.096V)"); break;
      case GAIN_TWO:       Serial.println("GAIN_TWO (±2.048V)"); break;
      case GAIN_FOUR:      Serial.println("GAIN_FOUR (±1.024V)"); break;
      case GAIN_EIGHT:     Serial.println("GAIN_EIGHT (±0.512V)"); break;
      case GAIN_SIXTEEN:   Serial.println("GAIN_SIXTEEN (±0.256V)"); break;
      default:             Serial.println("未知"); break;
    }
  }
  
  Serial.print("  数据速率: 128 SPS");
  Serial.println();
}

// ==================== 辅助函数 ====================
// 打印传感器地址
void printAddress(DeviceAddress deviceAddress) {
  for (uint8_t i = 0; i < 8; i++) {
    if (deviceAddress[i] < 16) Serial.print("0");
    Serial.print(deviceAddress[i], HEX);
    if (i < 7) Serial.print(":");
  }
  Serial.println();
}
// ==================== 错误处理函数 ====================
bool checkSensorError() {
  if (!sensorsFound) {
    Serial.println("错误: 传感器未正确连接!");
    return true;
  }
  
  float hotTemp = sensors.getTempC(hotThermometer);
  float coldTemp = sensors.getTempC(coldThermometer);
  
  if (hotTemp == DEVICE_DISCONNECTED_C || coldTemp == DEVICE_DISCONNECTED_C) {
    Serial.println("错误: 传感器读取失败!");
    return true;
  }
  
  return false;
}