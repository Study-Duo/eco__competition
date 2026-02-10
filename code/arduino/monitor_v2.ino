/*
TEG温差发电混合模式数据采集系统
版本：2.0
模式：RAW模式 - 仅采集原始数据，不在Arduino中进行计算
数据格式：CSV（时间戳,热端温度,冷端温度,原始电压,通道1,通道2,通道3）
*/

/* ==================== 接线说明 ====================
DS18B20引脚 -> Arduino Nano引脚
1. GND (黑色) -> GND
2. DQ  (黄色) -> D2 (数字引脚2)
3. VDD (红色) -> 5V

ADS1115引脚 -> Arduino Nano引脚
VDD  (电源正) -> 5V
GND  (地)     -> GND
SCL  (时钟线) -> A5 (SCL)
SDA  (数据线) -> A4 (SDA)
ADDR (地址)   -> GND (I2C地址0x48)
A0   (通道0)  -> TEG正极 (通过分压电阻)
A1   (通道1)  -> TEG负极
A2   (通道2)  -> 预留 (电流/负载监测)
A3   (通道3)  -> 预留 (其他传感器)

数据格式：
时间戳(ms),热端温度(°C),冷端温度(°C),TEG原始电压(mV),CH1_raw,CH2_raw,CH3_raw
*/

// ==================== 库文件引入 ====================
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

// ==================== 引脚定义 ====================
#define ONE_WIRE_BUS 2    // DS18B20数据引脚
#define LED_PIN 13        // 状态指示灯引脚

// ==================== 对象声明 ====================
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
Adafruit_ADS1115 ads;

// ==================== 全局变量声明 ====================
// 温度传感器地址
DeviceAddress sensorAddresses[4];
uint8_t sensorCount = 0;

// 数据缓存
float temperatures[4] = {0.0, 0.0, 0.0, 0.0};
int16_t adcValues[4] = {0, 0, 0, 0};
float voltageRaw = 0.0;  // 原始电压值(mV)

// 系统状态
bool systemReady = false;
bool adsReady = false;
bool tempReady = false;

// 采样设置
const uint32_t SAMPLE_INTERVAL = 1000;  // 采样间隔(ms)
const uint8_t AVG_SAMPLES = 5;          // 移动平均采样数
uint32_t lastSampleTime = 0;

// 数据缓冲区
char outputBuffer[128];

// ==================== 设置函数 ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  
  pinMode(LED_PIN, OUTPUT);
  
  Serial.println("TEG数据采集系统 - RAW模式");
  Serial.println("版本: 2.0");
  Serial.println("数据格式: CSV(时间戳,热端温度,冷端温度,电压,CH1,CH2,CH3)");
  Serial.println("==================================================");
  
  // 初始化I2C
  Wire.begin();
  
  // 初始化各模块
  initTemperatureSensors();
  initADS1115();
  
  // 检查系统状态
  checkSystemStatus();
  
  // 打印CSV标题行
  Serial.println("timestamp_ms,hot_temp_C,cold_temp_C,voltage_mV,ch1_raw,ch2_raw,ch3_raw");
  
  Serial.println("系统初始化完成，开始数据采集...");
}

// ==================== 主循环 ====================
void loop() {
  uint32_t currentTime = millis();
  
  // 定时采样
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    
    // 读取所有数据
    readAllSensors();
    
    // 发送数据（CSV格式）
    sendDataCSV(currentTime);
    
    // 闪烁LED指示工作状态
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}

// ==================== 温度传感器初始化 ====================
void initTemperatureSensors() {
  Serial.print("初始化DS18B20...");
  sensors.begin();
  delay(100);
  
  sensorCount = sensors.getDeviceCount();
  Serial.print("找到 ");
  Serial.print(sensorCount);
  Serial.println(" 个传感器");
  
  if (sensorCount >= 2) {
    for (uint8_t i = 0; i < min(sensorCount, 4); i++) {
      if (sensors.getAddress(sensorAddresses[i], i)) {
        sensors.setResolution(sensorAddresses[i], 12);
        Serial.print("传感器");
        Serial.print(i);
        Serial.print(": ");
        printAddress(sensorAddresses[i]);
      }
    }
    tempReady = true;
  } else {
    Serial.println("警告: 需要至少2个温度传感器");
  }
}

// ==================== ADS1115初始化（注意设置适宜的增益系数） ====================
void initADS1115() {
  Serial.print("初始化ADS1115...");
  
  if (ads.begin()) {
    // 配置所有通道为差分模式
    // 设置增益
    //GAIN_TWOTHIRDS	±6.144V		0.1875	
    //GAIN_ONE	±4.096V		0.125	
    //GAIN_TWO	±2.048V		0.0625	
    //GAIN_FOUR	±1.024V		0.03125	
    //GAIN_EIGHT	±0.512V		0.015625	
    //GAIN_SIXTEEN	±0.256V		0.0078125	

    ads.setGain(GAIN_TWO);  // ±2.048V
    ads.setDataRate(RATE_ADS1115_128SPS);
    adsReady = true;
    Serial.println("成功");
  } else {
    Serial.println("失败");
  }
}

// ==================== 系统状态检查 ====================
void checkSystemStatus() {
  systemReady = tempReady && adsReady;
  
  Serial.println("系统状态:");
  Serial.print("  温度传感器: ");
  Serial.println(tempReady ? "正常" : "异常");
  Serial.print("  ADS1115: ");
  Serial.println(adsReady ? "正常" : "异常");
  Serial.print("  系统状态: ");
  Serial.println(systemReady ? "就绪" : "等待");
  
  // 闪烁LED指示状态，正常闪烁两次，异常闪烁四次
  for (int i = 0; i < (systemReady ? 2 : 4); i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

// ==================== 读取所有传感器（注意设置对应的增益系数） ====================
void readAllSensors() {
  // 读取温度
  if (tempReady) {
    sensors.requestTemperatures();
    for (uint8_t i = 0; i < min(sensorCount, 4); i++) {
      float temp = sensors.getTempC(sensorAddresses[i]);
      temperatures[i] = (temp != DEVICE_DISCONNECTED_C) ? temp : -999.0;
    }
  } else {
    // 模拟数据用于测试
    temperatures[0] = 25.0 + random(-50, 50) / 10.0;
    temperatures[1] = 20.0 + random(-50, 50) / 10.0;
  }
  
  // 读取ADS1115
  if (adsReady) {
    // 通道0-1: TEG差分电压
    adcValues[0] = ads.readADC_Differential_0_1();
    
    // 通道2: 单端测量（预留）
    adcValues[1] = ads.readADC_SingleEnded(2);
    
    // 通道3: 单端测量（预留）
    adcValues[2] = ads.readADC_SingleEnded(3);
    
    // 转换原始电压值 (mV)
    voltageRaw = adcValues[0] * 0.0625;  // GAIN_TWO: 0.0625 mV/LSB
  } else {
    // 模拟数据
    voltageRaw = 500.0 + random(-100, 100);
    adcValues[1] = random(-1000, 1000);
    adcValues[2] = random(-1000, 1000);
  }
}

// ==================== 发送CSV数据 ====================
void sendDataCSV(uint32_t timestamp) {
  // 格式: timestamp_ms,hot_temp_C,cold_temp_C,voltage_mV,ch1_raw,ch2_raw,ch3_raw
  snprintf(outputBuffer, sizeof(outputBuffer),
           "%lu,%.2f,%.2f,%.2f,%d,%d,%d",
           timestamp,
           temperatures[0],  // 热端
           temperatures[1],  // 冷端
           voltageRaw,
           adcValues[0],     // TEG原始值
           adcValues[1],     // CH2原始值
           adcValues[2]);    // CH3原始值
  
  Serial.println(outputBuffer);
}

// ==================== 辅助函数 ====================
void printAddress(DeviceAddress deviceAddress) {
  for (uint8_t i = 0; i < 8; i++) {
    if (deviceAddress[i] < 16) Serial.print("0");
    Serial.print(deviceAddress[i], HEX);
    if (i < 7) Serial.print(":");
  }
  Serial.println();
}