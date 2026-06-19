/*

 * ver2_2.ino

 * Classifier — IMU (9축) + Environmental (3축) = 12축 통합 모델
 * Target  : Arduino Nano 33 BLE Sense Rev2
 * Model   : QAT INT8 TFLite (from ver2_1.ipynb)
 *
 * Required files in same sketch folder:
 *   model.h       — QAT INT8 TFLite model array
 *   model_norm.h  — StandardScaler params (NORM_MEAN[2400], NORM_STD[2400])
 *
 * Libraries (install via Library Manager):
 *   Arduino_BMI270_BMM150  — IMU (accel + gyro + mag)
 *   Arduino_HS300x         — Temperature + Humidity (HS3003)
 *   Arduino_LPS22HB        — Pressure (LPS22HB)
 *   Arduino_TensorFlowLite — TFLite Micro inference
 *
 * Usage:
 *   Serial Monitor (115200 baud) → type "infer"
 *   → 10s 수집 → 5개 비겹침 윈도우 추론 → 확률 평균 → 확신도 + 최종 결과 출력
 *
 * Serial output example:
 *   === Inference Result ===
 *   on_desk: 0.8923  (89.2%)
 *   outdoor: 0.0612  ( 6.1%)
 *   walking: 0.0465  ( 4.6%)
 *   >> Final: on_desk
 *   ========================
 */

#include <Arduino_BMI270_BMM150.h>
#include <Arduino_HS300x.h>
#include <Arduino_LPS22HB.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "model_norm.h"

// ── Parameters (must match ver2_1.ipynb) ─────────────────────────────────────
#define NUM_CLASSES        3
#define WINDOW_SAMPLES     200    // 2000ms @ 100Hz — 모델 입력 윈도우 크기
#define COLLECT_SAMPLES    1000   // 10000ms @ 100Hz — 1회 수집 길이
#define NUM_WINDOWS        (COLLECT_SAMPLES / WINDOW_SAMPLES)  // 5 (비겹침)
#define N_AXES             12     // accX Y Z, gyrX Y Z, magX Y Z, temp, hum, pres
#define INPUT_SIZE         2400   // WINDOW_SAMPLES * N_AXES
#define SAMPLE_INTERVAL_MS 10     // 100Hz

static const char* CLASSES[NUM_CLASSES] = {
  "on_desk", "outdoor", "walking"
};

// ── Buffers ───────────────────────────────────────────────────────────────────
// raw_buffer: 1000 x 12 x 4 bytes = 48 KB
static float raw_buffer[COLLECT_SAMPLES][N_AXES];
static float features[INPUT_SIZE];

// ── TFLite ────────────────────────────────────────────────────────────────────
static tflite::AllOpsResolver resolver;
static uint8_t arena[20 * 1024];

static tflite::MicroInterpreter interpreter(
    tflite::GetModel(model), resolver, arena, sizeof(arena));

// ── Collect data (COLLECT_SAMPLES @ 100Hz = 10s) ─────────────────────────────
static void collectData() {
  float last_mx = 0.0f, last_my = 0.0f, last_mz = 0.0f;
  float last_temp = 0.0f, last_hum = 0.0f, last_pres = 0.0f;

  // Warm up environmental sensors before collection
  last_temp = HS300x.readTemperature();
  last_hum  = HS300x.readHumidity();
  last_pres = BARO.readPressure();

  for (int t = 0; t < COLLECT_SAMPLES; t++) {
    unsigned long t_start = millis();

    // Accelerometer drives the 100Hz sample rate
    while (!IMU.accelerationAvailable());

    float ax, ay, az;
    IMU.readAcceleration(ax, ay, az);

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    if (IMU.gyroscopeAvailable())
      IMU.readGyroscope(gx, gy, gz);

    // Magnetometer ODR ~20Hz: use last known value when no new sample
    if (IMU.magneticFieldAvailable())
      IMU.readMagneticField(last_mx, last_my, last_mz);

    // Environmental sensors: read every 10 samples (~10Hz) to avoid I2C blocking
    if (t % 10 == 0) {
      last_temp = HS300x.readTemperature();  // °C
      last_hum  = HS300x.readHumidity();     // %RH
      last_pres = BARO.readPressure();       // kPa
    }

    raw_buffer[t][0]  = ax;        // g
    raw_buffer[t][1]  = ay;        // g
    raw_buffer[t][2]  = az;        // g
    raw_buffer[t][3]  = gx;        // dps
    raw_buffer[t][4]  = gy;        // dps
    raw_buffer[t][5]  = gz;        // dps
    raw_buffer[t][6]  = last_mx;   // µT
    raw_buffer[t][7]  = last_my;   // µT
    raw_buffer[t][8]  = last_mz;   // µT
    raw_buffer[t][9]  = last_temp; // °C
    raw_buffer[t][10] = last_hum;  // %RH
    raw_buffer[t][11] = last_pres; // kPa

    unsigned long elapsed = millis() - t_start;
    if (elapsed < SAMPLE_INTERVAL_MS)
      delay(SAMPLE_INTERVAL_MS - elapsed);
  }
}

// ── Preprocess: 지정 시작 인덱스부터 WINDOW_SAMPLES 구간 flatten → StandardScaler
static void preprocess(int start) {
  for (int t = 0; t < WINDOW_SAMPLES; t++)
    for (int a = 0; a < N_AXES; a++)
      features[t * N_AXES + a] = raw_buffer[start + t][a];

  for (int i = 0; i < INPUT_SIZE; i++)
    features[i] = (features[i] - NORM_MEAN[i]) / NORM_STD[i];
}

// ── TFLite inference (INT8) ───────────────────────────────────────────────────
static void runInference(float probs[NUM_CLASSES]) {
  TfLiteTensor* in = interpreter.input(0);
  float   in_scale = in->params.scale;
  int32_t in_zp    = in->params.zero_point;

  for (int i = 0; i < INPUT_SIZE; i++) {
    int32_t q = (int32_t)roundf(features[i] / in_scale) + in_zp;
    in->data.int8[i] = (int8_t)constrain(q, -128, 127);
  }

  interpreter.Invoke();

  TfLiteTensor* out = interpreter.output(0);
  float   out_scale = out->params.scale;
  int32_t out_zp    = out->params.zero_point;

  for (int i = 0; i < NUM_CLASSES; i++)
    probs[i] = (out->data.int8[i] - out_zp) * out_scale;
}

// ── setup() ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial);
  pinMode(LED_BUILTIN, OUTPUT);

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU (BMI270_BMM150) init failed");
    while (1);
  }
  Serial.println("IMU initialized.");

  if (!HS300x.begin()) {
    Serial.println("ERROR: HS300x (temp/humidity) init failed");
    while (1);
  }
  Serial.println("HS300x initialized.");

  if (!BARO.begin()) {
    Serial.println("ERROR: LPS22HB (pressure) init failed");
    while (1);
  }
  Serial.println("LPS22HB initialized.");

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed");
    while (1);
  }
  Serial.println("TFLite model loaded.");
  Serial.println("Ready. Send 'infer' to classify.");
}

// ── runClassify() ─────────────────────────────────────────────────────────────
static void runClassify() {
  Serial.println("Collecting 10s data (IMU + Environmental)...");
  digitalWrite(LED_BUILTIN, HIGH);

  collectData();

  Serial.print("Collection done. Averaging ");
  Serial.print(NUM_WINDOWS);
  Serial.println(" windows...");

  // 5개 비겹침 윈도우 추론 후 확률 누적
  float avg_probs[NUM_CLASSES] = {0.0f, 0.0f, 0.0f};
  float probs[NUM_CLASSES];

  for (int w = 0; w < NUM_WINDOWS; w++) {
    preprocess(w * WINDOW_SAMPLES);
    runInference(probs);
    for (int i = 0; i < NUM_CLASSES; i++)
      avg_probs[i] += probs[i];
  }

  // 평균 계산
  for (int i = 0; i < NUM_CLASSES; i++)
    avg_probs[i] /= (float)NUM_WINDOWS;

  // argmax
  int best = 0;
  for (int i = 1; i < NUM_CLASSES; i++)
    if (avg_probs[i] > avg_probs[best]) best = i;

  // 각 클래스 확신도 + 최종 결과 출력
  Serial.println();
  Serial.println("=== Inference Result ===");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(CLASSES[i]);
    Serial.print(": ");
    Serial.print(avg_probs[i], 4);
    Serial.print("  (");
    Serial.print(avg_probs[i] * 100.0f, 1);
    Serial.println("%)");
  }
  Serial.print(">> Final: ");
  Serial.println(CLASSES[best]);
  Serial.println("========================");
  Serial.println();

  digitalWrite(LED_BUILTIN, LOW);
}

// ── loop() ───────────────────────────────────────────────────────────────────
void loop() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.equalsIgnoreCase("infer")) runClassify();
}
