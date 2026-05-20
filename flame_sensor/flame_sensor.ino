#define FLAME_DO 27
#define FLAME_AO 34

float K = 800000.0;

float estimateDistance(int adc)
{
  if (adc < 1200 || adc > 3800) return -1;
  return sqrt(K / adc);
}

bool isRealFlame(int adc, int digital)
{
  static int last = 0;
  int diff = abs(adc - last);
  last = adc;

  if (digital == LOW && adc > 1200 && diff > 120)
    return true;

  return false;
}

void setup() {
  Serial.begin(115200);
  pinMode(FLAME_DO, INPUT);
}

void loop() {

  int adc = analogRead(FLAME_AO);
  int d = digitalRead(FLAME_DO);

  if (isRealFlame(adc, d)) {
    float dist = estimateDistance(adc);
    Serial.print("🔥 Flame detected  Distance≈");
    Serial.print(dist);
    Serial.println(" cm");
  } else {
    Serial.println("No flame");
  }

  delay(200);
}