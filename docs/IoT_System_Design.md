# IoT System Design

## Sensors
- **Type:** Accelerometer, Gyroscope
- **Location:** Smartphone
- **Specifications:**
  - Accelerometer range: ±2g to ±16g
  - Gyroscope range: ±250°/s to ±2000°/s
- **Limitations:**
  - Data noise may affect accuracy.
  - Continuous use impacts battery life.

## Edge Processing
- **Required:** Yes
- **Purpose:** Reduce data size and filter noise before transmission.
- **Device Requirements:**
  - Minimum of 2GB RAM.
  - Multi-core processor for real-time computations.

## Networking
- **Method:** Bluetooth for short-range, Wi-Fi for cloud communication.
- **Protocol:** HTTP or MQTT
- **Limitations:** Dependent on network stability for real-time updates.

## Data Storage and Processing
- **Storage:** Cloud-based (AWS S3 or Google Cloud)
- **Scalability:** Distributed storage for large datasets.
- **Processing Tools:** Python (TensorFlow, Pandas)
- **Insights Generated:** Activity classifications and health recommendations.
 
