
import csv
import os
import random
import time
from datetime import datetime

OUTPUT_FILE = os.path.join("data", "physical_sensors.csv")

SAMPLE_INTERVAL = 1
SAMPLE_COUNT = 120


def main():
    os.makedirs("data", exist_ok=True)

    # Starting values for SYNTHETIC test data only.
    temperature_c = 28.0
    humidity_percent = 60.0

    file_exists = (
        os.path.exists(OUTPUT_FILE)
        and os.path.getsize(OUTPUT_FILE) > 0
    )

    print("PHYSOS Sensor Simulation Engine")
    print("=" * 38)
    print("Mode: SIMULATION — not real sensor measurements")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Samples to generate: {SAMPLE_COUNT}")
    print("Press CTRL+C to stop early.\n")

    with open(
        OUTPUT_FILE, "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "source",
                "temperature_c",
                "humidity_percent",
            ])

        try:
            for index in range(SAMPLE_COUNT):
                # Small random variations to test the pipeline.
                temperature_c += random.uniform(-0.15, 0.15)
                humidity_percent += random.uniform(-0.5, 0.5)

                # Keep synthetic values within reasonable test bounds.
                temperature_c = max(20.0, min(40.0, temperature_c))
                humidity_percent = max(
                    20.0, min(90.0, humidity_percent)
                )

                timestamp = datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                )

                writer.writerow([
                    timestamp,
                    "SIMULATED",
                    round(temperature_c, 2),
                    round(humidity_percent, 2),
                ])
                file.flush()

                print(
                    f"{index + 1:03d}/{SAMPLE_COUNT} | "
                    f"{timestamp} | "
                    f"Temperature: {temperature_c:.2f} C | "
                    f"Humidity: {humidity_percent:.2f}%"
                )

                time.sleep(SAMPLE_INTERVAL)

        except KeyboardInterrupt:
            print("\nSimulation stopped early.")

    print(f"\nSensor test data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
