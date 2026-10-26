import psutil
import time
import csv
import os
from datetime import datetime


OUTPUT_FILE = "data/system_telemetry.csv"


def collect():
    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "cpu_percent",
                "ram_percent",
                "cpu_frequency",
                "disk_percent",
                "bytes_sent",
                "bytes_received"
            ])

        print("PHYSOS Telemetry Engine started.")
        print("Collecting data every second...")
        print("Press CTRL+C to stop.\n")

        previous_net = psutil.net_io_counters()

        try:
            while True:
                timestamp = datetime.now().isoformat()

                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory().percent

                frequency = psutil.cpu_freq()
                cpu_frequency = frequency.current if frequency else 0

                disk = psutil.disk_usage("/").percent

                current_net = psutil.net_io_counters()

                sent = current_net.bytes_sent - previous_net.bytes_sent
                received = current_net.bytes_recv - previous_net.bytes_recv

                previous_net = current_net

                writer.writerow([
                    timestamp,
                    cpu,
                    ram,
                    cpu_frequency,
                    disk,
                    sent,
                    received
                ])

                file.flush()

                print(
                    f"{timestamp} | "
                    f"CPU: {cpu:5.1f}% | "
                    f"RAM: {ram:5.1f}% | "
                    f"Freq: {cpu_frequency:7.1f} MHz"
                )

        except KeyboardInterrupt:
            print("\nPHYSOS Telemetry Engine stopped.")


if __name__ == "__main__":
    collect()