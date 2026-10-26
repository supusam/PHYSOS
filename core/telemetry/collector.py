
import csv
import os
from datetime import datetime

import psutil

OUTPUT_FILE = os.path.join("data", "system_telemetry.csv")


def collect():
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_FILE)

    previous_net = psutil.net_io_counters()

    print("PHYSOS Telemetry Engine started.")
    print("Collecting computer telemetry every second.")
    print("Press CTRL+C to stop.\n")

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "cpu_percent",
                "ram_percent",
                "cpu_frequency_mhz",
                "disk_percent",
                "bytes_sent",
                "bytes_received"
            ])

        try:
            while True:
                timestamp = datetime.now().isoformat(timespec="seconds")
                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory().percent

                frequency = psutil.cpu_freq()
                cpu_frequency = frequency.current if frequency else 0

                disk = psutil.disk_usage(os.path.abspath(os.sep)).percent

                current_net = psutil.net_io_counters()
                sent = current_net.bytes_sent - previous_net.bytes_sent
                received = current_net.bytes_recv - previous_net.bytes_recv
                previous_net = current_net

                writer.writerow([
                    timestamp, cpu, ram, cpu_frequency,
                    disk, sent, received
                ])
                file.flush()

                print(
                    f"{timestamp} | CPU: {cpu:.1f}% | "
                    f"RAM: {ram:.1f}% | "
                    f"Frequency: {cpu_frequency:.1f} MHz"
                )

        except KeyboardInterrupt:
            print("\nPHYSOS Telemetry Engine stopped.")


if __name__ == "__main__":
    collect()
