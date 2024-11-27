import os
import signal
import subprocess


class MemoryTracker:
    def start(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        self.process = subprocess.Popen(
            [os.path.join(cwd, "track_gpu_memory.sh")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def end(self):
        self.process.send_signal(signal.SIGINT)
        self.process.wait()
        stdout, stderr = self.process.communicate()
        peak_memory = stdout.decode().strip()
        return peak_memory
