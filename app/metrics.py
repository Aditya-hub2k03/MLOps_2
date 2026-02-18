import time

class Metrics:
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0

    def log_request(self, latency):
        self.request_count += 1
        self.total_latency += latency

    def get_metrics(self):
        avg_latency = (
            self.total_latency / self.request_count
            if self.request_count > 0 else 0
        )
        return {
            "total_requests": self.request_count,
            "average_latency": avg_latency
        }

metrics = Metrics()
