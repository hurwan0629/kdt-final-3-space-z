"""프로세스 CPU·RAM과 선택한 NVIDIA GPU 장치의 사용량을 측정한다."""

from __future__ import annotations

import math
import threading
import time
from typing import Any


class ResourceMonitor:
    """실행 시간과 CPU·GPU 사용률 및 RAM·GPU 메모리 사용량을 기록할 객체."""

    def __init__(
        self,
        sample_interval_seconds: float = 0.5,
        process_id: int | None = None,
        gpu_index: int | None = None,
    ) -> None:
        """측정 간격과 대상 프로세스·GPU를 받는다.

        인자:
            sample_interval_seconds: 자동 측정 간격(초). 양수여야 한다.
            process_id: CPU 사용률과 RAM을 측정할 프로세스 번호. None이면 현재 프로세스.
            gpu_index: 사용률과 메모리를 측정할 GPU 번호. None이면 GPU 측정을 생략한다.
                GPU 지표는 지정한 장치 전체를 기준으로 하며 프로세스 지표와 구분한다.

        반환:
            None. 설정을 검증하고 측정 준비만 한다. 측정 시작은 start()가 맡는다.
        """
        if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
            raise ValueError("측정 간격은 유한한 양수여야 합니다.")
        if process_id is not None and (type(process_id) is not int or process_id <= 0):
            raise ValueError("process_id는 양의 정수 또는 None이어야 합니다.")
        if gpu_index is not None and (type(gpu_index) is not int or gpu_index < 0):
            raise ValueError("gpu_index는 0 이상의 정수 또는 None이어야 합니다.")
        import psutil

        self.sample_interval_seconds = sample_interval_seconds
        self.process_id = process_id
        self.gpu_index = gpu_index
        self._psutil = psutil
        self._process = psutil.Process(process_id)
        self._lock = threading.RLock()  # 수동 sample과 주기적인 sample의 중복 집계를 막는다.
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._summary: dict[str, Any] | None = None
        self._nvml: Any = None
        self._gpu_handle: Any = None

    def start(self) -> None:
        """이전 기록을 비우고 경과 시간과 주기적인 자원 측정을 시작한다.

        인자:
            없음.

        처리:
            1) 시작 시간과 CPU 측정 기준을 준비한다.
            2) sample_interval_seconds 간격으로 sample()을 호출하도록 준비한다.
            측정 중 다시 호출하면 오류로 처리하며, GPU를 측정할 수 없어도 CPU·RAM은 측정한다.

        반환:
            None.
        """
        with self._lock:
            if self._closed or self._running:
                raise RuntimeError("종료했거나 이미 측정 중인 측정기는 시작할 수 없습니다.")
            # 1) 이전 결과를 비우고 CPU 누적 시간의 기준을 준비한다.
            self._summary = None
            self._connect_gpu()
            self._started_at = time.perf_counter()
            self._last_at = self._started_at
            self._last_cpu_seconds: float | None = self._cpu_seconds()
            self._cpu_integral = 0.0
            self._cpu_duration = 0.0
            self._sample_count = 0
            self._peak_ram: int | None = None
            self._gpu_utilization_sum = 0.0
            self._gpu_utilization_count = 0
            self._peak_gpu_memory: int | None = None
            self._stop_event.clear()
            self._running = True
            self.sample()
            # 2) 게임·학습 루프와 독립적으로 일정 간격마다 측정한다.
            self._thread = threading.Thread(target=self._sample_periodically, daemon=True, name="space-z-resource-monitor")
            self._thread.start()

    def sample(self) -> dict[str, float | int | None]:
        """현재 자원 사용량을 한 번 측정해 기록하고 반환한다.

        인자:
            없음. start() 이후 측정 중에 호출한다.

        반환:
            elapsed_seconds: 측정 시작 이후 실제 경과 시간(초).
            process_cpu_percent: 대상 프로세스 CPU 사용률(%). 여러 코어 사용 시 100을 넘을 수 있다.
            process_ram_bytes: 대상 프로세스의 RAM 상주 메모리(RSS, 바이트).
            gpu_device_utilization_percent: 지정한 GPU 장치 전체의 사용률(%).
            gpu_device_memory_bytes: 지정한 GPU 장치 전체의 사용 중 메모리(바이트).
            위 항목을 담은 딕셔너리. 측정하지 않거나 지원하지 않는 지표는 None.
        """
        with self._lock:
            if not self._running:
                raise RuntimeError("sample 전에 start()로 측정을 시작해야 합니다.")
            now = time.perf_counter()
            cpu_seconds = self._cpu_seconds()
            duration = now - self._last_at
            cpu_percent = None
            if cpu_seconds is not None and self._last_cpu_seconds is not None and duration > 0:
                cpu_percent = max(0.0, (cpu_seconds - self._last_cpu_seconds) / duration * 100)
                self._cpu_integral += cpu_percent * duration
                self._cpu_duration += duration
            self._last_at, self._last_cpu_seconds = now, cpu_seconds
            try:
                ram = self._process.memory_info().rss
            except self._psutil.Error:
                ram = None
            utilization = gpu_memory = None
            if self._gpu_handle is not None:
                # 일부 장치는 사용률만 지원하지 않으므로 GPU 지표를 각각 조회한다.
                try:
                    utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu
                except self._nvml.NVMLError:
                    pass
                try:
                    gpu_memory = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle).used
                except self._nvml.NVMLError:
                    pass
            self._sample_count += 1
            if ram is not None:
                self._peak_ram = max(self._peak_ram or 0, ram)
            if utilization is not None:
                self._gpu_utilization_sum += utilization
                self._gpu_utilization_count += 1
            if gpu_memory is not None:
                self._peak_gpu_memory = max(self._peak_gpu_memory or 0, gpu_memory)
            return {"elapsed_seconds": now - self._started_at, "process_cpu_percent": cpu_percent,
                    "process_ram_bytes": ram, "gpu_device_utilization_percent": utilization,
                    "gpu_device_memory_bytes": gpu_memory}

    def stop(self) -> dict[str, Any]:
        """주기적인 측정을 중단하고 이번 측정의 요약을 반환한다.

        인자:
            없음. start() 이후 호출한다.

        처리:
            1) 마지막 측정을 기록하고 주기적인 측정을 종료한다.
            2) 전체 경과 시간과 수집한 지표의 평균·최댓값을 계산한다.

        반환:
            elapsed_seconds, sample_count, mean_process_cpu_percent,
            peak_process_ram_bytes, mean_gpu_device_utilization_percent,
            peak_gpu_device_memory_bytes를 담은 딕셔너리.
            사용할 수 없는 지표는 None이며, 종료 후 다시 호출하면 같은 요약을 반환한다.
        """
        # 먼저 주기적인 작업을 깨워 종료한다. join 중에는 측정 잠금을 잡지 않는다.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            if not self._running:
                if self._summary is None:
                    raise RuntimeError("stop 전에 start()로 측정을 시작해야 합니다.")
                return dict(self._summary)
            self.sample()
            self._running = False
            self._summary = {
                "elapsed_seconds": time.perf_counter() - self._started_at,
                "sample_count": self._sample_count,
                "mean_process_cpu_percent": self._cpu_integral / self._cpu_duration if self._cpu_duration else None,
                "peak_process_ram_bytes": self._peak_ram,
                "mean_gpu_device_utilization_percent": self._gpu_utilization_sum / self._gpu_utilization_count if self._gpu_utilization_count else None,
                "peak_gpu_device_memory_bytes": self._peak_gpu_memory,
            }
            return dict(self._summary)

    def close(self) -> None:
        """측정을 종료하고 측정기가 소유한 자원을 해제한다.

        인자:
            없음.

        반환:
            None. 측정용 작업과 GPU 연결 등을 정리한다.
            측정 전이나 해제 후에 다시 호출해도 안전해야 한다.
        """
        if self._closed:
            return
        try:
            if self._running:
                self.stop()
        finally:
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except self._nvml.NVMLError:
                    pass
            self._nvml = self._gpu_handle = None
            self._closed = True

    def _cpu_seconds(self) -> float | None:
        """프로세스의 누적 CPU 초를 반환한다. 프로세스가 종료되거나 접근 불가하면 None."""
        try:
            cpu_times = self._process.cpu_times()
            return float(cpu_times.user + cpu_times.system)
        except self._psutil.Error:
            return None

    def _connect_gpu(self) -> None:
        """요청한 GPU에 연결한다. 라이브러리·장치·드라이버가 없으면 GPU 측정을 생략한다."""
        if self.gpu_index is None or self._nvml is not None:
            return
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return
        self._nvml = pynvml
        try:
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except pynvml.NVMLError:
            self._gpu_handle = None

    def _sample_periodically(self) -> None:
        """중단 요청 전까지 설정한 간격으로 자원을 기록한다."""
        while not self._stop_event.wait(self.sample_interval_seconds):
            self.sample()
