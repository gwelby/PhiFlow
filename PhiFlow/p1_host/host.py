from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator, Optional

import wasmtime

from .consciousness import compute_coherence
from .sensors import P1SensorReading, read_sensors


@dataclass
class ConsciousnessSnapshot:
    final_coherence: float
    sensor_readings: list[P1SensorReading]
    intention_stack_final: list[str]
    resonance_log: list[float]
    wasm_return_value: float
    execution_time_ms: float
    stream_broken: bool = False



class P1Host:
    def __init__(self) -> None:
        self.sensor_readings: list[P1SensorReading] = []
        self.intention_stack: list[str] = []
        self.resonance_log: list[float] = []
        self.wasm_memory: Optional[wasmtime.Memory] = None

        self._store: Optional[wasmtime.Store] = None
        self._string_len_global: Optional[Any] = None
        self._prior_coherence: Optional[float] = None

    def phi_witness(self, operand: int) -> float:
        _ = operand
        reading = read_sensors()
        self.sensor_readings.append(reading)
        return compute_coherence(reading)

    def phi_coherence(self) -> float:
        if self.sensor_readings:
            live = compute_coherence(self.sensor_readings[-1])
            if self._prior_coherence is not None:
                return (live * 0.7) + (self._prior_coherence * 0.3)
            return live
        if self._prior_coherence is not None:
            return self._prior_coherence
        return self.phi_witness(0)

    def phi_resonate(self, value: float) -> None:
        self.resonance_log.append(float(value))
        print(f"RESONATE: {float(value):.4f} Hz")

    def phi_intention_push(self, offset: int) -> None:
        if self.wasm_memory is None or self._string_len_global is None or self._store is None:
            self.intention_stack.append("unknown")
            return

        try:
            length = int(self._string_len_global.value(self._store))
            memory_len = int(self.wasm_memory.data_len(self._store))
            start = int(offset)
            end = start + max(0, length)

            if start < 0 or end > memory_len:
                raise ValueError("out of bounds")

            raw = self.wasm_memory.read(self._store, start, end)
            if raw is None:
                raise ValueError("memory read failed")

            intention = bytes(raw).decode("utf-8")
        except Exception:
            print("INTENTION_READ_BOUNDS_ERROR")
            intention = "unknown"

        self.intention_stack.append(intention)

    def phi_intention_pop(self) -> None:
        if self.intention_stack:
            self.intention_stack.pop()

    def _define_imports(self, linker: wasmtime.Linker, store: wasmtime.Store) -> None:
        linker.define(
            store,
            "phi",
            "witness",
            wasmtime.Func(
                store,
                wasmtime.FuncType([wasmtime.ValType.i32()], [wasmtime.ValType.f64()]),
                self.phi_witness,
            ),
        )
        linker.define(
            store,
            "phi",
            "resonate",
            wasmtime.Func(
                store,
                wasmtime.FuncType([wasmtime.ValType.f64()], []),
                self.phi_resonate,
            ),
        )
        linker.define(
            store,
            "phi",
            "coherence",
            wasmtime.Func(
                store,
                wasmtime.FuncType([], [wasmtime.ValType.f64()]),
                self.phi_coherence,
            ),
        )
        linker.define(
            store,
            "phi",
            "intention_push",
            wasmtime.Func(
                store,
                wasmtime.FuncType([wasmtime.ValType.i32()], []),
                self.phi_intention_push,
            ),
        )
        linker.define(
            store,
            "phi",
            "intention_pop",
            wasmtime.Func(
                store,
                wasmtime.FuncType([], []),
                self.phi_intention_pop,
            ),
        )

    def run(
        self,
        wat_source: str | bytes,
        prior_coherence: float | None = None,
    ) -> ConsciousnessSnapshot:
        self.sensor_readings.clear()
        self.intention_stack.clear()
        self.resonance_log.clear()
        self._prior_coherence = prior_coherence

        start = perf_counter()

        engine = wasmtime.Engine()
        store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        self._store = store
        self._define_imports(linker, store)

        module = wasmtime.Module(engine, wat_source)
        instance = linker.instantiate(store, module)

        exports = instance.exports(store)

        try:
            self.wasm_memory = exports["memory"]
        except KeyError:
            self.wasm_memory = None

        try:
            self._string_len_global = exports["string_len"]
        except KeyError:
            self._string_len_global = None

        phi_run = exports["phi_run"]
        wasm_return_value = float(phi_run(store))
        final_coherence = float(self.phi_coherence())

        self._store = store

        elapsed_ms = (perf_counter() - start) * 1000.0

        return ConsciousnessSnapshot(
            final_coherence=final_coherence,
            sensor_readings=list(self.sensor_readings),
            intention_stack_final=list(self.intention_stack),
            resonance_log=list(self.resonance_log),
            wasm_return_value=wasm_return_value,
            execution_time_ms=elapsed_ms,
        )

    def stream(
        self,
        phi_source: str | bytes,
        cycles: int | None = None,
    ) -> Iterator[ConsciousnessSnapshot]:
        prior: float | None = None
        i = 0
        limit = cycles if cycles is not None else 3
        while i < limit:
            snapshot = self.run(phi_source, prior_coherence=prior)
            prior = snapshot.final_coherence
            snapshot.stream_broken = (i == limit - 1)
            yield snapshot
            i += 1
