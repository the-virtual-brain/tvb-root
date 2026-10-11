import time
import numpy
from tvb.simulator import integrators, models, coupling, simulator
from tvb.datatypes import connectivity


def test_benchmark_efficiency():
    conn = connectivity.Connectivity.from_file()
    conn.speed = numpy.array([4.0])

    sim_duration = 50.0  # ms

    # Legacy configuration: Fine step size with single-stage evaluation
    sim_fine = simulator.Simulator(
        model=models.Generic2dOscillator(a=numpy.array([0.5])),
        connectivity=conn,
        coupling=coupling.Linear(a=numpy.array([0.014])),
        integrator=integrators.HeunDeterministic(dt=0.005),
        simulation_length=sim_duration
    )
    sim_fine.integrator.stages = 1
    sim_fine.configure()

    start_time = time.perf_counter()
    for _ in sim_fine():
        pass
    elapsed_fine = time.perf_counter() - start_time

    # New configuration: Coarse step size with 2nd order multi-stage evaluation
    sim_coarse = simulator.Simulator(
        model=models.Generic2dOscillator(a=numpy.array([0.5])),
        connectivity=conn,
        coupling=coupling.Linear(a=numpy.array([0.014])),
        integrator=integrators.HeunDeterministic(dt=0.02),
        simulation_length=sim_duration
    )
    sim_coarse.integrator.stages = 2
    sim_coarse.configure()

    start_time = time.perf_counter()
    for _ in sim_coarse():
        pass
    elapsed_coarse = time.perf_counter() - start_time

    speedup = elapsed_fine / elapsed_coarse

    print(f"\nExecution Benchmark (sim_duration = {sim_duration} ms):")
    print(f"Legacy fine stepping (dt=0.005 ms, stages=1): {elapsed_fine:.4f} s")
    print(f"Batched 2nd order stepping (dt=0.02 ms, stages=2): {elapsed_coarse:.4f} s")
    print(f"Observed Wall-Clock Speedup: {speedup:.2f}x\n")

    assert elapsed_coarse < elapsed_fine, "Batched coarse step should finish faster than fine step."