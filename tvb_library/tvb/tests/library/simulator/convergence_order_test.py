import numpy
from tvb.simulator import integrators, models, coupling


def test_heun_second_order_convergence():
    model = models.Generic2dOscillator(a=numpy.array([0.5]))
    c_func = coupling.Linear(a=numpy.array([0.5]))
    weights = numpy.array([[0.0, 1.0], [1.0, 0.0]])

    def compute_coupling(state_var):
        val = weights.dot(state_var)
        return val[numpy.newaxis, :, numpy.newaxis]

    def full_dfun(X, c, local_c=0.0):
        return model.dfun(X, c, local_c)

    # Initial state for 2 state variables, 2 nodes, 1 mode: shape (2, 2, 1)
    X0 = numpy.array([[[1.0], [0.5]], [[0.2], [-0.4]]])

    T = 0.5
    dt_ref = 0.00005
    n_ref = int(T / dt_ref)

    # Ground truth reference run
    X_ref = X0.copy()
    heun_ref = integrators.HeunDeterministic(dt=dt_ref)
    for _ in range(n_ref):
        c_tn = compute_coupling(X_ref[0, :, 0])
        k1 = full_dfun(X_ref, c_tn)
        inter = X_ref + dt_ref * k1
        c_inter = compute_coupling(inter[0, :, 0])

        c_stages = numpy.array([c_tn, c_inter])
        X_ref = heun_ref.scheme(X_ref, full_dfun, c_stages, 0.0, 0.0)

    dts = [0.02, 0.01, 0.005, 0.0025]
    errors = []

    for dt in dts:
        X = X0.copy()
        n_steps = int(round(T / dt))
        heun = integrators.HeunDeterministic(dt=dt)

        for _ in range(n_steps):
            c_tn = compute_coupling(X[0, :, 0])
            k1 = full_dfun(X, c_tn)
            inter = X + dt * k1
            c_inter = compute_coupling(inter[0, :, 0])

            c_stages = numpy.array([c_tn, c_inter])
            X = heun.scheme(X, full_dfun, c_stages, 0.0, 0.0)

        err = numpy.sqrt(numpy.mean((X - X_ref) ** 2))
        errors.append(err)

    print(f"\nTested step sizes: {dts}")
    print(f"Observed errors: {errors}")
    slope, _ = numpy.polyfit(numpy.log(dts), numpy.log(errors), 1)
    print(f"Convergence slope: {slope:.3f}")

    assert slope > 1.9, f"Expected second-order convergence (~2.0), observed {slope:.3f}"