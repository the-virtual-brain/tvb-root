from sympy import symbols, cse, Matrix, exp
Vector = lambda *args: Matrix([arg for arg in args])
syms = 'nu_max, r, v0, y1, y2, a_1, J, y0, a_3, y3, y4, y5, lrc, short_range_coupling, A, B, a, b, mu, a_2, a_4'
nu_max, r, v0, y1, y2, a_1, J, y0, a_3, y3, y4, y5, lrc, short_range_coupling, A, B, a, b, mu, a_2, a_4 = symbols(syms)
sigm_y1_y2 = 2 * nu_max / (1 + exp(r * (v0 - (y1 - y2))))
sigm_y0_1  = 2 * nu_max / (1 + exp(r * (v0 - (a_1 * J * y0))))
sigm_y0_3  = 2 * nu_max / (1 + exp(r * (v0 - (a_3 * J * y0))))
state = Vector(y0, y1, y2, y3, y4, y5)
param = Vector(nu_max, r, v0, a_1, J, a_3, lrc, short_range_coupling, A, B, a, b, mu, a_2, a_4)
drift = Vector(
    y3,
    y4,
    y5,
    A * a * sigm_y1_y2 - 2.0 * a * y3 - a ** 2 * y0,
    A * a * (mu + a_2 * J * sigm_y0_1 + lrc + short_range_coupling)
        - 2.0 * a * y4 - a ** 2 * y1,
    B * b * (a_4 * J * sigm_y0_3) - 2.0 * b * y5 - b ** 2 * y2,
)
n = 6
flatjac = lambda x, y: x.diff(y).reshape(len(x)*len(y)).tolist()
terms = drift.reshape(1, n).tolist()[0] + flatjac(drift, state) + flatjac(drift, param)
do_cse = True
if do_cse:
    reps, rexs = cse(terms)
else:
    reps, rexs = [], terms
for lhs, rhs in reps:
    print(lhs, '=', rhs)
drift_rex = rexs[:n]
jac_state = rexs[n:n+n*n]
jac_param = rexs[n+n*n:]
for i, ex in enumerate(drift_rex):
    print('xt[%d]' % i, '=', ex)
for i in range(n):
    for j in range(n):
        val = jac_state[i*n + j]
        if val == 0:
            continue
        print("jxtx[%d, %d]" % (i, j), '=', val)

for i in range(n):
    for j in range(n):
        val = jac_param[i*n + j]
        if val == 0:
            continue
        print("jxtp[%d, %d]" % (i, j), '=', val)