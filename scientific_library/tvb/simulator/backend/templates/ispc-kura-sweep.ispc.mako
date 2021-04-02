#define M_PI_F 3.141592653589793f
#define PI_2 (2 * M_PI_F)

// we assume this is a power of 2
#define NH ((uint) 256)
#define state(time, i_node) (state_pwi[(((time) & (NH - 1))*n_node + (i_node))*n_thread + id])

// typedef uint32 uint;

inline uint state_idx(uint n_node, uint n_thread, uint time, uint i_node, uint i_thread)/*{{{*/
{
    return (time * n_node + i_node) * n_thread + i_thread;
}/*}}}*/

inline float wrap_2_pi(float x)/*{{{*/
{
    return x - ((float)(x > PI_2))*PI_2 + ((float)(x < PI_2))*PI_2;
    /*
    if (x < 0.0f)
        return x + PI_2;
    else if (x > PI_2)
        return x - PI_2;
    else
        return x;
    */
}/*}}}*/

inline float next_theta(float theta, float dt, float omega, float c, float rec_n, float sum)/*{{{*/
{
    return wrap_2_pi(theta + dt * (omega + c * rec_n * sum));
}/*}}}*/


void integrate_couplings(
    uniform uint i_speed,

    uniform uint i_step,
    uniform uint n_node,
    uniform uint n_step,
    uniform uint n_coupling,
    uniform uint n_speed,
    uniform float dt,
    uniform float speeds[],
    uniform float weights[],
    uniform float lengths[],
    uniform float couplings[],
    uniform float state_pwi[],
    uniform float tavg[]
)
{
    uniform const float speed_value = speeds[i_speed];
    uniform const float rec_n = 1.0f / n_node;
    uniform const float rec_speed_dt = 1.0f / speed_value / dt;
    uniform const float omega = 10.0 * 2.0 * M_PI_F / 1e3;
    uniform const float sig = sqrt(dt) * sqrt(2.0 * 1e-5);
    uniform const uint n_thread = n_coupling * n_speed;

    foreach (i_coupling = 0 ... n_coupling)
    {
        varying const uint id = i_speed * n_coupling + i_coupling;
        varying const float c = couplings[i_coupling];

        for (uniform uint i_node = 0; i_node < n_node; ++i_node)
            tavg[i_node*n_thread + id] = 0.0f;

        for (uniform uint t = i_step; t < (i_step + n_step); ++t)
        {
            for (uniform uint i_node = 0; i_node < n_node; ++i_node)
            {
                varying float theta_i = state(t, i_node);
                uniform uint i_n = i_node * n_node;
                varying float sum = 0.0f;

                for (uniform uint j_node = 0; j_node < n_node; ++j_node)
                {
                    uniform float wij = weights[i_n + j_node];
                    if (wij == 0.0)
                        continue;
                    uniform int dij = round(lengths[i_n + j_node] * rec_speed_dt);
                    varying float theta_j = state(t-dij+NH, j_node);
                    sum += wij * sin(theta_j - theta_i);
                } // j_node
                state(t+1, i_node) = next_theta(theta_i, dt, omega, c, rec_n, sum);
                tavg[i_node*n_thread + id] += sin(theta_i);
            } // for i_node
        } // for t
    } // for id
} // kernel integrate

export void integrate(
    uniform uint i_step,
    uniform uint n_node,
    uniform uint n_step,
    uniform uint n_coupling,
    uniform uint n_speed,
    uniform float dt,
    uniform float speeds[],
    uniform float weights[],
    uniform float lengths[],
    uniform float couplings[],
    uniform float state_pwi[],
    uniform float tavg[]
)
{
    for (uniform uint i_speed = 0; i_speed < n_speed; ++i_speed)
    {
        integrate_couplings(i_speed, i_step, n_node, n_step, n_coupling, n_speed,
            dt, speeds, weights, lengths, couplings, state_pwi, tavg);
    }
}


// vim: sw=4 sts=4 ts=8 et ai ft=ispc
