/*
    Native TVB Simulation Core
    ==========================

   This device kernel file defines a set of functions that, in conjunction
   with code snippets from TVB classes, allows us to run a majority of the
   TVB simulator's systems on the GPU in a massively parallel configuration,
   or on the CPU at native code speed.

   This is not currently intended for regular user consumption.

    some naming conventions to keep us sane

        n___ -> number of ___ (no s at the end)
        i___ -> index of ___

   CONTENTS

   - global constant variables (irony noted thx)
   - globals' getter & setter functions
   - wrap function
   - model_dfun
   - noise_gfun
   - integrator
   - coupling
   - update (main step function, entry point from Python)


   marmaduke woodman <mw@eml.cc>

*/

// static on CPU, device constants on GPU (we get 64 kB only)
// cf http://wwwae.ciemat.es/~cardenas/CUDA/T6-ConstantMemory.pdf
#ifdef TVBGPU
#define DEFGLOBAL __device__ __constant__
#endif
#ifndef TVBGPU
#define DEFGLOBAL static
#endif
DEFGLOBAL int horizon; // max(idel) + 1
DEFGLOBAL int n_node;  // number of nodes
DEFGLOBAL int n_thr ;  // number of threads on GPU
DEFGLOBAL int n_rthr;  // number of non-padded threads on GPU
DEFGLOBAL int n_svar;  // number of state variables
DEFGLOBAL int n_cvar;  // number of coupling variables
DEFGLOBAL int n_cfpr;  // number of coupling function parameters
DEFGLOBAL int n_mmpr;  // number of mass model parameters
DEFGLOBAL int n_nspr;  // number of noise parameters
DEFGLOBAL int n_inpr;  // number of integration parameters
DEFGLOBAL int n_tavg;  // number of time points to average over
DEFGLOBAL int n_msik;  // number of steps to take in kernel
DEFGLOBAL int n_mode;  // number of modes in the mass model
#ifdef DEFGLOBAL
#undef DEFGLOBAL
#endif

// access globals when not GPU
#ifndef TVBGPU
#define DEFACCESS(name)\
    void set_##name(int _##name){name=_##name;}\
    int get_##name() {return name;}

DEFACCESS(horizon)
DEFACCESS(n_node)
DEFACCESS(n_thr)
DEFACCESS(n_rthr)
DEFACCESS(n_svar)
DEFACCESS(n_cvar)
DEFACCESS(n_cfpr)
DEFACCESS(n_mmpr)
DEFACCESS(n_nspr)
DEFACCESS(n_inpr)
DEFACCESS(n_tavg)
DEFACCESS(n_msik)
DEFACCESS(n_mode)
#endif /* ifndef TVBGPU */


/* wrap implements the behavior of Python's modulo operator */
#ifdef TVBGPU
__device__ 
#endif
int wrap(int i_step)
{
  if (i_step>=0)
    return i_step % horizon;
  else
    if ((-i_step) % horizon == 0)
      return 0;
    else
      return horizon + (i_step % horizon);
}


/* wrapper for model specific code computing RHSs of diff-eqs */
#ifdef TVBGPU
__device__ 
#endif
void model_dfun(
        float * _dx, float *_x, float *mmpr, float*input // ting
    )
{
#define X(i) _x[n_thr*i]
#define DX(i) _dx[n_thr*i]
#define P(i) mmpr[n_thr*i]
#define I(i) input[i]

    // begin model code
$model_dfun
    // end model specific code

#undef X
#undef DX
#undef P
#undef I
}



#ifdef TVBGPU
__device__ 
#endif
void noise_gfun(
        float *gx, float *x, float *nspr  // out, state, noise pars
    )
{
#define X(i) x[n_thr*i]
#define GX(i) gx[n_thr*i]
#define P(i) nspr[n_thr*i]

    // begin (linear additive) noise definition
$noise_gfun
    // end noise definition

#undef X
#undef GX
#undef P
}


#ifdef TVBGPU
__device__ 
#endif
void integrate(
        float *x, float *dx1, float *dx2, float *gx, float *ns,     // state/workspace arrays
        float *inpr, float *nspr, float *mmpr, float *input, float *stim //rator, noise & model parameter arrays
    )
{
#define X(i)     x   [n_thr*i]
#define DX1(i) dx1   [n_thr*i]
#define DX2(i) dx2   [n_thr*i] // used in Heun method! don't remove!
#define GX(i)   gx   [n_thr*i]
#define P(i)  inpr   [i]
#define NS(i)   ns   [n_thr*i]
#define STIM(i) stim [n_thr*i]

    // begin integration scheme code
$integrate
    // end scheme code

#undef X
#undef DX1
#undef DX2
#undef GX
#undef P
#undef NS
#undef STIM
}


/* 
   history       [                       i_step                  i_node           i_svar       i_thr   ]
    .shape                               n_step                  n_node           n_svar       n_thr
    .strides        n_node*n_svar*n_thr            n_svar*n_thr            n_thr            1
    ND -> lin      (n_node*n_svar*n_thr)*i_step + (n_svar*n_thr)*i_node + (n_thr)*i_svar + (1)*i_thr
    
    ND->lin factored        i_thr + n_thr*(i_svar + n_svar*(i_node + n_node*i_step))

    but, note that below, cvar used instead of svar because history only stores
    the history of the coupling variables. like, duh.

*/

#ifdef TVBGPU
__device__ 
#endif
void coupling(
        float *input, float *x,                             // output & state
        int * idel, float * conn, float * hist,             // connectivity & history                       
        float *cfpr, int *cvars,                            // coupling function parameters
        int i_step, int i_node                              // indexing
    )
{
#define I input[n_thr*i_cvar]
#define HISTORY_INDEX (n_thr*(i_cvar + n_cvar*(j_node + n_node*wrap(i_step - 1 - *idel))))
#define XJ hist[HISTORY_INDEX]
#define XI x[n_thr*(*cvars)]
#define GIJ (*conn)
#define P(i) cfpr[i]

    // note data pointers' original value
    int   *idel_ = idel;
    float *conn_ = conn;

    // iter over coupling vars
    for (int i_cvar=0; i_cvar<n_cvar; i_cvar++, cvars++)
    {

        // begin coupling function
$coupling
        // end coupling function

        // restore pointer values
        idel = idel_;
        conn = conn_;
    }


#undef I
#undef HISTORY_INDEX
#undef XJ
#undef XI
#undef GIJ
#undef P
}


/* advance simulation one step */
#ifdef TVBGPU
__global__ 
#endif
void update(

    // thread invariant, call varying
    int i_step,             // index of current step

    // thread invariant, call invariant
    int *idel,              // int(delay matrix / dt), shape == (n_nodes, n_nodes)
    int *cvars,             // coupling variable indices, shape == (n_cvar, ) 
    float *inpr,            // integration parameters

    // possibly, but not currently, thread varying, call invariant
    float *conn,            // connectivity matrix, shape == (n_nodes, n_nodes)
    float *cfpr,            // coupling func parameters, shape == (n_pars, )

    // thread varying, call invariant
    float *nspr,                // .shape == (        n_nodes, n_nspr, n_thr)
    float *mmpr,                // .shape == (        n_nodes, n_mmpr, n_thr)

    // thread varying, call varying 
    float *input,               // .shape == (               , n_cvars, n_thr)
    float *x,                   // .shape == (        n_nodes, n_svar, n_thr) 
    float *hist,                // .shape == (horizo, n_nodes, n_cvars, n_thr)
    float *dx1,                 // .shape == (               , n_svar, n_thr) 
    float *dx2,                 // .shape == (               , n_svar, n_thr) 
    float *gx,                  // .shape == (               , n_svar, n_thr) 
    float *ns,                  // .shape == (        n_nodes, n_svar, n_thr) 
    float *stim,                // .shape == (        n_nodes, n_svar, n_thr)
    float *tavg                 // .shape == (        n_nodes, n_svar, n_thr)
    )

{

#ifdef TVBGPU
    // for simplicity we only use one dimension of the software coord sys
    int i_thr = blockDim.x*blockIdx.x + threadIdx.x;

    // don't run the padded threads
    if (i_thr < n_rthr) { 
#else

    int i_thr = 0;

 #ifdef TVBOMP
  #pragma omp parallel private(i_thr)
    { // eventually use many cores on same machine...
    i_thr = omp_get_thread_num();
 #endif

#endif

    // per thread pointers
    float *nspr_  = nspr  + i_thr
        , *mmpr_  = mmpr  + i_thr   // most of these workspace arrays can be
        , *input_ = input + i_thr   // aligned per thread so that when it 
        , *x_     = x     + i_thr   // is indexed, no need to add i_thr at
        , *hist_  = hist  + i_thr   // every access. stride of thr is 1, so
        , *dx1_   = dx1   + i_thr   // it's necessary to stride the -2 index
        , *dx2_   = dx2   + i_thr   // only by n_thr. upshow is that all
        , *gx_    = gx    + i_thr   // memory access to these arrays is 
        , *ns_    = ns    + i_thr   // aligned across all threads.
        , *stim_  = stim  + i_thr 
        , *tavg_  = tavg  + i_thr;

    // per node pointers
    float *_x,  *_hist,  *_ns, *_stim,  *_nspr,  *_mmpr, *_conn;
    int *_idel; 

    // multistep in-kernel
    for (int i_msik=0; i_msik<n_msik; i_msik++, i_step++)
    {

        // (re)set per node pointers for this step
        _x = x_; _hist = hist_; _ns = ns_; _stim = stim_; _nspr = nspr_; 
        _mmpr = mmpr_; _idel = idel, _conn = conn;


        for (int i_node=0; i_node<n_node; i_node++)
        {   // C77, cousin of F77
            coupling(input_, _x, _idel, _conn, hist_, cfpr, cvars, i_step, i_node);
            integrate(_x, dx1_, dx2_, gx_, _ns, inpr, _nspr, _mmpr, input, _stim);

            // certain arrays must be arg'd w/ offset because other functions
            // don't index w.r.t. i_node:
            _x    += n_svar*n_thr;
            _ns   += n_svar*n_thr;
            _stim += n_svar*n_thr;
            _nspr += n_nspr*n_thr; // n_nspr should take account for n_svar
            _mmpr += n_mmpr*n_thr;

            // idel and conn increment by n_node but not threads
            _idel += n_node;
            _conn += n_node;
        }

        // point _hist to per thread target, copy new state
        _hist = hist_ + n_thr*n_node*wrap(i_step)*n_cvar;
        for (int i_node=0; i_node<n_node; i_node++)
            for (int i_cvar=0; i_cvar<n_cvar; i_cvar++)
                _hist[n_thr*(i_cvar + n_cvar*i_node)] = x_[n_thr*(cvars[i_cvar] + n_svar*i_node)];

        /*
           Temporal averaging logic:

                if i_step % n_tavg == 0:
                    tavg[:] = x
                else:
                    tavg += x/n_tavg

            then caller does

                if i_step % n_tavg == n_tavg-1:
                    grab_a_copy_of_tavg()

           */

#define ALL for (int i=0; i<n_node*n_svar; i++) 
        if (n_tavg>0)
        {
            if ((i_step % n_tavg) == 0) ALL tavg_[i] = x_[i];
            else
            {
                float norm = 1.0/n_tavg;
                ALL tavg_[i] += norm*x_[i];
            }
        }
#undef ALL
    }

 #ifdef TVBOMP
    }
 #endif

 #ifdef TVBGPU
    } // end if (i_thr < n_rthr) guard for padding threads
 #endif


} // end __global__ void step(...)


