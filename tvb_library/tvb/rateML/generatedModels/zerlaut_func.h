// prototypes
__device__ float TF_excitatory(float fe, float fi, float fe_ext, float fi_ext, float W, float E_L_e);
__device__ float TF_inhibitory(float fe, float fi, float fe_ext, float fi_ext, float W, float E_L_i);
__device__ float TF(float fe, float fi, float fe_ext, float fi_ext, float W,
float P0, float P1, float P2, float P3, float P4, float P5, float P6, float P7, float P8, float P9, float E_l) ;
__device__ void get_fluct_regime_vars(float Fe, float Fi, float Fe_ext, float Fi_ext, float W,
                    float E_l, float *mu_V, float *sigma_V, float *T_V);
__device__ float threshold_func(float muV, float sigmaV, float TvN, float P0, float P1, float P2, float P3, float P4,
                                float P5, float P6, float P7, float P8, float P9);
__device__ float estimate_firing_rate(float mu_V, float sigma_V, float T_V, float V_thre);


// util functions
__device__ float TF_excitatory(float fe, float fi, float fe_ext, float fi_ext, float W, float E_L_e) {

        /*
        transfer function for excitatory population
        :param fe: firing rate of excitatory population
        :param fi: firing rate of inhibitory population
        :param fe_ext: external excitatory input
        :param fi_ext: external inhibitory input
        :param W: level of adaptation
        :return: result of transfer function
        */

        return TF(fe, fi, fe_ext, fi_ext, W, P_e0, P_e1, P_e2, P_e3, P_e4, P_e5, P_e6, P_e7, P_e8, P_e9, E_L_e);
}

__device__ float TF_inhibitory(float fe, float fi, float fe_ext, float fi_ext, float W, float E_L_i) {

        /*
        transfer function for inhibitory population
        :param fe: firing rate of excitatory population
        :param fi: firing rate of inhibitory population
        :param fe_ext: external excitatory input
        :param fi_ext: external inhibitory input
        :param W: level of adaptation
        :return: result of transfer function
        */

        return TF(fe, fi, fe_ext, fi_ext, W, P_i0, P_i1, P_i2, P_i3, P_i4, P_i5, P_i6, P_i7, P_i8, P_i9, E_L_i);
}

//header with global definitions for consts
__device__ float TF(float fe, float fi, float fe_ext, float fi_ext, float W,
float P0, float P1, float P2, float P3, float P4, float P5, float P6, float P7, float P8, float P9, float E_l) {

	/*
	transfer function for inhibitory population
	Inspired from the next repository :
	https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
	:param fe: firing rate of excitatory population
	:param fi: firing rate of inhibitory population
	:param fe_ext: external excitatory input
	:param fi_ext: external inhibitory input
	:param W: level of adaptation
	:param P: Polynome of neurons phenomenological threshold (order 9)
	:param E_L: leak reversal potential
	:return: result of transfer function
	*////////

    // local variable for pointers
    float mu_V;
    float sigma_V;
    float T_V;

	get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, E_l, &mu_V, &sigma_V, &T_V);


	float V_thre = threshold_func(mu_V, sigma_V, T_V*g_L/C_m,
								 P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
	V_thre *= 1e3;  // the threshold need to be in mv and not in Volt
	return estimate_firing_rate(mu_V, sigma_V, T_V, V_thre);
}

 __device__ void get_fluct_regime_vars(float Fe, float Fi, float Fe_ext, float Fi_ext, float W,
                    float E_l, float *mu_V, float *sigma_V, float *T_V)
{
    /*
    Compute the mean characteristic of neurons.
    Inspired from the next repository :
    https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
    :param Fe: firing rate of excitatory population
    :param Fi: firing rate of inhibitory population
    :param Fe_ext: external excitatory input
    :param Fi_ext: external inhibitory input
    :param W: level of adaptation
    :param Q_e: excitatory quantal conductance
    :param tau_e: excitatory decay
    :param E_e: excitatory reversal potential
    :param Q_i: inhibitory quantal conductance
    :param tau_i: inhibitory decay
    :param E_i: inhibitory reversal potential
    :param E_L: leakage reversal voltage of neurons
    :param g_L: leak conductance
    :param C_m: membrane capacitance
    :param E_L: leak reversal potential
    :param N_tot: cell number
    :param p_connect_e: connectivity probability of excitatory neurons
    :param p_connect_i: connectivity probability of inhibitory neurons
    :param g: fraction of inhibitory cells
    :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
    *///////

    // firing rate
    // 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
    float fe = (Fe+1.0e-6)*(1.-g)*p_connect_e*N_tot + Fe_ext*K_ext_e;
    float fi = (Fi+1.0e-6)*g*p_connect_i*N_tot + Fi_ext*K_ext_i;

    // conductance fluctuation and effective membrane time constant
    float mu_Ge = Q_e*tau_e*fe;
    float mu_Gi = Q_i*tau_i*fi;  // Eqns 5 from [MV_2018]
    float mu_G = g_L+mu_Ge+mu_Gi;  // Eqns 6 from [MV_2018]
    float T_m = C_m/mu_G; // Eqns 6 from [MV_2018]

    // membrane potential
    *mu_V = (mu_Ge*E_e+mu_Gi*E_i+g_L*E_l-W)/mu_G;  // Eqns 7 from [MV_2018]
    // post-synaptic membrane potential event s around muV
    float U_e = Q_e/mu_G*(E_e-*mu_V);
    float U_i = Q_i/mu_G*(E_i-*mu_V);
    // Standard deviation of the fluctuations
    // Eqns 8 from [MV_2018]
    float powfe = powf((U_e*tau_e), 2);
    float powfi = powf((U_i*tau_i), 2);

    *sigma_V = sqrt(fe*powfe/(2.*(tau_e+T_m))+fi*powfi/(2.*(tau_i+T_m)));
    // Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
    float T_V_numerator = (fe*powfe + fi*powfi);
    float T_V_denominator = (fe*powfe/(tau_e+T_m) + fi*powfi/(tau_i+T_m));
    // T_V = numpy.divide(T_V_numerator, T_V_denominator, out=numpy.ones_like(T_V_numerator),
    //                    where=T_V_denominator != 0.0) # avoid numerical error but not use with numba
    *T_V = T_V_numerator/T_V_denominator;
    return;
}

//@staticmethod
//@jit(nopython=True,cache=True)
//def threshold_func(muV, sigmaV, TvN, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9):
__device__ float threshold_func(float muV, float sigmaV, float TvN, float P0, float P1, float P2, float P3, float P4,
                                float P5, float P6, float P7, float P8, float P9) {
    /*
    The threshold function of the neurons
    :param muV: mean of membrane voltage
    :param sigmaV: variance of membrane voltage
    :param TvN: autocorrelation time constant
    :param P: Fitted coefficients of the transfer functions
    :return: threshold of neurons
    *////////////

    // Normalization factors page 48 after the equation 4 from [ZD_2018]
    float muV0= -60.0, DmuV0 = 10.0;
    float sV0 = 4.0, DsV0 = 6.0;
    float TvN0 = 0.5, DTvN0 = 1.0;
    float V = (muV-muV0)/DmuV0;
    float S = (sigmaV-sV0)/DsV0;
    float T = (TvN-TvN0)/DTvN0;

    // Eqns 11 from [MV_2018]
    return P0 + P1*V + P2*S + P3*T + powf((P4*V),2) + powf((P5*S),2) + powf((P6*T),2) + P7*V*S + P8*V*T + P9*S*T;
}

//@staticmethod
//def estimate_firing_rate(muV, sigmaV, Tv, Vthre):
__device__ float estimate_firing_rate(float mu_V, float sigma_V, float T_V, float V_thre) {
    /*
    The threshold function of the neurons
    :param muV: mean of membrane voltage
    :param sigmaV: variance of membrane voltage
    :param Tv: autocorrelation time constant
    :param Vthre:threshold of neurons
    */
    // Eqns 10 from [MV_2018]
//    printf("mu_V T_V sigma_V V_thre %f %f %f %f\n", mu_V, sigma_V, T_V, V_thre);
//    printf("erfc %f\n", erfcf((V_thre-mu_V) / (SQRT2*sigma_V)) / (2*T_V)  );

    return erfcf((V_thre-mu_V) / (SQRT2*sigma_V)) / (2*T_V);
}