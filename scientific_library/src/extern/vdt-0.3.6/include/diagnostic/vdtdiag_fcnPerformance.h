#ifndef VDTDIAG_FCNPERFORMANCE_H_
#define VDTDIAG_FCNPERFORMANCE_H_

#include <iomanip>
#include "vdtdiag_helper.h"

/**
 * Class that represents the CPU performance of a mathematical function.
 * The quantities stored are the number mean time per execution and the 
 * associated error. Two constructors are available: one for the scalar and one 
 * for the vector signature.
 * TODO:
 *  o Write the timings on disk on an ascii file
 *
 * Author Danilo Piparo 
 * 
**/

/* 
 * VDT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//-----------------------------------------------------------------------------

template<class T>
class fcnPerformance{
public:

//-----------------------------------------------------------------------------
/** 
* Construct from name, input and scalar function and number of repetitions.
* Scalar signature.
**/
fcnPerformance(const std::string& fcnName,
               const std::vector<T>& input, 
               const std::function<T(T)> fcn,
               const uint32_t repetitions=10):
                m_fcn_name(fcnName),
                m_input(input),
                m_input2(input),
                m_fcn(fcn),
                m_repetitions(repetitions){

  measureTime();
  
}

//-----------------------------------------------------------------------------
fcnPerformance(const std::string& fcnName,
               const std::vector<T>& input1,
               const std::vector<T>& input2,
               const std::function<T(T,T)> fcn,
               const uint32_t repetitions=10):
                m_fcn_name(fcnName),
                m_input(input1),
                m_input2(input2),
                m_fcn2D(fcn),
                m_repetitions(repetitions){

  measureTime2D();

}

//-----------------------------------------------------------------------------
/** 
* Construct from name, input and scalar function and number of repetitions.
* Array signature.
**/
fcnPerformance(const std::string& fcnName,
               const std::vector<T>& input,
               std::function<void(uint32_t,T*,T* )> fcnv,
               const uint32_t repetitions=10):
                m_fcn_name(fcnName),
                m_input(input),
                m_input2(input),
                m_fcnv(fcnv),
                m_repetitions(repetitions){

  measureTimev();

}

//------------------------------------------------------------------------------
fcnPerformance(const std::string& fcnName,
               const std::vector<T>& input1,
               const std::vector<T>& input2,
               std::function<void(uint32_t,T*,T*,T* )> fcnv,
               const uint32_t repetitions=10):
                m_fcn_name(fcnName),
                m_input(input1),
                m_input2(input2),
                m_fcn2Dv(fcnv),
                m_repetitions(repetitions){

  measureTime2Dv();

}

//------------------------------------------------------------------------------


/// Nothing to do here
~fcnPerformance(){}

//------------------------------------------------------------------------------
/// Print the timing on screen. If a scale is povided, express the time in terms of a speedup factor.
  void print(std::ostream& stream = std::cout, const double scale=1.){
    stream << std::setprecision(2)
              << std::fixed
              << "Function " << std::setw(15) << std::left << m_fcn_name << " : " 
              << m_avg_time << " +- " << m_avg_time_err 
              << " ns";
    if (scale!=1.)
      stream << " --> " << scale/m_avg_time << "X speedup!";
    stream << std::endl;
  }

//-----------------------------------------------------------------------------
/// Get the mean elapsed time per call
const T getAvg()const {
  return m_avg_time;
}
//-----------------------------------------------------------------------------
/// Get the error on the mean elapsed time per call
const T getAvgErr()const {
  return m_avg_time_err;
}

//-----------------------------------------------------------------------------

private:
  /// The name of the benchmarked function
  const std::string m_fcn_name;
  /// Const reference to the input values
  const std::vector<T>& m_input;
  /// Const reference to the input values
  const std::vector<T>& m_input2;
  /// Scalar function 
  const std::function<T(T)> m_fcn;
  /// Scalar function
  const std::function<T(T,T)> m_fcn2D;
  /// Array function (cannot coexist with scalar)
  const std::function<void(uint32_t,T*,T*)> m_fcnv;
  /// Array function (cannot coexist with scalar)
  const std::function<void(uint32_t,T*,T*,T*)> m_fcn2Dv;
  /// Number of repetitions of the measurement for stability
  const uint32_t m_repetitions;
  /// Mean time
  double m_avg_time;
  /// Error on the mean time
  double m_avg_time_err;  

//-----------------------------------------------------------------------------
/// Measure the timings of the function, scalar signature
  void measureTime(){
  
  // momenta to calculate mean and rms, they will be filled later
  uint64_t t=0;
  uint64_t t2=0;

  // An useful quantity
  const uint32_t size=m_input.size();

  // Allocate the array of results. Necessary to circumvent compiler optimisations
  double* results_arr = new double[size];
  // Set up some warm-up iterations
  const uint32_t warm_up = m_repetitions;
  // Allocate once the delta outside the loop
  uint64_t deltat=0;  
  // The timer which is used to mesure the time interval
  vdth::timer fcntimer;  
  // Start the loop on the repetitions
  for (uint32_t irep=0;irep<m_repetitions+warm_up;irep++){
    // Isolate the array inside the vector
    const T* input_arr=&m_input[0];

    // Perform the measurment on _all_ the input values
    fcntimer.start();
    for (uint32_t i=0;i<size;++i)
      results_arr[i] = m_fcn(input_arr[i]);
    fcntimer.stop();        
    
    // Record only if part of the warm-up
    if (irep>=warm_up){
      deltat=fcntimer.get_elapsed_time();      
      t+=deltat;
      t2+=deltat*deltat;
      }
      
    // To avoid optimisations, call a dummy function
    std::vector<T> results(results_arr,results_arr+size);
    fool_optimisation(results);      
    }
  delete [] results_arr;
  
  // Calculate mean and error on the mean
  const uint64_t iterations = size * m_repetitions;
  calculate_mean_and_err(t,t2,iterations);
  }
  //-----------------------------------------------------------------------------
  /// Measure the timings of the function, scalar signature
    void measureTime2D(){

    // momenta to calculate mean and rms, they will be filled later
    uint64_t t=0;
    uint64_t t2=0;

    // An useful quantity
    const uint32_t size=m_input.size();

    // Allocate the array of results. Necessary to circumvent compiler optimisations
    double* results_arr = new double[size];
    // Set up some warm-up iterations
    const uint32_t warm_up = m_repetitions;
    // Allocate once the delta outside the loop
    uint64_t deltat=0;
    // The timer which is used to mesure the time interval
    vdth::timer fcntimer;
    // Start the loop on the repetitions
    for (uint32_t irep=0;irep<m_repetitions+warm_up;irep++){
      // Isolate the array inside the vector
      const T* input_arr1=&m_input[0];
      const T* input_arr2=&m_input2[0];

      // Perform the measurment on _all_ the input values
      fcntimer.start();
      for (uint32_t i=0;i<size;++i)
        results_arr[i] = m_fcn2D(input_arr1[i],input_arr2[i]);
      fcntimer.stop();

      // Record only if part of the warm-up
      if (irep>=warm_up){
        deltat=fcntimer.get_elapsed_time();
        t+=deltat;
        t2+=deltat*deltat;
        }

      // To avoid optimisations, call a dummy function
      std::vector<T> results(results_arr,results_arr+size);
      fool_optimisation(results);
      }
    delete [] results_arr;

    // Calculate mean and error on the mean
    const uint64_t iterations = size * m_repetitions;
    calculate_mean_and_err(t,t2,iterations);
    }
//-----------------------------------------------------------------------------
/// Measure the timings of the function, array signature
  void measureTimev(){
  // See explainations in the scalar method!
  uint64_t t=0.;
  uint64_t t2=0.;

  const uint32_t size=m_input.size();

  const uint32_t warm_up = m_repetitions;
  uint64_t deltat=0;
  vdth::timer fcntimer;

  for (uint32_t irep=0;irep<m_repetitions+warm_up;irep++){
    
      T* input_arr= const_cast<T*> (&m_input[0]);
      T* results_arr=new T[size];
      
      fcntimer.start();
      m_fcnv(size,input_arr,results_arr);
      fcntimer.stop();      

      if (irep>=warm_up){
      	deltat = fcntimer.get_elapsed_time();
        t+=deltat;
        t2+=deltat*deltat;
        }

     std::vector<T> results(results_arr,results_arr+size);
     delete[] results_arr;
     fool_optimisation(results);
    }
    
  const uint64_t iterations = size * m_repetitions;
  calculate_mean_and_err(t,t2,iterations);
  }

  //-----------------------------------------------------------------------------
  /// Measure the timings of the function, array signature
void measureTime2Dv(){
	// See explainations in the scalar method!
		uint64_t t=0.;
		uint64_t t2=0.;

		const uint32_t size=m_input.size();

		const uint32_t warm_up = m_repetitions;
		uint64_t deltat=0;
		vdth::timer fcntimer;

		for (uint32_t irep=0;irep<m_repetitions+warm_up;irep++){

				T* input_arr1= const_cast<T*> (&m_input[0]);
				T* input_arr2= const_cast<T*> (&m_input2[0]);
				T* results_arr=new T[size];

				fcntimer.start();
				m_fcn2Dv(size,input_arr1,input_arr2,results_arr);
				fcntimer.stop();

				if (irep>=warm_up){
					deltat = fcntimer.get_elapsed_time();
					t+=deltat;
					t2+=deltat*deltat;
					}

			 std::vector<T> results(results_arr,results_arr+size);
			 delete[] results_arr;
			 fool_optimisation(results);
			}

		const uint64_t iterations = size * m_repetitions;
		calculate_mean_and_err(t,t2,iterations);
		}

//------------------------------------------------------------------------------
/// Loop on the values in order to force the compiler to actually calculate them for real
void fool_optimisation(const std::vector<T>& results){
    for (const T& res:results )
      if (res == -0.123)
        std::cout << "To fool the compiler's optimisations!\n";
 }

//------------------------------------------------------------------------------
/// Calculate Mean elapsed time and error on the mean
void calculate_mean_and_err(const double t, const double t2,const uint64_t iterations){
  
  // Mean is easy
  m_avg_time = t / iterations;

  // Calculate the error on the mean
  // RMS, one dof is gone for the mean, so iterations-1
  const double rms2=(double(t2) - iterations*m_avg_time*m_avg_time)/(iterations-1);
  const double rms=sqrt(rms2);
  m_avg_time_err = rms / sqrt(iterations); //2 sqrts, but we go for clarity here.

}

//------------------------------------------------------------------------------
  
};


#endif 
