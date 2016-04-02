/**
 * Helper functions used for the diagnostic of the vdt routines.
 * They are not optimised for speed.
 * Authors: Danilo Piparo CERN
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

#ifndef VDTHELPER_H_
#define VDTHELPER_H_

#include <bitset>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include "inttypes.h"
// #include "x86intrin.h"
#include <cmath> //for log2
#include "time.h"
#include "sys/time.h"

#ifdef __APPLE__
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <unistd.h>
#endif


namespace{

// Establish the size of the double and single precision and the bitsets
constexpr double _tmp=0;
constexpr uint32_t dp_size_in_bits = sizeof(_tmp)*8;
using dp_bitset = std::bitset<dp_size_in_bits>;

}

namespace vdth{

//------------------------------------------------------------------------------

// Useful alias for some functions
using dpdpfunction = std::function<double(double)>;
using dpdpfunctionv = std::function<void (uint32_t, double*, double*)>;
using spspfunction = std::function<float(float)>;
using spspfunctionv = std::function<void(uint32_t, float*, float*)>;

using dpdp2function = std::function<double(double,double)>;
using dpdp2functionv = std::function<void (uint32_t, double*, double*, double*)>;
using spsp2function = std::function<float(float,float)>;
using spsp2functionv = std::function<void(uint32_t, float*, float*, float*)>;
//maybe for convenience
template<class T> using genfpfunction = std::function<T(T)>;
template<class T> using genfpfunctionv = std::function<void(uint32_t, T*, T*)>;
template<class T> using genfp2function = std::function<T(T,T)>;
template<class T> using genfp2functionv = std::function<void(uint32_t, T*, T*, T*)>;
//------------------------------------------------------------------------------
/// Useful union

union standard{
	double dp;
	float sp[2];
	uint64_t li;
};

//------------------------------------------------------------------------------

template<class T>
uint32_t inline getSizeInbits(const T x){
	return sizeof(x) * 8;
}

//------------------------------------------------------------------------------

/// Convert a fp into a uint64_t not optimised for speed
template<class T>
inline uint64_t fp2uint64(const T x){
	const uint32_t size = getSizeInbits(x);
	standard conv;
	conv.dp=0.;
	if (size==dp_size_in_bits)
		conv.dp=x;
	else
		conv.sp[0]=x;
	return conv.li;

}

//------------------------------------------------------------------------------
/// Convert a double into a bitset
template<class T>
inline const dp_bitset fp2bs( const T x ){
	dp_bitset const bits (fp2uint64(x));
	return bits;
}

//------------------------------------------------------------------------------
/// Print as a dp formatted bitset
template<class T>
const std::string getbsasstr(const T x){

	const uint32_t size = getSizeInbits(x);

	uint32_t offset = 0;
	uint32_t exp_size = 11;
	uint32_t mant_size = 52;
	if (size!=dp_size_in_bits){
		offset = 32;
		exp_size = 8;
		mant_size = 23;
	}

	// Convert the bitstream to string
	std::string bitset_as_string (fp2bs(x).to_string());

	std::ostringstream os;

	// sign
	os  << bitset_as_string[offset] << " ";
	// exponent
	for (unsigned int i=offset+1;i<offset+1+exp_size;i++)
		os <<  bitset_as_string[i];
	os << " ";
	//mantissa
	for (unsigned int i=offset+1+exp_size;i<offset+1+exp_size+mant_size;i++)
		os <<  bitset_as_string[i];

	return os.str();
}


//------------------------------------------------------------------------------
/// Returns most significative different bit dp
template <class T>
uint16_t diffbit(const T a,const T b ){
	/// make a xor
	uint64_t ia = fp2uint64(a);
	uint64_t ib = fp2uint64(b);
	uint64_t c = ia>ib? ia-ib : ib -ia;
	//uint64_t c = ia^ib;
	/// return the log2+1
	return log2(c)+1;
}

//------------------------------------------------------------------------------

///Check and print which instructions sets are enabled.
void print_instructions_info(){

	std::ostringstream os;
	os << "List of enabled instructions' sets:\n";

	os << " o SSE2 instructions set "
#ifndef __SSE2__
			<< "not "
#endif
			<< "enabled.\n"

			<< " o SSE3 instructions set "
#ifndef __SSE3__
			<< "not "
#endif
			<< "enabled.\n"

			<< " o SSE4.1 instructions set "
#ifndef __SSE4_1__
			<< "not "
#endif
			<< "enabled.\n"

			<< " o AVX instructions set "
#ifndef __AVX__
			<< "not "
#endif
			<< "enabled.\n";
	std::cout << os.str();
}

//------------------------------------------------------------------------------

/// Print the different bit of two fp numbers
template<class T>
void print_different_bit(const T a, const T b, const bool show_identical=true){

	std::cout.precision(10);
	std::cout << "Different bit between " << a << " and " << b
			<< " is " << diffbit(a,b) << std::endl;
	if (show_identical)
		std::cout << getbsasstr(a) << std::endl
		<< getbsasstr(b) << std::endl<< std::endl;
}


//------------------------------------------------------------------------------

/// Invoke two functions and print on screen their argument and different bits
template<class T>
void printFuncDiff(const std::string& func_name, std::function<T(T)> f1,std::function<T(T)> f2, const T x){
	std::cout << "Function " << func_name << "(" << x << ")" <<  std::endl;
	print_different_bit(f1(x),f2(x),true);
}

/// Invoke two functions and print on screen their argument and different bits
template<class T>
void printFuncDiff(const std::string& func_name, std::function<T(T,T)> f1,std::function<T(T,T)> f2, const T x, const T y){
	std::cout << "Function " << func_name << "(" << x << ", "<< y <<")" <<  std::endl;
	print_different_bit(f1(x,y),f2(x,y),true);
}

//------------------------------------------------------------------------------

/// Invoke two functions and print on screen their argument and different bits
template<class T>
void printFuncDiff(const std::string& func_name,
		genfpfunctionv<T> f1,
		genfpfunctionv<T> f2,
		T* x_arr,
		const uint32_t size){
	std::cout << "Function " << func_name << std::endl;
	T* res_1 = new T[size];
	f1(size,x_arr,res_1);
	T* res_2 = new T[size];
	f2(size,x_arr,res_2);
	for (uint32_t i=0;i<size;i++){
		std::cout << "Calculated in " << x_arr[i] << std::endl;
		print_different_bit(res_1[i],res_2[i],true);
	};
	delete [] res_1;
	delete [] res_2;
}

/// Invoke two functions and print on screen their argument and different bits
template<class T>
void printFuncDiff(const std::string& func_name,
		genfp2functionv<T> f1,
		genfp2functionv<T> f2,
		T* x_arr,
		T* y_arr,
		const uint32_t size){
	std::cout << "Function " << func_name << std::endl;
	T* res_1 = new T[size];
	f1(size,x_arr,y_arr,res_1);
	T* res_2 = new T[size];
	f2(size,x_arr,y_arr,res_2);
	for (uint32_t i=0;i<size;i++){
		std::cout << "Calculated in (" << x_arr[i] << ", " << y_arr[i] << ")" << std::endl;
		print_different_bit(res_1[i],res_2[i],true);
	};
	delete [] res_1;
	delete [] res_2;
}

//------------------------------------------------------------------------------
// Function tests
/// Test a fp function with a double (double) signatures
template<class T>
void printFuncDiff(const std::string& name,
		std::function<T(T)> fpfunction,
		std::function<T(T)> fpfunction_ref,
		T* fpvals,
		const uint32_t size){

	for (uint32_t i=0;i<size;i++)
		printFuncDiff ( name,
				(std::function<T(T)>) fpfunction,
				(std::function<T(T)>) fpfunction_ref,
				fpvals[i] );

}


//------------------------------------------------------------------------------
// Function tests
/// Test a fp function with a double (double) signatures
template<class T>
void printFuncDiff(const std::string& name,
		std::function<T(T,T)> fpfunction,
		std::function<T(T,T)> fpfunction_ref,
		T* fpvals1,
		T* fpvals2,
		const uint32_t size){

	for (uint32_t i=0;i<size;i++)
		printFuncDiff ( name,
				(std::function<T(T,T)>) fpfunction,
				(std::function<T(T,T)>) fpfunction_ref,
				fpvals1[i],
				fpvals2[i]);

}

//------------------------------------------------------------------------------
/// Get the clock cycles
class timer{
public:
	timer(){}
	~timer(){}
	void print(){
		const uint64_t nsecs=get_elapsed_time();
		std::cout << "Time elapsed: " << nsecs << " nanoseconds.\n";// ("
		//<< m_get_elapsed_clocks(nsecs) << " clock)\n";
	}
#if defined (__APPLE__)
	void inline start(){m_time1=mach_absolute_time();}
	void inline stop(){m_time2=mach_absolute_time();}
	uint64_t get_elapsed_time(){
		static mach_timebase_info_data_t    sTimebaseInfo;
		const uint64_t elapsed = m_time2 - m_time1;
		// Convert to nanoseconds.
		// Have to do some pointer fun because AbsoluteToNanoseconds
		// works in terms of UnsignedWide, which is a structure rather
		// than a proper 64-bit integer.

		if ( sTimebaseInfo.denom == 0 ) {
			(void) mach_timebase_info(&sTimebaseInfo);
		}

		// Do the maths. We hope that the multiplication doesn't
		// overflow; the price you pay for working in fixed point.

		uint64_t elapsedNano = elapsed * sTimebaseInfo.numer / sTimebaseInfo.denom;

		return elapsedNano;
	}

private:
	uint64_t m_time1,m_time2;

#else
	void inline start(){
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_time1);
	}
	void inline stop(){
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &m_time2);
	}

	/// Return time in nanoseconds

	uint64_t get_elapsed_time(){
		timespec temp;
		temp.tv_sec = m_time2.tv_sec-m_time1.tv_sec;
		temp.tv_nsec = m_time2.tv_nsec-m_time1.tv_nsec;
		uint64_t elapsed_time = temp.tv_nsec;
		elapsed_time += 1e9*temp.tv_sec;
		return elapsed_time;
	}

private:
	timespec m_time1,m_time2;
#endif

};

//------------------------------------------------------------------------------
// inline uint64_t getcpuclock() {
//  return __rdtsc();
// }


}//end of namespace vdth
#endif
