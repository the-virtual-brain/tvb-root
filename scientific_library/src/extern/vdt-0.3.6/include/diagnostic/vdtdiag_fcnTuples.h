/* vdtdiag_fcnTuples.h
*
*	created: 12.7.2012
*
*	Contains two functions (scalar, vector form)
*	that fill a std::vector with std::tuple(s) binding
*	together a function, a function name and a random pool
*	of appropriate range.
*
*	This function should serve as a central resource for tuples,
*	function names alone and so on.
*
*	Author: Ladislav Horky, Danilo Piparo
*/

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

#ifndef _VDTDIAG_TUPLES_
#define _VDTDIAG_TUPLES_

#include "vdtdiag_helper.h"
#include "vdtdiag_random.h"
#include "vdtMath.h"
#include <tuple>
#include <string>


//external libs
#include "externalLibcfg.h"
#ifdef _VC_AVAILABLE_
#include "vdtdiag_vcWrapper.h"
#endif

using namespace vdt;

template<class T> using genfpfcn_tuple = std::tuple<const std::string, vdth::genfpfunction<T>,const std::vector<T>& >;
template<class T> using genfpfcnv_tuple = std::tuple<const std::string, vdth::genfpfunctionv<T>,const std::vector<T>& >;
template<class T> using genfpfcn2D_tuple = std::tuple<const std::string,
																											vdth::genfp2function<T>,
																											const std::vector<T>&,
																											const std::vector<T>&>;
template<class T> using genfpfcn2Dv_tuple = std::tuple<const std::string,
																											 vdth::genfp2functionv<T>,
																											 const std::vector<T>&,
																											 const std::vector<T>&>;

/// Fills vector passed in first parameter with fcn tuples based on random pools passed in following parameters
void getFunctionTuples(	std::vector<genfpfcn2D_tuple<double>>* fcn_tuples,
			            			randomPool2D<double>& moneone2Pool){
    fcn_tuples->clear();

    fcn_tuples->push_back(genfpfcn2D_tuple<double>( "Identity2D", identity2D, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2D_tuple<double>( "Atan2", atan2, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2D_tuple<double>( "Fast_Atan2", fast_atan2, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));

}

void getFunctionTuples(	std::vector<genfpfcn2D_tuple<float>>* fcn_tuples,
					            	randomPool2D<float>& moneone2Pool){
    fcn_tuples->clear();

    fcn_tuples->push_back(genfpfcn2D_tuple<float>( "Identity2Df", identity2Df, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2D_tuple<float>( "Atan2f", atan2f, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2D_tuple<float>( "Fast_Atan2f", fast_atan2f, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));

}

/// Fills vector passed in first parameter with fcn tuples based on random pools passed in following parameters
void getFunctionTuplesvect(	std::vector<genfpfcn2Dv_tuple<double>>* fcn_tuples,
					             	randomPool2D<double>& moneone2Pool){
    fcn_tuples->clear();

    fcn_tuples->push_back(genfpfcn2Dv_tuple<double>( "Identity2Dv", identity2Dv, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2Dv_tuple<double>( "Atan2v", atan2v, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2Dv_tuple<double>( "Fast_Atan2v", fast_atan2v, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));

}

/// Fills vector passed in first parameter with fcn tuples based on random pools passed in following parameters
void getFunctionTuplesvect(	std::vector<genfpfcn2Dv_tuple<float>>* fcn_tuples,
					             	randomPool2D<float>& moneone2Pool){
    fcn_tuples->clear();

    fcn_tuples->push_back(genfpfcn2Dv_tuple<float>( "Identity2Dfv", identity2Dfv, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2Dv_tuple<float>( "Atan2fv", atan2fv, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));
    fcn_tuples->push_back(genfpfcn2Dv_tuple<float>( "Fast_Atan2fv", fast_atan2fv, moneone2Pool.getNumbersX(), moneone2Pool.getNumbersY() ));

}

/// Fills vector passed in first parameter with fcn tuples based on random pools passed in following parameters
void getFunctionTuples(	std::vector<genfpfcn_tuple<double>>* fcn_tuples,
						randomPool<double>& symmetricrPool,
						randomPool<double>& positiverPool,
						randomPool<double>& mone2onerPool,
						randomPool<double>& expPool){
    fcn_tuples->clear();

    fcn_tuples->push_back(genfpfcn_tuple<double>( "Identity", identity, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Exp", exp, expPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Log", log, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Sin", sin, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Cos", cos, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Tan", tan, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Asin", asin, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Acos", acos, mone2onerPool.getNumbers() )); 
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Atan", atan, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Isqrt", isqrt, positiverPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Inverse", inv, symmetricrPool.getNumbers() ));    

    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Exp", fast_exp, expPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Log", fast_log, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Sin", fast_sin, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Cos", fast_cos, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Tan", fast_tan, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Asin", fast_asin, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Acos", fast_acos, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Atan", fast_atan, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Isqrt", fast_isqrt, positiverPool.getNumbers() ));     
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Inv", fast_inv, symmetricrPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Apr_Isqrt", fast_approx_isqrt, positiverPool.getNumbers() ));       
    fcn_tuples->push_back(genfpfcn_tuple<double>( "Fast_Apr_Inv", fast_approx_inv, symmetricrPool.getNumbers() ));    

#ifdef _VC_AVAILABLE_
	fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Identity", vc_identity, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Log", vc_log, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Sin", vc_sin, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Cos", vc_cos, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Asin", vc_asin, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Atan", vc_atan, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Isqrt", vc_isqrt, positiverPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcn_tuple<double>( "VC_Inverse", vc_inv, symmetricrPool.getNumbers() ));  
#endif
}

void getFunctionTuples(	std::vector<genfpfcn_tuple<float>>* fcn_tuples,
						randomPool<float>& symmetricrPool,
						randomPool<float>& positiverPool,
						randomPool<float>& mone2onerPool,
						randomPool<float>& expPool){
	fcn_tuples->clear();

    fcn_tuples->push_back(genfpfcn_tuple<float>( "Identityf", identityf, symmetricrPool.getNumbers() ));            
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Expf", expf, expPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Logf", logf, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Sinf", sinf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Cosf", cosf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Tanf", tanf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Asinf", asinf, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Acosf", acosf, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Atanf", atanf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Isqrtf", isqrtf, positiverPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Inversef", invf, symmetricrPool.getNumbers() ));       

    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Expf", fast_expf, expPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Logf", fast_logf, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Sinf", fast_sinf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Cosf", fast_cosf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Tanf", fast_tanf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Asinf", fast_asinf, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Acosf", fast_acosf, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Atanf", fast_atanf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Isqrtf", fast_isqrtf, positiverPool.getNumbers() ));       
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Invf", fast_invf, symmetricrPool.getNumbers() ));  
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Apr_Isqrtf", fast_approx_isqrtf, positiverPool.getNumbers() ));       
    fcn_tuples->push_back(genfpfcn_tuple<float>( "Fast_Apr_Invf", fast_approx_invf, symmetricrPool.getNumbers() )); 

#ifdef _VC_AVAILABLE_
	fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Identityf", vc_identityf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Logf", vc_logf, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Sinf", vc_sinf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Cosf", vc_cosf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Asinf", vc_asinf, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Atanf", vc_atanf, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Isqrtf", vc_isqrtf, positiverPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcn_tuple<float>( "VC_Inversef", vc_invf, symmetricrPool.getNumbers() ));  
#endif
}

/// Vector form
void getFunctionTuplesvect(std::vector<genfpfcnv_tuple<double>>* fcn_tuples,
						randomPool<double>& symmetricrPool,
						randomPool<double>& positiverPool,
						randomPool<double>& mone2onerPool,
						randomPool<double>& expPool){
	fcn_tuples->clear();

        fcn_tuples->push_back(genfpfcnv_tuple<double>( "Identityv", identityv, symmetricrPool.getNumbers() ));        
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Expv", vdt::expv, expPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Logv", vdt::logv, positiverPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Sinv", vdt::sinv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Cosv", vdt::cosv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Tanv", vdt::tanv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Asinv", vdt::asinv, mone2onerPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Acosv", vdt::acosv, mone2onerPool.getNumbers() )); 
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Atanv", vdt::atanv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Isqrtv", vdt::isqrtv, positiverPool.getNumbers() ));    
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Inversev", vdt::invv, symmetricrPool.getNumbers() ));    

	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Expv", fast_expv, expPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Logv", fast_logv, positiverPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Sinv", fast_sinv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Cosv", fast_cosv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Tanv", fast_tanv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Asinv", fast_asinv, mone2onerPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Acosv", fast_acosv, mone2onerPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Atanv", fast_atanv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Isqrtv", fast_isqrtv, positiverPool.getNumbers() ));        
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Invv", fast_invv, symmetricrPool.getNumbers() ));  
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Apr_Isqrtv", fast_approx_isqrtv, positiverPool.getNumbers() ));   
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "Fast_Apr_Invv", fast_approx_invv, symmetricrPool.getNumbers() ));

#ifdef _VC_AVAILABLE_
	fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Identityv", vc_identityv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Logv", vc_logv, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Sinv", vc_sinv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Cosv", vc_cosv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Asinv", vc_asinv, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Atanv", vc_atanv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Isqrtv", vc_isqrtv, positiverPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcnv_tuple<double>( "VC_Inversev", vc_invv, symmetricrPool.getNumbers() ));  
#endif
}

void getFunctionTuplesvect(std::vector<genfpfcnv_tuple<float>>* fcn_tuples,
						randomPool<float>& symmetricrPool,
						randomPool<float>& positiverPool,
						randomPool<float>& mone2onerPool,
						randomPool<float>& expPool){
	fcn_tuples->clear();
        
        fcn_tuples->push_back(genfpfcnv_tuple<float>( "Identityfv", identityfv, symmetricrPool.getNumbers() ));        
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Expfv", vdt::expfv, expPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Logfv", vdt::logfv, positiverPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Sinfv", vdt::sinfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Cosfv", vdt::cosfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Tanfv", vdt::tanfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Asinfv", vdt::asinfv, mone2onerPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Acosfv", vdt::acosfv, mone2onerPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Atanfv", vdt::atanfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Isqrtfv", vdt::isqrtfv, positiverPool.getNumbers() ));    
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Inversefv", vdt::invfv, symmetricrPool.getNumbers() ));    

	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Expfv", fast_expfv, expPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Logfv", fast_logfv, positiverPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Sinfv", fast_sinfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Cosfv", fast_cosfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Tanfv", fast_tanfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Asinfv", fast_asinfv, mone2onerPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Acosfv", fast_acosfv, mone2onerPool.getNumbers() )); 
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Atanfv", fast_atanfv, symmetricrPool.getNumbers() ));
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Isqrtfv", fast_isqrtfv, positiverPool.getNumbers() ));            
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Invfv", fast_invfv, symmetricrPool.getNumbers() ));
        fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Apr_Isqrtfv", fast_approx_isqrtfv, positiverPool.getNumbers() ));         
        fcn_tuples->push_back(genfpfcnv_tuple<float>( "Fast_Apr_Invfv", fast_approx_invfv, symmetricrPool.getNumbers() ));

#ifdef _VC_AVAILABLE_
	fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Identityfv", vc_identityfv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Logfv", vc_logfv, positiverPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Sinfv", vc_sinfv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Cosfv", vc_cosfv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Asinfv", vc_asinfv, mone2onerPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Atanfv", vc_atanfv, symmetricrPool.getNumbers() ));
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Isqrtfv", vc_isqrtfv, positiverPool.getNumbers() ));    
    fcn_tuples->push_back(genfpfcnv_tuple<float>( "VC_Inversefv", vc_invfv, symmetricrPool.getNumbers() ));  
#endif
}

/// Helper function retrieves basic function names like Cos, Sin, Exp... (no Fast_, no extensions)
void getFunctionBasicNames(std::vector<std::string>* names){

	// Prepare dummy tuple to retrieve function names 
	std::vector<genfpfcn_tuple<double>> tmpTuples;
	randomPool<double> dummy(0,0,1);
	getFunctionTuples(&tmpTuples, dummy,dummy,dummy,dummy);

	names->clear();

        // A better algo to do that
        std::string name;
        for(unsigned int i=0;i<tmpTuples.size();i++){
          name=std::get<0>(tmpTuples[i]);
          if (name.find("Fast")==std::string::npos &&
              name.find("Identity")==std::string::npos){
            names->push_back(name);  
          }
        }
        
	return;
}
#endif

