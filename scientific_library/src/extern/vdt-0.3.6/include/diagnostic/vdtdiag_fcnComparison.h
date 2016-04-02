/**
 * This file contains the classes to store and compare the 
 * arithmetical performance of the mathematical functions.
 * 
 * Author Danilo Piparo
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

#ifndef _VDT_COMPARISON_
#define _VDT_COMPARISON_


#include <limits>
#include "vdtdiag_helper.h"
#include "vdtdiag_filePersistence.h"
#include "vdtdiag_interfaces.h"


/**
 * Class that represents the comparison of the response of two mathematical 
 * functions. It is initialised with input values and outputs of the two 
 * functions or with an ascii file. It dumps on disk its status as ascii file 
 * as well. Methods to fetch Mean, RMS, min and MAX of the differences between 
 * functions outputs are provided.
 **/

template<typename T>
class fcnComparison1D:public IfcnComparison<T>{
	using vectT=std::vector<T>;
public:
//------------------------------------------------------------------------------
	fcnComparison1D(const std::string& name,
			const vectT& input,
            const vectT& out1,
            const vectT& out2):
         IfcnComparison<T>(name,out1,out2),
         m_input(input){};
//------------------------------------------------------------------------------
	fcnComparison1D(const std::string& input_filename):
		IfcnComparison<T>(input_filename){
		std::ifstream ifile ( input_filename );
			std::string line;
			//skip the 2 header lines but read the func name from resp
			for (uint16_t i=0;i<5;++i)
				std::getline(ifile,line);
			//read data from file
			//read stats:
			fpFromHex<double> mean, rms;
			ifile >> IfcnComparison<T>::m_max >> IfcnComparison<T>::m_min >> mean >> rms;
			IfcnComparison<T>::m_mean = mean.getValue();
			IfcnComparison<T>::m_RMS = rms.getValue();
			//read rest of file
			fpFromHex<T> in_val, out1_val, out2_val;
			uint16_t tmp_diff;
			T dummy;	//input value for Python, now useless
			while(ifile >> in_val >> out1_val >> out2_val >> tmp_diff >> dummy) {
				m_input.push_back(in_val.getValue());
				IfcnComparison<T>::m_out1.push_back(out1_val.getValue());
				IfcnComparison<T>::m_out2.push_back(out2_val.getValue());
				IfcnComparison<T>::m_diff_bitv.push_back(tmp_diff);
			}
	};
//------------------------------------------------------------------------------
	~fcnComparison1D(){};
//------------------------------------------------------------------------------

  /// Print to screen the information
  void print(){

	// Loop over all numbers
	uint32_t counter=0;
	const uint32_t size=m_input.size();
	std::cout << "Function Performance Comparison:\n";
	for (uint32_t i=0;i<size;++i){
	  //DECIMAL
	  const uint32_t width=std::numeric_limits<T>::digits10 +2;
	  // Patchwork, but it's ok to read!
	  const uint32_t dec_repr_w=width+7;
	  std::cout << std::setprecision(width);
	  std::cout <<  counter++ << "/" << size << " " << IfcnComparison<T>::m_name;
	  std::cout.setf(std::ios_base::scientific);
	  std::cout << "( " << std::setw(dec_repr_w) << m_input[i] << " ) = "
				<< std::setw(dec_repr_w) << IfcnComparison<T>::m_out1[i] << " "
				<< std::setw(dec_repr_w) << IfcnComparison<T>::m_out2[i]; //<< "\t"
	  std::cout.unsetf(std::ios_base::scientific);
	  std::cout.setf(std::ios_base::showbase);
	  std::cout << std::setbase(16)
				<< " "<< vdt::details::fp2uint(IfcnComparison<T>::m_out1[i])
				<< " " << vdt::details::fp2uint(IfcnComparison<T>::m_out2[i])
				<< " "<< std::setbase(10) << IfcnComparison<T>::m_diff_bitv[i] << std::endl;
	  std::cout.unsetf(std::ios_base::showbase);
	}
	// now the stats
	IfcnComparison<T>::printStats();
  }
  //------------------------------------------------------------------------------
  /// Dump on ascii file
	void writeFile(const std::string& output_filename){
		const std::string preamble("VDT function arithmetics performance comparison file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (IfcnComparison<T>::m_from_file){
			std::string line;
			std::ifstream ifile ( IfcnComparison<T>::m_ifile_name );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << IfcnComparison<T>::m_ifile_name << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Comparison specs/function name = " << IfcnComparison<T>::m_name << std::endl
				<< "Format: input out1 out2 diffbit (decimal)input\nFirst line are stats: Max Min 0xMean 0xRMS\n";
			// Do not write dec, but HEX!
			// First line are stats
			ofile << IfcnComparison<T>::m_max << " " << IfcnComparison<T>::m_min << " "
					<< fpToHex<double>(IfcnComparison<T>::m_mean)
					<< fpToHex<double>(IfcnComparison<T>::m_RMS) << std::endl;
			// Now the rest of file
			ofile.precision(std::numeric_limits<T>::digits10);
			for (uint32_t i=0;i<m_input.size();++i)
				ofile << fpToHex<T>(m_input[i]) << fpToHex<T>(IfcnComparison<T>::m_out1[i])
				<< fpToHex<T>(IfcnComparison<T>::m_out2[i]) << IfcnComparison<T>::m_diff_bitv[i] << " "
				<< std::fixed << m_input[i] <<std::endl;	//m_input[i] for python to easily read it
		}
	}

//------------------------------------------------------------------------------

private:
	vectT m_input;

};



template<typename T>
class fcnComparison2D:public IfcnComparison<T>{
	using vectT=std::vector<T>;
public:
//------------------------------------------------------------------------------
	fcnComparison2D(const std::string& name,
			const vectT& input1,
			const vectT& input2,
            const vectT& out1,
            const vectT& out2):
         IfcnComparison<T>(name,out1,out2),
         m_input1(input1),
         m_input2(input2){};
//------------------------------------------------------------------------------
	fcnComparison2D(const std::string& input_filename):
		IfcnComparison<T>(input_filename){
		std::ifstream ifile ( input_filename );
			std::string line;
			//skip the 2 header lines but read the func name from resp
			for (uint16_t i=0;i<5;++i)
				std::getline(ifile,line);
			//read data from file
			//read stats:
			fpFromHex<double> mean, rms;
			ifile >> IfcnComparison<T>::m_max >> IfcnComparison<T>::m_min >> mean >> rms;
			IfcnComparison<T>::m_mean = mean.getValue();
			IfcnComparison<T>::m_RMS = rms.getValue();
			//read rest of file
			fpFromHex<T> in_val1, in_val2, out1_val, out2_val;
			uint16_t tmp_diff;
			T dummy1, dummy2;	//input value for Python, now useless
			while(ifile >> in_val1 >> in_val2 >> out1_val >> out2_val >> tmp_diff >> dummy1 >> dummy2) {
				m_input1.push_back(in_val1.getValue());
				m_input2.push_back(in_val2.getValue());
				IfcnComparison<T>::m_out1.push_back(out1_val.getValue());
				IfcnComparison<T>::m_out2.push_back(out2_val.getValue());
				IfcnComparison<T>::m_diff_bitv.push_back(tmp_diff);
			}
	};
//------------------------------------------------------------------------------
	~fcnComparison2D(){};
//------------------------------------------------------------------------------

  /// Print to screen the information
  void print(){

	// Loop over all numbers
	uint32_t counter=0;
	const uint32_t size=m_input1.size();
	std::cout << "Function Performance Comparison:\n";
	for (uint32_t i=0;i<size;++i){
	  //DECIMAL
	  const uint32_t width=std::numeric_limits<T>::digits10 +2;
	  // Patchwork, but it's ok to read!
	  const uint32_t dec_repr_w=width+7;
	  std::cout << std::setprecision(width);
	  std::cout <<  counter++ << "/" << size << " " << IfcnComparison<T>::m_name;
	  std::cout.setf(std::ios_base::scientific);
	  std::cout << "( " << std::setw(dec_repr_w) << m_input1[i] << ", "<< m_input2[i]<< " ) = "
				<< std::setw(dec_repr_w) << IfcnComparison<T>::m_out1[i] << " "
				<< std::setw(dec_repr_w) << IfcnComparison<T>::m_out2[i]; //<< "\t"
	  std::cout.unsetf(std::ios_base::scientific);
	  std::cout.setf(std::ios_base::showbase);
	  std::cout << std::setbase(16)
				<< " "<< vdt::details::fp2uint(IfcnComparison<T>::m_out1[i])
				<< " " << vdt::details::fp2uint(IfcnComparison<T>::m_out2[i])
				<< " "<< std::setbase(10) << IfcnComparison<T>::m_diff_bitv[i] << std::endl;
	  std::cout.unsetf(std::ios_base::showbase);
	}
	// now the stats
	IfcnComparison<T>::printStats();
  }
  //------------------------------------------------------------------------------
  /// Dump on ascii file
	void writeFile(const std::string& output_filename){
		const std::string preamble("VDT function arithmetics performance comparison file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (IfcnComparison<T>::m_from_file){
			std::string line;
			std::ifstream ifile ( IfcnComparison<T>::m_ifile_name );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << IfcnComparison<T>::m_ifile_name << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Comparison specs/function name = " << IfcnComparison<T>::m_name << std::endl
				<< "Format: input out1 out2 diffbit (decimal)input\nFirst line are stats: Max Min 0xMean 0xRMS\n";
			// Do not write dec, but HEX!
			// First line are stats
			ofile << IfcnComparison<T>::m_max << " " << IfcnComparison<T>::m_min << " "
					<< fpToHex<double>(IfcnComparison<T>::m_mean)
					<< fpToHex<double>(IfcnComparison<T>::m_RMS) << std::endl;
			// Now the rest of file
			ofile.precision(std::numeric_limits<T>::digits10);
			for (uint32_t i=0;i<m_input1.size();++i)
				ofile << fpToHex<T>(m_input1[i]) << fpToHex<T>(m_input2[i])
				<< fpToHex<T>(IfcnComparison<T>::m_out1[i]) << fpToHex<T>(IfcnComparison<T>::m_out2[i])
				<< IfcnComparison<T>::m_diff_bitv[i] << " "
				<< std::fixed << m_input1[i] << " " << m_input2[i] <<std::endl;	//inputs for python to easily read it
		}
	}

//------------------------------------------------------------------------------

private:
	vectT m_input1;
	vectT m_input2;
};



template<class T>
class fcnComparison_old{
  using vectT=std::vector<T>;
public:

  /// Ctor from input, output1 and output2.
  fcnComparison_old(const std::string& name,
                const vectT& input, 
                const vectT& out1,
                const vectT& out2):
                m_from_file(false),
    			m_ifile_name("From scratch"),
    			m_name(name),
               	m_input(input),
                m_out1(out1),
                m_out2(out2){
	
    // A basic consistency check
    assert(input.size()==out1.size());
    assert(input.size()==out2.size());
                  
    // Calculate the differences in bit
    m_fillBitDiffs();
    
    // Calculate the stats
    m_calcStats();    
	}

	/// Construct from a file
  fcnComparison_old(const std::string& input_filename):
		m_from_file(true),
		m_ifile_name(input_filename),
		m_name(std::string("From ")+input_filename){
		
			std::ifstream ifile ( input_filename );
			std::string line;
			//skip the 5 header lines but read the func name from resp		
			for (uint16_t i=0;i<2;++i)
				std::getline(ifile,line);
			//read data from file	
			//read stats:
			fpFromHex<double> mean, rms;
			ifile >> m_max >> m_min >> mean >> rms;
			m_mean = mean.getValue();
			m_RMS = rms.getValue();
			//read rest of file
			fpFromHex<T> in_val, out1_val, out2_val;
			uint16_t tmp_diff;
			T trash;	//input value for Python, now useless
			while(ifile >> in_val >> out1_val >> out2_val >> tmp_diff >> trash) {
				m_input.push_back(in_val.getValue());
				m_out1.push_back(out1_val.getValue());
				m_out2.push_back(out2_val.getValue());
				m_diff_bitv.push_back(tmp_diff);
			}

			// The same stuff as before
			assert(m_input.size()==m_out1.size());
			assert(m_input.size()==m_out2.size());
			//m_fillBitDiffs();
			//m_calcStats();  
	}

//------------------------------------------------------------------------------

  /// Nothing to do here
  ~fcnComparison_old(){};

//------------------------------------------------------------------------------    
    
  /// Print to screen the information
  void print(){
    
    // Loop over all numbers
    uint32_t counter=0;
    const uint32_t size=m_input.size();
    std::cout << "Function Performance Comparison:\n";
    for (uint32_t i=0;i<size;++i){
      //DECIMAL
      const uint32_t width=std::numeric_limits<T>::digits10 +2;
      // Patchwork, but it's ok to read!
      const uint32_t dec_repr_w=width+7;
      std::cout << std::setprecision(width);
      std::cout <<  counter++ << "/" << size << " " << m_name;
      std::cout.setf(std::ios_base::scientific);
      std::cout << "( " << std::setw(dec_repr_w) << m_input[i] << " ) = "                         
                << std::setw(dec_repr_w) << m_out1[i] << " "
                << std::setw(dec_repr_w) << m_out2[i]; //<< "\t"
      std::cout.unsetf(std::ios_base::scientific);
      std::cout.setf(std::ios_base::showbase);
      std::cout << std::setbase(16)
                << " "<< vdt::details::fp2uint(m_out1[i])
                << " " << vdt::details::fp2uint(m_out2[i])
                << " "<< std::setbase(10) << m_diff_bitv[i] << std::endl;
      std::cout.unsetf(std::ios_base::showbase);
    }
    // now the stats
    printStats();
  }

  //------------------------------------------------------------------------------  

  void printStats(const bool tabular = false){
	if(!tabular){
		std::cout << "Stats for " << m_name << ":\n"
              << std::setprecision(2)
              << "Max diff bit: " << m_max << "\n"
              << "Min diff bit: " << m_min << "\n"
              << "Mean diff bit: " << m_mean << "\n"
              << "RMS diff bit: " << m_RMS << "\n"; 
	}else{
		std::cout << std::setw(30) << std::left << m_name
			<< std::setw(5) << std::right << m_max
			<< std::setw(5) << std::right << m_min
			<< std::setiosflags(std::ios::fixed)
			<< std::setw(7) << std::right << std::setprecision(2) << m_mean
			<< std::setw(7) << std::right << std::setprecision(2) << m_RMS << "\n";
	}
  };

  /// Dump on ascii file
	void writeFile(const std::string& output_filename){
		const std::string preamble("VDT function arithmetics performance comparison file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (m_from_file){
			std::string line;
			std::ifstream ifile ( m_ifile_name );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << m_ifile_name << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Comparison specs/function name = " << m_name << std::endl
				<< "Format: input out1 out2 diffbit (decimal)input\nFirst line are stats: Max Min 0xMean 0xRMS\n";
			// Do not write dec, but HEX!
			// First line are stats
			ofile << m_max << " " << m_min << " " << fpToHex<double>(m_mean) << fpToHex<double>(m_RMS) << std::endl;
			// Now the rest of file
			ofile.precision(std::numeric_limits<T>::digits10);
			for (uint32_t i=0;i<m_input.size();++i)
				ofile << fpToHex<T>(m_input[i]) << fpToHex<T>(m_out1[i]) << fpToHex<T>(m_out2[i]) << m_diff_bitv[i] << " "
					<< std::fixed << m_input[i] <<std::endl;	//m_input[i] for python to easily read it
		}
	}

//-----------------------------------------------------

	// Handy functions:
	bool hasDifference(){return (m_max > 0);}

private:

  const bool m_from_file;
  const std::string m_ifile_name;
  const std::string m_name;
  vectT m_input;
  vectT m_out1;
  vectT m_out2;
  std::vector<uint16_t> m_diff_bitv;
  double m_mean = 0;
  double m_RMS = 0;
  uint16_t m_min=255;
  uint16_t m_max=0;
  
  /// Fill the vector of different bits
  void m_fillBitDiffs(){
      const uint32_t size=m_input.size();
      m_diff_bitv.reserve(size);
      for (uint32_t i=0;i<size;++i)
        m_diff_bitv.push_back(vdth::diffbit(m_out1[i],m_out2[i]));
  }
  
  /// Calculate min,max,mean and RMS
  void m_calcStats(){
    double sum_x=0.;
    double sum_x2=0.;
    // Loop on the vectors and caculate the 2 momenta
    for (auto& bitdiff:m_diff_bitv){
      // momenta
      sum_x+=bitdiff;
      sum_x2+=bitdiff*bitdiff;
      // min and max
      if (bitdiff<m_min)m_min=bitdiff;
      if (bitdiff>m_max)m_max=bitdiff;
    }
    // Now the mean!
    const uint32_t size=m_input.size();
    m_mean=sum_x/size;
    if (size==1)
      m_RMS=-1;
    else
      m_RMS=(sum_x2 - size*m_mean*m_mean)/(size-1);
    
  }
  
};


// For compatibility
template <typename T>
using fcnComparison = fcnComparison1D<T>;


#endif
