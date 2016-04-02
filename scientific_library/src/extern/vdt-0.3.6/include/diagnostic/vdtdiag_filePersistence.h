/* vdtdiag_filePersistence.h
*
*	created on 7.7.2012
*
*	Some useful manipulators for writting/reading
*	floating point values to/from stream in hex format
*
*	Author: Ladislav Horky
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

#ifndef _VDT_FILE_PERSISTENCE_
#define _VDT_FILE_PERSISTENCE_

#include<string>
#include<fstream>
#include<iostream>
#include<iomanip>
#include"vdtcore_common.h"

/// Useful manipulators

/// Usage: stream << fpToHex<T>(0.0) << ...;
template<class T>
class fpToHex{
	const T fpvalue;
public:
	/// Constructor just stores the value
	fpToHex(const T val):fpvalue(val){}
	~fpToHex(){}

	/// Overloaded insertion operator, prints "0xFFF..FFF " including last 'space'
	friend std::ostream &operator<<(std::ostream &stream, const fpToHex &obj){
		return stream << "0x"<< std::setbase(16) <<vdt::details::fp2uint(obj.fpvalue) << std::setbase(10) <<' ';
	}
};

/// Usage: fpFromHex<T> foo; stream >> foo; val = foo.getValue();
template<class T>
class fpFromHex{
	uint64_t value;
public:
	/// Nothing
	fpFromHex(){}
	~fpFromHex(){}

	/// Overloaded insertion operator, just reads int value
	friend std::istream &operator>>(std::istream &stream, fpFromHex &obj){
		stream >> std::setbase(16) >> obj.value >> std::setbase(10);
		return stream;
	}

	/// Performs conversion to fp
	T getValue(){
		if(sizeof(T) == 8) 
			return vdt::details::uint642dp(value);
		else
			return vdt::details::uint322sp((uint32_t)value);
	}
};

#endif
