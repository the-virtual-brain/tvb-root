/**
 * Checks for the difference between pairs of floating points numbers.
 **/

#include "vdtdiag_helper.h"
#include <iostream>

using namespace vdth;

int main(){

	print_different_bit(3.f,-3.f);
	print_different_bit(123.,123.00001);
	print_different_bit(4.f,4.f);

}
