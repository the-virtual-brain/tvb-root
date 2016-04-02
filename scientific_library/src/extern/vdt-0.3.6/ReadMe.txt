VDT is a library of mathematical functions, implemented in 
double and single precision. The implementation is fast and 
with the aid of modern compilers (e.g. gcc 4.7) autovectorisable.
For more information visit:
https://svnweb.cern.ch/trac/vdt
 
The basic idea is to exploit Pade polynomials. A lot of ideas were inspired by 
the cephes math library (by Stephen L. Moshier, moshier@na-net.ornl.gov) as 
well as portions of actual code. The Cephes library can be found here: 
http://www.netlib.org/cephes

Implemented functions
 - log
 - exp
 - sin
 - cos
 - tan
 - asin
 - acos
 - atan
 - inverse sqrt
 - inverse (faster than division, based on isqrt)


To compile it:
cmake -DCMAKE_INSTALL_PREFIX=$THEINSTALLDIR  .
make
make install

If you would like to compile the executables necessary for the diagnostics
(cpu and arithmetic performance measurements) type
cmake -D DIAG=1 .
make

The executables will be put in the /bin directory.

Copyright Danilo Piparo, Vincenzo Innocente, Thomas Hauth (CERN) 2012

VDT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser Public License for more details.

You should have received a copy of the GNU Lesser Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

