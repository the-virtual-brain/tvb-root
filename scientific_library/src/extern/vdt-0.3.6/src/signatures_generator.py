#! /usr/bin/env python

"""
Generate the .cc file with the signatures
of the vector functions and if requested 
the ones for the preload
"""

RESTRICT="__restrict__"

LIBM_FUNCTIONS_LIST=["asin",
                "atan",
                "atan2",
                "acos",
                "cos",
                "exp",
                "log",
                "sin",
                "tan"]

FUNCTIONS_LIST=LIBM_FUNCTIONS_LIST+\
               ["isqrt",
                "inv",
                "identity",
                "identity2D",
                "fast_asin",
                "fast_acos",
                "fast_atan",
                "fast_atan2",
                "fast_cos",
                "fast_exp",
                "fast_inv",
                "fast_approx_inv",
                "fast_log",
                "fast_sin",
                "fast_isqrt",
                "fast_approx_isqrt",
                "fast_tan"]

VDT_VECTOR_HEADER='vdtMath.h'
VDT_VECTOR_IMPL='vdtMath_signatures.cc'

#------------------------------------------------------------------

def create_preload_signatures():
  code="// Automatically generated signatures for preload\n\n"
  for fpSuffix,fpType in (("","double"),("f","float"),("","float")):
    for function in LIBM_FUNCTIONS_LIST:
      libmFunction="%s%s" %(function,fpSuffix)
      vdtFunction = "vdt::fast_%s" %(libmFunction)
      if fpSuffix=="" and fpType=="float":
        vdtFunction+="f"
      if function=="atan2":
        code += "%s %s(%s x, %s y){return %s(x,y);};\n"%(fpType,libmFunction,fpType,fpType,vdtFunction)
      else:
        code += "%s %s(%s x){return %s(x);};\n"%(fpType,libmFunction,fpType,vdtFunction)
  return  code

#------------------------------------------------------------------

def create_vector_signature(fcn_name,is_double=False,is_impl=False):
  # For single precision
  suffix="fv"
  float_suffix="f"
  type="float"
  if is_double:
    suffix="v"
    type="double"
    float_suffix=""
  prefix=""
  vfcn_name="%s%s" %(fcn_name,suffix)
  in_data_type="%s const * %s" %(type, RESTRICT)
  out_data_type="%s* %s" %(type, RESTRICT)
  new_fcn_name="%s%s" %(prefix,fcn_name)
  code =  "void %s%s(const uint32_t size, %s iarray, %s oarray)" %(new_fcn_name,suffix,in_data_type,out_data_type)

  # Special case
  if "atan2" in fcn_name or "identity2D" in fcn_name:
      code =  "void %s%s(const uint32_t size, %s iarray1, %s iarray2, %s oarray)" %(new_fcn_name,suffix,in_data_type,in_data_type,out_data_type)

  if is_impl:
    impl_code = "{\n"+\
          "  for (uint32_t i=0;i<size;++i)\n"+\
		  "    oarray[i]=%s%s(iarray[i]);\n" %(new_fcn_name,float_suffix)+\
          "}\n\n"
    if "atan2" in fcn_name or "identity2D" in fcn_name:
      impl_code = "{\n"+\
          "  for (uint32_t i=0;i<size;++i)\n"+\
          "    oarray[i]=%s%s(iarray1[i],iarray2[i]);\n" %(new_fcn_name,float_suffix)+\
          "}\n\n"   
    code+=impl_code
  else:
	code += ";\n"	  
  return code
		 
#------------------------------------------------------------------
		 
def create_vector_signatures(is_impl=False):
  code="namespace vdt{\n"
  for is_double in (True,False):
    for fcn_name in sorted(FUNCTIONS_LIST):
      code+=create_vector_signature(fcn_name,is_double,is_impl)
  code += "} // end of vdt namespace"
  return code

#------------------------------------------------------------------   
		   
def get_impl_file():
  code= "// Automatically generated\n"+\
        '#include "%s"\n' %VDT_VECTOR_HEADER+\
        create_vector_signatures(is_impl=True)+\
        "\n"# the final newline

  return code

#------------------------------------------------------------------

def create_impl(preloadSignatures):
  ofile=file(VDT_VECTOR_IMPL,'w')
  ofile.write(get_impl_file())
  if preloadSignatures:
    ofile.write(create_preload_signatures())
  ofile.close()

#------------------------------------------------------------------

from optparse import OptionParser

if __name__ == "__main__":
  parser = OptionParser(usage="usage: %prog [options]")
  parser.add_option("-p",
                    action="store_true",
                    dest="preload_flag",
                    default=False,
                    help="Create symbols for the preload")
  (options, args) = parser.parse_args()
  create_impl(preloadSignatures=options.preload_flag)

