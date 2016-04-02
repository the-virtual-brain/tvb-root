#! /usr/bin/python

"""
Generates vc wrapper - both header and .cc file
"""

RESTRICT="__restrict__"
VC_PREF="vc_"
VC_SCOPE_OP="Vc::"

FUNCTIONS_LIST=["asin",
                "atan",
                "cos",
                "inv",
                "log",
                "sin",
                "isqrt",
                "identity"]

VC_WRAPPER_HEADER='vdtdiag_vcWrapper.h'
VC_WRAPPER_IMPL='vdtdiag_vcWrapper.cc'
                
#------------------------------------------------------------------

def get_type_dependent_parts(is_double, is_vector):
  suffix=""
  type="double"
  if(is_double):
    if(is_vector):
      suffix="v"
  else:
    type="float"
    suffix="f"
    if(is_vector):
      suffix="fv" 
      
  data_type="%s" %(type)
  if(is_vector):
    data_type="%s* %s" %(type, RESTRICT)
  vc_data_type="%s%s_v" %(VC_SCOPE_OP,type)
  return (type, data_type, vc_data_type, suffix)
  
#------------------------------------------------------------------
  
def get_function_prototype_and_vctype(fcn_name,is_double,is_vector):
  (type,data_type,vc_type,suffix)=get_type_dependent_parts(is_double,is_vector)
  prototype="%s %s%s%s(%s x)" %(type,VC_PREF,fcn_name,suffix,data_type) 
  if(is_vector):
    prototype="void %s%s%s(const uint32_t size, %s iarray, %s oarray)" %(VC_PREF,fcn_name,suffix,data_type,data_type)
  return (prototype,vc_type)  

#-------------------------------------------------------------------

# translation of raw name withnout suffixes
def get_vc_fcnname_translation(fcn_name):
  if(fcn_name == "inv"):
    vc_name = "reciprocal"
  elif(fcn_name == "isqrt"):
    vc_name = "rsqrt"
  else:
    vc_name = fcn_name
  return vc_name
  
#-------------------------------------------------------------------

def get_function_code(fcn_name,vc_data_type,is_vector):
  if(is_vector):
    code = "{\n" +\
        "\t%s in,out;\n" %(vc_data_type) +\
        "\tconst uint32_t vcSize = %s::Size;\n" %(vc_data_type) +\
        "\tuint32_t step,i;\n"\
        "\tfor(step=0;step < size-(size%vcSize); step += vcSize){\n" +\
        "\t   in.load(iarray+step, Vc::Unaligned);\n"
    if(fcn_name == "identity"):
      code += "\t   in.store(oarray+step,Vc::Unaligned);\n"
    else:
      code += "\t   Vc::%s(in).store(oarray+step,Vc::Unaligned);\n" %(get_vc_fcnname_translation(fcn_name))
    code += "\t}\n"+\
        "\tuint32_t tail = size-step;\n" +\
        "\tfor(i=0;i<tail;i++)\n"  +\
        "\t   in[i]=iarray[step+i];\n"
    if(fcn_name == "identity"):
      code += "\tout = in;\n"
    else:
      code += "\tout = Vc::%s(in);\n" %(get_vc_fcnname_translation(fcn_name))
    code += "\tfor(i=0;i<tail;i++)\n" +\
        "\t   oarray[step+i]=out[i];\n}\n\n"
  else:
    code = "{\n" +\
      "\t%s in;\n" %(vc_data_type) +\
      "\tin[0] = x;\n"
    if(fcn_name == "identity"):
      code += "\t%s out = in;\n\treturn out[0];\n}\n\n" %(vc_data_type)
    else:
      code += "\treturn Vc::%s(in)[0];\n}\n\n" %(get_vc_fcnname_translation(fcn_name))
  
  return code
  
#---------------------------------------------------------------------
                
def create_vcWrapper_signature(fcn_name,is_double=False,is_vector=False,is_impl=False):
  (code,vc_type) = get_function_prototype_and_vctype(fcn_name,is_double,is_vector)
  if is_impl:
    code += get_function_code(fcn_name,vc_type,is_vector)
  else:
    code += ";\n"	  
  return code
		 
#------------------------------------------------------------------
		 
def create_vcWrapper_signatures(is_impl=False):
  code=""
  for is_vector in (False,True):
    for is_double in (True,False):
      for fcn_name in sorted(FUNCTIONS_LIST):
        code+=create_vcWrapper_signature(fcn_name,is_double,is_vector,is_impl)
  return code


#------------------------------------------------------------------   
   
def get_header_file():
  code= "// Automatically generated\n"\
        '#ifndef __VDT_VC_WRAPPER__\n'\
        '#define __VDT_VC_WRAPPER__\n'\
        '#include "inttypes.h"\n'+\
        '//#ifdef __cplusplus\n'\
        'extern "C"{\n'\
        '//#endif\n'+\
        create_vcWrapper_signatures(is_impl=False)+\
        '//#ifdef __cplusplus\n'\
        '}\n'\
        '//#endif\n'\
        '\n'\
        '#endif\n'
  return code

#------------------------------------------------------------------   
		   
def get_impl_file():
  code= "// Automatically generated\n"+\
        '#include "%s"\n' %VC_WRAPPER_HEADER+\
        '#include "Vc/Vc"\n'+\
        '//#ifdef __cplusplus\n'\
        'extern "C"{\n'\
        '//#endif\n'+\
        create_vcWrapper_signatures(is_impl=True)+\
        '//#ifdef __cplusplus\n'\
        '}\n'\
        '//#endif\n'
  return code
		  
#------------------------------------------------------------------

def create_header():
  ofile=file(VC_WRAPPER_HEADER,'w')
  ofile.write(get_header_file())
  ofile.close()
  
#------------------------------------------------------------------

def create_impl():
  ofile=file(VC_WRAPPER_IMPL,'w')
  ofile.write(get_impl_file())
  ofile.close()

#------------------------------------------------------------------
  
if __name__ == "__main__":
  create_header()
  create_impl()
