# !/usr/bin/env python

'''
Trivial script to build the commandline to check the accuracy of the 
VDT functions.
'''

response_filename_template="%s__%s%s__response.txt"

functions=[\
"Acos",
"Acosv",
"Asin",
"Asinv",
"Atan",
"Atanv",
"Atan2",
"Atan2v",
"Cos",
"Cosv",
"Exp",
"Expv",
"Isqrt",
"Isqrtv",
"Log",
"Logv",
"Sin",
"Sinv",
"Tan",
"Tanv",
"Acosf",
"Acosfv",
"Asinf",
"Asinfv",
"Atanf",
"Atanfv",
"Atan2f",
"Atan2fv",
"Cosf",
"Cosfv",
"Expf",
"Expfv",
"Isqrtf",
"Isqrtfv",
"Logf",
"Logfv",
"Sinf",
"Sinfv",
"Tanf",
"Tanfv"]




def get_refs(nick,fast=""):
    if fast!="":
        fast+="_"
    refstring=""
    for function in functions:
        refstring+="%s," %response_filename_template%(nick,fast,function)
    return refstring[:-1]

def get_tests(nick):
    return get_refs(nick,"Fast")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
      print "Usage is checkAccuracy.py nick"
      sys.exit(1)
    nick=sys.argv[1]
    tests=get_tests(nick)
    refs=get_refs(nick)
    command='vdtArithmComparison -n=%s -T="%s" -R="%s"' %(nick,tests,refs)
    print command
        
        
        

