
.. moduleauthor:: Stuart Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>


=============================
Suggested Coding Conventions.
=============================
What follow are some suggestions for the coding conventions we should adopt
within the Python code of "The Virtual Brain", hereafter TVB.

When in doubt implementing algorithms from the literature, please follow
guidelines described at:
http://codecapsule.com/2012/01/18/how-to-implement-a-paper/

Coding Style
------------
The standard reference for Python coding style is the BDFL's PEP 8:

    http://www.python.org/dev/peps/pep-0008/
    

To ease and simplify the process of attaining a consistent style across the
code-base I would suggest we agree upon an automated code checker such as
PyLint:

    http://pypi.python.org/pypi/pylint/
 

and require as a bare minimum an evaluation of > 7/10. By default, PyLint
essentially checks for consistency in line with the PEP 8 specification.
However, if there are strong objections to any of the requirements enforced by
PyLint, we can always adapt the configuration file to more adequately reflect
TVB requirements.

Floating point numbers should always have at least one number before and 
after the dot. Regexp please ?


DocString Style
---------------
The equivalent of PEP 8 for DocStrings is PEP 257:
    http://www.python.org/dev/peps/pep-0257/


Additional Conventions
----------------------
Zen:
    http://www.python.org/dev/peps/pep-0020/
    
Classes self description:
    All classes should have appropriately defined __repr__ and __str__ methods.

Module specific markup:
    http://docs.python.org/documenting/markup.html
    particularly ``..moduleauthor:``, perhaps in the module itself rather than 
    the associated .rst file in docs, to identify those to be held responsible.

Logging!:
    We use module level logging, based on the logging mechanism from gemenos,
    with a fallback to direct use of the python logging package. This means all 
    modules have
    
    ::

        from tvb.simulator.common import get_logger
        LOG = get_logger
       
    then you can call ``LOG.error(msg)``, ``LOG.warn(msg)``, ``LOG.info(msg)``, 
    etc. as is appropriate. The principle benefit over other ways of keeping 
    track of things (printf debugging,  the warning module) is flexibility. 
    Then to distinguish between classes within modules use messages of the form
    
    :: 
    
        LOG.debug("%s:a useful message a=%s"%(str(self), self.an_attribute))

    which makes use of the "Classes self description", mentioned above.

First three lines of module files:
    ::
    
        # -*- coding: utf-8 -*-
        
        """
    
    as the construct ``#!usr/bin/env python`` is not cross platform.

Never import \*:
    That is don't use ``from module import *``, as this adds needless
    difficulty to tracing the source of functions, methods, etc. 
    
Avoid ``from module import Class, Function, Etc`` where possible:
    Only really an issue when circular import of modules occurs. See, 
        http://docs.python.org/faq/programming.html#what-are-the-best-practices-for-using-import-in-a-module
    
    for more details.

Always use NumPy:
    We assume for the moment that most datatypes are NumPy arrays unless 
    is necessary to have a class, list, tuple etc.  

Line continuation:
    Where necessary add parentheses to enable implicit line continuation rather
    than explicit  continuation using \\, as the latter form opens the
    possibility for subtle bugs to be introduced.


Some terminoloy
---------------
Some useful python terminoloy can be found here:
    http://wiki.python.org/moin/Distutils/Terminology


Use a file/module template
--------------------------
 As a means of facilitating convergence on these conventions a template python
 file should be created for TVB project and be used as a starting point for any
 TVB Python modules.


 As a rough first example see, template_tvb.py

