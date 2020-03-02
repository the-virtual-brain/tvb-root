from mako.template import Template
from model.model import Model

def regTVB_templating(model_filename):
    """
    function will start generation of regular TVB models according to fp_xml
    modelfile.py is placed results into tvb/simulator/models
    for new models models/__init.py__ is auto_updated if model is unfamiliar to tvb
    file_class_name is the name of the producedfile and also the class name

    .. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>
    """

    fp_xml = 'NeuroML/XMLmodels/' + model_filename.lower() + '.xml'
    modelfile = "../simulator/models/" + model_filename.lower() + ".py"

    model = Model()
    model.import_from_file(fp_xml)

    modelist = list()
    modelist.append(model.component_types[model_filename])

    # do some inventory. check if boundaries are set for any sv to print the boundaries section in template
    svboundaries = 0
    for i, sv in enumerate(modelist[0].dynamics.state_variables):
        if sv.boundaries != 'None' and sv.boundaries != '' and sv.boundaries:
            svboundaries = 1
            continue

    # start templating
    template = Template(filename='tmpl8_regTVB.py')
    model_str = template.render(
                            dfunname=model_filename,
                            const=modelist[0].constants,
                            dynamics=modelist[0].dynamics,
                            svboundaries=svboundaries,
                            exposures=modelist[0].exposures
                            )
    # write template to file
    with open(modelfile, "w") as f:
        f.writelines(model_str)

    # write new model to init.py such it is familiar to TVB
    doprint=1
    with open("../simulator/models/__init__.py", "r+") as f:
        for line in f.readlines():
            if ("from ." + model_filename.lower() + " import " + model_filename) in line:
                doprint=0
        if doprint:
            f.writelines("\nfrom ." + model_filename.lower() + " import " + model_filename)
