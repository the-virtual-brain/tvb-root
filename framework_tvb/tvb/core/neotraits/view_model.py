from tvb.basic.neotraits.api import HasTraits, Attr


class ViewModel(HasTraits):
    """
    TODO: just inherit HT and override attrs or support automated way to generate VM from existent HT. Needed changes:
        - HT objects that are going to DTSF should be kept as GID on VM
        - Equations can be kept the same
        - support UI names for attrs with choices
    """


class UploadAttr(Attr):
    """
    # TODO: take also the extension type
    """


class DataTypeGidAttr(Attr):
    """
    # TODO: keep a link between datatype type and gid, plus take filters
    """


class ChoicesAttr(Attr):
    """
    # TODO: allow UI names for choices
    """


class EquationAttr(Attr):
    """
    # TODO: there are places where we need eq params as a nested form. Figure out a proper model
    """
