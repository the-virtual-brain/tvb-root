import uuid
from tvb.basic.neotraits.api import HasTraits, Attr


class ViewModel(HasTraits):
    """
    TODO: just inherit HT and override attrs or support automated way to generate VM from existent HT. Needed changes:
        - HT objects that are going to DTSF should be kept as GID on VM
        - Equations can be kept the same
        - support UI names for attrs with choices
    """


class Str(Attr):
    def __init__(self, field_type=str, default=None, doc='', label='', required=True,
                 final=False, choices=None):
        super(Str, self).__init__(field_type, default, doc, label, required, final, choices)


class DataTypeGidAttr(Attr):
    """
    Keep a GID but also link the type of DataType it should point to
    """

    def __init__(self, linked_datatype, field_type=uuid.UUID, filters=None, default=None, doc='', label='', required=True,
                 final=False, choices=None):
        super(DataTypeGidAttr, self).__init__(field_type, default, doc, label, required, final, choices)
        self.linked_datatype = linked_datatype
        self.filters = filters


class EquationAttr(Attr):
    """
    # TODO: there are places where we need eq params as a nested form. Figure out a proper model
    """
