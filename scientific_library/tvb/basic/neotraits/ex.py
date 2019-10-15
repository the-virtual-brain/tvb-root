class TraitError(Exception):
    def __init__(self, msg, trait=None, attr=None):
        self.trait = trait
        self.attr = attr
        super(TraitError, self).__init__(msg)


    def __str__(self):
        lines = [self.args[0]]
        if self.attr:
            lines.append('  attribute {}'.format(self.attr))
        if self.trait:
            lines.append('  class {}'.format(type(self.trait).__name__))

        return '\n'.join(lines)



class TraitAttributeError(TraitError, AttributeError):
    pass


class TraitValueError(TraitError, ValueError):
    pass


class TraitTypeError(TraitError, TypeError):
    pass

