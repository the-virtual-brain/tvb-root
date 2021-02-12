
import os

from mako.template import Template
from mako.lookup import TemplateLookup
from mako.exceptions import text_error_template


here = os.path.dirname(os.path.abspath(__file__))


class MakoUtilMix:

    @property
    def lookup(self):
        lookup = TemplateLookup(directories=[here])
        return lookup

    def render_template(self, source, content):
        template = Template(source, lookup=self.lookup, strict_undefined=True)
        try:
            source = template.render(**content)
        except Exception as exc:
            print(text_error_template().render())
            raise exc
        return source

    def insert_line_numbers(self, source):
        lines = source.split('\n')
        numbers = range(1, len(lines) + 1)
        nu_lines = ['%03d\t%s' % (nu, li) for (nu, li) in zip(numbers, lines)]
        nu_source = '\n'.join(nu_lines)
        return nu_source
