# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

import os
import numpy
import cherrypy
import tvb.interfaces
import tvb.core.neotraits.forms
from jinja2 import PackageLoader, Environment
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Int, Dim, List
from tvb.core.neotraits.forms import StrField, IntField, BoolField, Form, FormField, ArrayField

# inject jinja config
# to ensure sanity do this once at a top level in the app

tvb.core.neotraits.forms.jinja_env = jinja_env = Environment(
    loader=PackageLoader('tvb.interfaces.web.templates', 'form_fields'),
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


class BaBaze(HasTraits):
    s = Attr(str, label='the s', doc='the mighty string')
    sign = Int(label='sign', choices=(-1, 0, 1), default=0)


class Bar(BaBaze):
    airplane_meal = Attr(str, choices=('cheese', 'chicken'), required=False)
    portions = Int(default=1, label='portions')
    is_fancy = Attr(bool, default=True)


class Baz(BaBaze):
    airplane_sweets = List(of=str, choices=('nuts', 'chocolate', 'prunes'))


class BarAndBaz(HasTraits):
    bar = Attr(field_type=Bar)
    baz = Attr(field_type=Baz)
    array = NArray(dtype=int, shape=(Dim.any, Dim.any), default=numpy.arange(6).reshape((2, 3)))


class Tree(HasTraits):
    name = Attr(str)


class Foo(HasTraits):
    # BaBaze.get_known_subclasses() -> forms
    air = Attr(field_type=BaBaze)  # choice of configs
    # TreeIndex.query.all()
    tree = Attr(field_type=Tree)  # choice of instances


class BaBazeForm(Form):
    def __init__(self):
        super(BaBazeForm, self).__init__()
        # these beg for metaprogramming
        self.s = StrField(BaBaze.s)
        self.sign = IntField(BaBaze.sign)


class BarForm(BaBazeForm):
    def __init__(self):
        super(BarForm, self).__init__()
        self.airplane_meal = StrField(Bar.airplane_meal)
        self.portions = IntField(Bar.portions)
        self.is_fancy = BoolField(Bar.is_fancy)


class BazForm(BaBazeForm):
    def __init__(self):
        super(BazForm, self).__init__()
        self.airplane_sweets = ArrayField(Baz.airplane_sweets)


class BarAndBazForm(Form):
    def __init__(self):
        super(BarAndBazForm, self).__init__()
        self.bar = FormField(BarForm, 'bar', label='bar')  # BarAndBaz.bar
        self.baz = FormField(BazForm, 'baz', label='baaz')
        # not from trait
        self.happy = BoolField(Attr(bool, label='clap'), 'clasp')
        self.array = ArrayField(BarAndBaz.array)

    def fill_from_trait(self, trait):
        super(BarAndBazForm, self).fill_from_trait(trait)
        self.bar.form.fill_from_trait(trait.bar)
        self.baz.form.fill_from_trait(trait.baz)

    def fill_trait(self, trait):
        super(BarAndBazForm, self).fill_trait(trait)
        trait.bar = Bar()
        trait.baz = Baz()
        self.bar.form.fill_trait(trait.bar)
        self.baz.form.fill_trait(trait.baz)


class View(object):

    # mock a storage
    def __init__(self):
        bar = Bar(airplane_meal='cheese', sign=-1, s='opaline')
        baz = Baz(airplane_sweets=('nuts', 'chocolate'), s='noting')
        self.trait = BarAndBaz(bar=bar, baz=baz)

    def _get_trait_instance(self):
        return self.trait

    def _set_trait_instance(self, barbaz):
        self.trait = barbaz

    @cherrypy.expose
    def index(self, **kwargs):
        output = ''

        if cherrypy.request.method == 'GET':
            barbaz = self._get_trait_instance()
            # get defaults from traits
            form = BarAndBazForm()
            form.fill_from_trait(barbaz)
        else:
            form = BarAndBazForm()
            form.fill_from_post(kwargs)
            if form.validate():
                barbaz = BarAndBaz()
                try:
                    form.fill_trait(barbaz)
                    self._set_trait_instance(barbaz)
                except Exception as ex:
                    form.errors.append(ex.message)

        barbaz = self._get_trait_instance()
        output = '\n'.join([str(barbaz), str(barbaz.bar), str(barbaz.baz)])
        return jinja_env.get_template('test1.html').render(adapter_form=form, output=output)


def main():
    static_root_dir = os.path.dirname(os.path.abspath(tvb.interfaces.__file__))
    cherrypy.config.update({
        'global': {
            'engine.autoreload.on' : False
        }
    })
    app_config = {
        '/static': {
            'tools.staticdir.root': static_root_dir,
            'tools.staticdir.on': True,
            'tools.staticdir.dir': os.path.join('web', 'static'),
        }
    }

    cherrypy.tree.mount( View(), '/', app_config)
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == '__main__':
    main()
