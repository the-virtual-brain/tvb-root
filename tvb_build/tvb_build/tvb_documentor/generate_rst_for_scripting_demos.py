# I'll admit, I'm just going to copy and paste the output of this into the Demos.rst
# but we should probably eventually maybe make it better.

import os
import os.path
import glob

here = os.path.abspath(os.path.dirname(__file__))
demo_folder = os.path.sep.join([here, '..', '..', '..', 'tvb_documentation', 'demos'])

nburl = 'http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos'

demos = []
for ipynb_fname in glob.glob(os.path.join(demo_folder, '*.ipynb')):
    _, fname = os.path.split(ipynb_fname)
    title = ' '.join([s.title() for s in fname.split('.')[0].split('_')])
    demos.append((fname, title))

# generate refs first
ref_fmt = '.. _{title}: {nburl}/{fname}'
for fname, title in demos:
    print ref_fmt.format(fname=fname, title=title, nburl=nburl)

# now figure directives
fig_fmt = '''.. figure:: figures/{bname}.png
      :width: 200px
      :figclass: demo-figure
      :target: `{title}`_

      `{title}`_

'''
for fname, title in demos:
    bname, _ = fname.split('.')
    print fig_fmt.format(bname=bname, title=title)
