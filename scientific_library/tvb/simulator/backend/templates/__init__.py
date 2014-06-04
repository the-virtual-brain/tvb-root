"Templates & code generation"

import os
import glob
import logging

log = logging.getLogger(__name__)

# expect templates here, eventually elsewhere
here = os.path.dirname(os.path.abspath(__file__))
log.debug('looking in %r for templates', here)

# eventually sep C, CL, etc.
filetypes = ['cu']
sources = {}
log.debug('template filetypes %r', filetypes)

for ft in filetypes:
    patt = os.path.join(here, '*.' + ft)
    files = glob.glob(patt)
    log.debug('globbing for %r found %d files: %r', patt, len(files),
                [os.path.basename(f) for f in files])
    for f in files:
        try:
            with open(f, 'r') as fd:
                sources[os.path.basename(f)] = fd.read()
            log.debug('read %r', f)
        except Exception as exc:
            log.exception(exc)
            log.debug('failed to read %r', f)
