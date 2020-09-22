# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import HasTraits, List
from tvb.contrib.cosimulation.monitors import CosimStateMonitor, CosimHistoryMonitor, CosimCouplingMonitor


class TVBtoCosimInterfaces(HasTraits):

    """This class holds lists of
       - state_interfaces,
       - history_interfaces,
       - and coupling_interfaces,
       monitors"""

    state_interfaces = List(of=CosimStateMonitor)
    history_interfaces = List(of=CosimHistoryMonitor)
    coupling_interfaces = List(of=CosimCouplingMonitor)

