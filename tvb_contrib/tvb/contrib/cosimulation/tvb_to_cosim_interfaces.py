# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import HasTraits, List
from tvb.contrib.cosimulation.monitors import CosimStateMonitor, CosimHistoryMonitor, CosimCouplingMonitor


class TVBtoCosimInterfaces(HasTraits):

    state_interfaces = List(of=CosimStateMonitor)
    history_interfaces = List(of=CosimHistoryMonitor)
    coupling_interfaces = List(of=CosimCouplingMonitor)

