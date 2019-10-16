from tvb.simulator.coupling import Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, \
    Difference, Kuramoto

from tvb.core.neotraits._forms import Form, ArrayField, ScalarField


def get_coupling_to_form_dict():
    coupling_class_to_form = {
        Linear: LinearCouplingForm,
        Scaling: ScalingCouplingForm,
        HyperbolicTangent: HyperbolicTangentCouplingForm,
        Sigmoidal: SigmoidalCouplingForm,
        SigmoidalJansenRit: SigmoidalJansenRitForm,
        PreSigmoidal: PreSigmoidalCouplingForm,
        Difference: DifferenceCouplingForm,
        Kuramoto: KuramotoCouplingForm
    }
    return coupling_class_to_form


def get_ui_name_to_coupling_dict():
    ui_name_to_coupling = {}
    for coupling_class in get_coupling_to_form_dict().keys():
        ui_name_to_coupling.update({coupling_class.__name__: coupling_class})

    return ui_name_to_coupling


def get_form_for_coupling(coupling_class):
    return get_coupling_to_form_dict().get(coupling_class)


class LinearCouplingForm(Form):

    def __init__(self, prefix=''):
        super(LinearCouplingForm, self).__init__(prefix)
        self.a = ArrayField(Linear.a, self)
        self.b = ArrayField(Linear.b, self)


class ScalingCouplingForm(Form):

    def __init__(self, prefix=''):
        super(ScalingCouplingForm, self).__init__(prefix)
        self.a = ArrayField(Scaling.a, self)


class HyperbolicTangentCouplingForm(Form):

    def __init__(self, prefix=''):
        super(HyperbolicTangentCouplingForm, self).__init__(prefix)
        self.a = ArrayField(HyperbolicTangent.a, self)
        self.b = ArrayField(HyperbolicTangent.b, self)
        self.midpoint = ArrayField(HyperbolicTangent.midpoint, self)
        self.sigma = ArrayField(HyperbolicTangent.sigma, self)


class SigmoidalCouplingForm(Form):

    def __init__(self, prefix=''):
        super(SigmoidalCouplingForm, self).__init__(prefix)
        self.cmin = ArrayField(Sigmoidal.cmin, self)
        self.cmax = ArrayField(Sigmoidal.cmax, self)
        self.midpoint = ArrayField(Sigmoidal.midpoint, self)
        self.a = ArrayField(Sigmoidal.a, self)
        self.sigma = ArrayField(Sigmoidal.sigma, self)


class SigmoidalJansenRitForm(Form):

    def __init__(self, prefix=''):
        super(SigmoidalJansenRitForm, self).__init__(prefix)
        self.cmin = ArrayField(SigmoidalJansenRit.cmin, self)
        self.cmax = ArrayField(SigmoidalJansenRit.cmax, self)
        self.midpoint = ArrayField(SigmoidalJansenRit.midpoint, self)
        self.r = ArrayField(SigmoidalJansenRit.r, self)
        self.a = ArrayField(SigmoidalJansenRit.a, self)


class PreSigmoidalCouplingForm(Form):

    def __init__(self, prefix=''):
        super(PreSigmoidalCouplingForm, self).__init__(prefix)
        self.H = ArrayField(PreSigmoidal.H, self)
        self.Q = ArrayField(PreSigmoidal.Q, self)
        self.G = ArrayField(PreSigmoidal.G, self)
        self.P = ArrayField(PreSigmoidal.P, self)
        self.theta = ArrayField(PreSigmoidal.theta, self)
        self.dynamic = ScalarField(PreSigmoidal.dynamic, self)
        self.globalT= ScalarField(PreSigmoidal.globalT, self)


class DifferenceCouplingForm(Form):

    def __init__(self, prefix=''):
        super(DifferenceCouplingForm, self).__init__(prefix)
        self.a = ArrayField(Difference.a, self)


class KuramotoCouplingForm(Form):

    def __init__(self, prefix=''):
        super(KuramotoCouplingForm, self).__init__(prefix)
        self.a = ArrayField(Kuramoto.a, self)