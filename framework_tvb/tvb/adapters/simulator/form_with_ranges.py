from abc import abstractmethod
from tvb.basic.neotraits.api import NArray
from tvb.adapters.simulator.range_parameter import RangeParameter
from tvb.core.neotraits.forms import Form


class FormWithRanges(Form):
    @staticmethod
    @abstractmethod
    def get_params_configurable_in_phase_plane():
        """Return a list with all parameter names that can be configured in Phase Plane viewer for the current model"""

    def _get_parameters_for_pse(self):
        pass

    def _gather_parameters_with_range_defined(self):
        parameters_with_range = []
        for trait_field in self.trait_fields:
            if isinstance(trait_field.trait_attribute, NArray) and trait_field.trait_attribute.domain:
                parameters_with_range.append(trait_field.trait_attribute)
        return parameters_with_range

    def get_range_parameters(self):
        parameters_for_pse = self._gather_parameters_with_range_defined()

        result = []
        for range_param in parameters_for_pse:
            range_obj = RangeParameter(range_param.field_name, float, range_param.domain,
                                       isinstance(range_param, NArray))
            result.append(range_obj)
        return result
