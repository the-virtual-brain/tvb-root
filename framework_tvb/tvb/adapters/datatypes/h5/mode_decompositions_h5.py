import numpy
from tvb.datatypes.mode_decompositions import PrincipalComponents, IndependentComponents

from tvb.core.neotraits.h5 import H5File, Reference, DataSet, Scalar


class PrincipalComponentsH5(H5File):
    def __init__(self, path):
        super(PrincipalComponentsH5, self).__init__(path)
        self.source = Reference(PrincipalComponents.source, self)
        self.weights = DataSet(PrincipalComponents.weights, self, expand_dimension=2)
        self.fractions = DataSet(PrincipalComponents.fractions, self, expand_dimension=1)
        self.norm_source = DataSet(PrincipalComponents.norm_source, self, expand_dimension=1)
        self.component_time_series = DataSet(PrincipalComponents.component_time_series,
                                             self, expand_dimension=1)
        self.normalised_component_time_series = DataSet(PrincipalComponents.normalised_component_time_series,
                                                        self, expand_dimension=1)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.weights.append(partial_result.weights, close_file=False)

        self.fractions.append(partial_result.fractions, close_file=False)

        partial_result.compute_norm_source()
        self.norm_source.append(partial_result.norm_source, close_file=False)

        partial_result.compute_component_time_series()
        self.component_time_series.append(partial_result.component_time_series, close_file=False)

        partial_result.compute_normalised_component_time_series()
        self.normalised_component_time_series.append(partial_result.normalised_component_time_series)

    def read_fractions_data(self, from_comp, to_comp):
        """
        Return a list with fractions for components in interval from_comp, to_comp and in
        addition have in position n the sum of the fractions for the rest of the components.
        """
        from_comp = int(from_comp)
        to_comp = int(to_comp)
        all_data = self.fractions[:].flat
        sum_others = 0
        for idx, val in enumerate(all_data):
            if idx < from_comp or idx > to_comp:
                sum_others += val
        return numpy.array(all_data[from_comp:to_comp].tolist() + [sum_others])

    def read_weights_data(self, from_comp, to_comp):
        """
        Return the weights data for the components in the interval [from_comp, to_comp].
        """
        from_comp = int(from_comp)
        to_comp = int(to_comp)
        data_slice = slice(from_comp, to_comp, None)
        weights_shape = self.weights.shape
        weights_slice = [slice(size) for size in weights_shape]
        weights_slice[0] = data_slice
        weights_data = self.weights[tuple(weights_slice)]
        return weights_data.flatten()


class IndependentComponentsH5(H5File):

    def __init__(self, path):
        super(IndependentComponentsH5, self).__init__(path)
        self.source = Reference(IndependentComponents.source, self)
        self.mixing_matrix = DataSet(IndependentComponents.mixing_matrix, self, expand_dimension=2)
        self.unmixing_matrix = DataSet(IndependentComponents.unmixing_matrix, self, expand_dimension=2)
        self.prewhitening_matrix = DataSet(IndependentComponents.prewhitening_matrix, self, expand_dimension=2)
        self.n_components = Scalar(IndependentComponents.n_components, self)
        self.norm_source = DataSet(IndependentComponents.norm_source, self, expand_dimension=1)
        self.component_time_series = DataSet(IndependentComponents.component_time_series,
                                             self, expand_dimension=1)
        self.normalised_component_time_series = DataSet(IndependentComponents.normalised_component_time_series,
                                                        self, expand_dimension=1)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.unmixing_matrix.append(partial_result.unmixing_matrix, close_file=False)
        self.prewhitening_matrix.append(partial_result.prewhitening_matrix, close_file=False)
        partial_result.compute_norm_source()
        self.norm_source.append(partial_result.norm_source, close_file=False)
        partial_result.compute_component_time_series()
        self.component_time_series.append(partial_result.component_time_series, close_file=False)
        partial_result.compute_normalised_component_time_series()
        self.normalised_component_time_series.append(partial_result.normalised_component_time_series, close_file=False)
        partial_result.compute_mixing_matrix()
        self.mixing_matrix.append(partial_result.mixing_matrix)
