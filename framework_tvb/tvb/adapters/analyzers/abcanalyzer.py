from tvb.core.adapters.abcadapter import ABCAdapter


class ABCAnalyzer(ABCAdapter):

    def configure(self, view_model):
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series.hex)
        self.generic_attributes.parent_burst = self.input_time_series_index.fk_parent_burst
