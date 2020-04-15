# coding=utf-8
import numpy as np
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits
from tvb.basic.readers import H5Reader
from tvb.contrib.scripts.utils.data_structures_utils import labels_to_inds


class BaseModel(HasTraits):

    @staticmethod
    def from_instance(instance, **kwargs):
        return (instance.__class__())._copy_from_instance(instance, **kwargs)

    @staticmethod
    def set_attributes(instance, **kwargs):
        for attr, value in kwargs.items():
            try:
                setattr(instance, attr, value)
            except:
                get_logger(__name__).warning("Failed to set attribute %s to %s!" % (attr, instance.__class__.__name__))
        return instance

    @staticmethod
    def copy_attributes(trg_instance, src_instance, **kwargs):
        attributes = dict(src_instance.__dict__)
        attributes.update(kwargs)
        del attributes["gid"]
        return BaseModel.set_attributes(trg_instance, **attributes)

    def _copy_from_instance(self, instance, **kwargs):
        return BaseModel.copy_attributes(self, instance, **kwargs)

    @classmethod
    def from_tvb_instance(cls, instance, **kwargs):
        return cls()._copy_from_instance(instance, **kwargs)

    @classmethod
    def from_h5_file(cls, source_file, **kwargs):
        result = cls()
        reader = H5Reader(source_file)
        for attr in dir(result):
            try:
                val = reader.read_field(attr)
            except:
                continue
            if isinstance(val, np.ndarray) and val.dtype == "O":
                setattr(result, attr, np.array([np.str(v) for v in val]))
            else:
                setattr(result, attr, val)
        for attr, value in kwargs.items():
            setattr(result, attr, value)
        return result

    @classmethod
    def from_tvb_file(cls, filepath, return_tvb_instance=False, **kwargs):
        tvb_instance = cls.from_file(filepath)
        result = cls.from_tvb_instance(tvb_instance, **kwargs)
        if return_tvb_instance:
            return result, tvb_instance
        else:
            return result

    def to_tvb_instance(self, tvb_datatype, **kwargs):
        return BaseModel.copy_attributes(tvb_datatype(), self, **kwargs)

    @staticmethod
    def labels2inds(all_labels, labels):
        return labels_to_inds(all_labels, labels)
