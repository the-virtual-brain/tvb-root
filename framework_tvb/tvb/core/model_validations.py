# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
This is intended to be a Benchmarking  and Validator script.

.. moduleauthor:: bogdan.neacsa
"""

from tvb.basic.profile import TvbProfile as tvb_profile
tvb_profile.set_profile(["-profile", "CONSOLE_PROFILE"], try_reload=False)

import sys
from time import sleep
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS
from tvb.core.services.flow_service import FlowService
from tvb.core.services.operation_service import OperationService
from tvb.core.entities.storage import dao
from tvb.core.entities.model import STATUS_STARTED, STATUS_FINISHED, STATUS_ERROR, PARAM_RANGE_PREFIX

KEY_PROJECT = 'project'



class ModelValidator(object):
    overwrites = {}


    def __init__(self, overwrites=None, settings_file=None):
        """ Parameters can be overwritten either from a settigns file or from a dictionary. """
        if overwrites is not None:
            self.overwrites.update(overwrites)
        if settings_file is not None:
            settings = open(sys.argv[1]).read()
            for line in settings.split('\n'):
                key, value = line.split('=')
                self.overwrites[key.strip()] = value.strip()
        if KEY_PROJECT not in self.overwrites:
            raise Exception("Settings file should contain the id of the project: %s=1" % KEY_PROJECT)
        self.project = dao.get_project_by_id(self.overwrites[KEY_PROJECT])
        self.flow_service = FlowService()
        self.operation_service = OperationService()


    def launch_validation(self):
        """
        Prepare the arguments to be submitted and launch actual operations group.
        TODO: Now get the results and check if any errors
        """
        _, algo_group = self.flow_service.get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)
        simulator_adapter = self.flow_service.build_adapter_instance(algo_group)
        launch_args = {}
        flatten_interface = simulator_adapter.flaten_input_interface()
        prepared_flatten_interface = self.flow_service.prepare_parameters(flatten_interface, self.project.id,
                                                                          algo_group.fk_category)
        for entry in prepared_flatten_interface:
            value = entry['default']
            if isinstance(value, dict):
                value = str(value)
            if hasattr(value, 'tolist'):
                value = value.tolist()
            launch_args[entry['name']] = value
        launch_args.update(self.overwrites)
        
        nr_of_operations = 1
        for key in self.overwrites:
            if key.startswith(PARAM_RANGE_PREFIX):
                range_values = self.operation_service.get_range_values(launch_args, key)
                nr_of_operations *= len(range_values)
        do_launch = False
        print "Warning! This will launch %s operations. Do you agree? (yes/no)" % nr_of_operations
        while 1:
            accept = raw_input()
            if accept.lower() == 'yes':
                do_launch = True
                break
            if accept.lower() == 'no':
                do_launch = False
                break
            print "Please type either yes or no"
        
        if do_launch:
            self.launched_operations = self.flow_service.fire_operation(simulator_adapter, self.project.administrator,
                                                                        self.project.id, **launch_args)
            return self.validate_results(0)
        else:
            return "Operation canceled by user."


    def validate_results(self, last_verified_index):
        error_count = 0
        while last_verified_index < len(self.launched_operations):
            operation_to_check = self.launched_operations[last_verified_index]
            operation = dao.get_operation_by_id(operation_to_check.id)
            if operation.status == STATUS_STARTED:
                sleep(10)
            if operation.status == STATUS_ERROR:
                sys.stdout.write("E(" + str(operation_to_check.id) + ")")
                error_count += 1
                last_verified_index += 1
                sys.stdout.flush()
            if operation.status == STATUS_FINISHED:
                last_verified_index += 1
                sys.stdout.write('.')
                sys.stdout.flush()
        if error_count:
            return "%s operations in error; %s operations successfully." % (error_count,
                                                                            len(self.launched_operations) - error_count)
        return "All operations finished successfully!"


def main(settings_file):
    validator = ModelValidator(settings_file=settings_file)
    print validator.launch_validation()
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Usage should be 'python model_validations.py settings_file")
    validator = ModelValidator(settings_file=sys.argv[1])
    print validator.launch_validation()
    exit(0)

