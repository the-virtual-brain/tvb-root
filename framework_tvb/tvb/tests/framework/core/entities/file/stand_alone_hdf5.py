# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>

This test is designed for the sole purpose to test for concurrecny problems during
simultaneous read operations of hdf5 files with either pytables, h5py or tvb's hdf5manager
(which currently also uses h5py as a base).

Is NOT intended to be included in the automated tests run periodically.
"""

import os
import sys
import threading
import numpy
import tables as hdf5
import h5py
import tvb.core.entities.file.hdf5_storage_manager as hdf5storage

excep_count = 0
isok = True
exception = ''
data_files = []
ROOT_STORAGE = os.path.dirname(__file__)
TEST_FILE_NAME = 'test_h5'


def test_using_hdf5manager(nr_of_threads):
    if os.path.exists(os.path.join(ROOT_STORAGE, TEST_FILE_NAME)):
        os.remove(os.path.join(ROOT_STORAGE, TEST_FILE_NAME))

    def init_some_data():
        dummy_data = numpy.random.random(size=(16000, 3))
        HDF5_MANAGER = hdf5storage.HDF5StorageManager(ROOT_STORAGE, TEST_FILE_NAME)
        HDF5_MANAGER.store_data('vertices', dummy_data)

    def read_some_data(th_nr):
        global isok
        global exception
        try:
            HDF5_MANAGER = hdf5storage.HDF5StorageManager(ROOT_STORAGE, TEST_FILE_NAME)
            read_data = HDF5_MANAGER.get_data('vertices')
            if read_data.shape != (16000, 3):
                raise Exception("Something went wrong")
            if int(th_nr) > 50000:
                raise Exception("GOT TO 50000 SHOULD STOP NOW")
        except Exception as ex:
            isok = False
            exception = ex.message

    init_some_data()
    read_some_data('1')

    th_nr = 1
    while isok:
        th_nr += 1
        for i in range(80):
            th = threading.Thread(target=read_some_data, kwargs={"th_nr": str(th_nr * 80 + i)})
            th.start()
        for thread in threading.enumerate():
            if thread is not threading.currentThread():
                thread.join()
    print("RAN FINE FOR " + str(th_nr * 80) + " THREADS")
    print("STOPPED DUE TO EXCEPTION " + str(exception))


def test_using_h5py(nr_of_threads):
    storage_path = os.path.join(ROOT_STORAGE, TEST_FILE_NAME)

    if os.path.exists(storage_path):
        os.remove(storage_path)

    def init_some_data():
        dummy_data = numpy.random.random(size=(16000, 3))
        h5_file = h5py.File(storage_path, 'w')
        h5_file.create_dataset('vertices', data=dummy_data)
        h5_file.close()

    def read_some_data(th_nr):
        global isok
        global exception
        try:
            h5_file = h5py.File(storage_path, 'r')
            read_data = h5_file['/vertices']
            if read_data.shape != (16000, 3):
                raise Exception("Actually got shape : " + str(read_data.shape))
            h5_file.close()
            if int(th_nr) > 50000:
                raise Exception("GOT TO 50000 SHOULD STOP NOW")
        except Exception as ex:
            isok = False
            exception = ex.message

    init_some_data()
    read_some_data('1')

    th_nr = 1
    while isok:
        th_nr += 1
        for i in range(80):
            th = threading.Thread(target=read_some_data, kwargs={"th_nr": str(th_nr * 80 + i)})
            th.start()
        for thread in threading.enumerate():
            if thread is not threading.currentThread():
                thread.join()
    print("RAN FINE FOR " + str(th_nr * 80) + " THREADS")
    print("STOPPED DUE TO EXCEPTION " + str(exception))


def test_using_pytables(nr_of_threads):
    storage_path = os.path.join(ROOT_STORAGE, TEST_FILE_NAME)

    if os.path.exists(storage_path):
        os.remove(storage_path)

    def init_some_data():
        dummy_data = numpy.random.random(size=(16000, 3))
        h5_file = hdf5.openFile(storage_path, 'w', "Testing purposes.")
        atom = hdf5.Atom.from_dtype(dummy_data.dtype)
        data_array = h5_file.createCArray('/', 'vertices', atom, dummy_data.shape,
                                          createparents=True, byteorder='little')
        data_array[:] = dummy_data
        h5_file.close()

    def read_some_data(th_nr):
        h5_file = hdf5.openFile(storage_path, 'r', "Testing purposes.")
        read_data = h5_file.getNode('/', 'vertices')
        read_data = read_data.read()
        if read_data.shape != (16000, 3):
            raise Exception("Something went wrong")
        try:
            h5_file.close()
        except KeyError:
            pass

    init_some_data()

    while 1:
        th = threading.Thread(target=read_some_data, kwargs={"th_nr": '1'})
        th.start()


if __name__ == '__main__':
    nr_of_threads = sys.argv[2]
    if sys.argv[1] == 'tvb':
        test_using_hdf5manager(int(nr_of_threads))
    elif sys.argv[1] == 'tables':
        test_using_pytables(int(nr_of_threads))
    elif sys.argv[1] == 'h5py':
        test_using_h5py(int(nr_of_threads))
