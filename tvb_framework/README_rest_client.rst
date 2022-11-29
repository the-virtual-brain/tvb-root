TVB REST client
===============

The `tvb-rest-client` is a helper package built with the intention to simplify a
Python client interaction with TVB REST Server.
All the logic necessary to prepare and send requests towards the REST server, is embedded under this client API.

**GET** requests are sent from this python client using the **requests** library.

For the **POST** requests, a client has to attach a file with some input configuration.
Such a file is usually an **H5** in TVB specific format.
Thus, `tvb-rest-client` has all the logic for preparing those H5 files and sending requests.
Also, the REST server uses a Keycloak client at log in time, so this client will open a browser that allows the user to
log in, before attempting to make the requests.

Usage
=====
You should have a TVB REST server running, or access to a public one.
Then into `tvb-rest-client` you need to provide the URL towards this TVB REST server.
For the following example, we will suppose TVB REST server runs on *http://localhost:9090*

To launch a TVB REST server locally, you should download `tvb-framework` version >2.0. and launch it::

    $ python -m tvb.interfaces.web.run WEB_PROFILE  # Launch TVB web and REST servers locally


Accessing the client API entry-point
-------------------------------------

If the TVB REST server you want to access runs at another address, change the parameter
in the bellow TVBClient instantiation.

.. code-block:: python

    from tvb.interfaces.rest.client.tvb_client import TVBClient
    tvb_client = TVBClient("http://localhost:9090")
..


Attempt to login and start using the client API to send requests, by calling different types of methods:

- methods that return a list of DTOs

.. code-block:: python

    tvb_client.browser_login()
    list_of_user_projects = tvb_client.get_project_list()
    list_of_datatypes_in_project = tvb_client.get_data_in_project(list_of_user_projects[0].gid)
    list_of_operations_for_datatype = tvb_client.get_operations_for_datatype(list_of_datatypes_in_project[0].gid)
..

- method that download data files locally, under a folder chosen by the client

.. code-block:: python

    datatype_path = tvb_client.retrieve_datatype(list_of_datatypes_in_project[0].gid, download_folder)
..

- method that loads in memory the datatype downloaded previously

.. code-block:: python

    datatype = tvb_client.load_datatype_from_file(datatype_path)
..

- methods that launch operations in the TVB server
    Such an operation requires the client to prepare the operation configuration and send it in an H5 file together with the requests.

    By using the client API, the user only needs to instantiate the proper Model class and send it as argument to the following method.
    It wraps the serialization of the Model inside the H5 and the attaching to the POST request.

    The example above launches a Fourier analyzer, we suppose the Fourier AlgorithmDTO is *list_of_operations_for_datatype[0]*.

.. code-block:: python

    from tvb.adapters.analyzers.fourier_adapter import FFTAdapterModel, FourierAdapter

    project_gid = list_of_user_projects[0].gid
    model = FFTAdapterModel()
    # logic to fill the model with required attributes
    operation_gid = tvb_client.launch_operation(project_gid, FourierAdapter, model)
..

- method to monitor the status of an operation

.. code-block:: python

    monitor_operation(tvb_client, operation_gid)
..

Acknowledgments
===============
This project has received funding from the European Union’s Horizon 2020 Framework Programme for Research and
Innovation under the Specific Grant Agreement Nos. 785907 (Human Brain Project SGA2), 945539 (Human Brain Project SGA3)
and VirtualBrainCloud 826421.