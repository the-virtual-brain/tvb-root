TVB REST client
===============

The tvb-rest-client is a helper package built with the intention to simplify the client interaction with TVB REST API.

All the logic necessary to prepare and send requests towards the REST server, is embeded under a client API.

**GET** requests are sent from this python client using the **requests** library.

For the **POST** requests, a client has to attach a file with some input configuration.
Such a file is usually an **H5** in TVB specific format.
Thus, tvb-rest-client has all the logic for preparing those H5 files and sending requests.

Usage
=====
You should provide the URL towards the TVB REST server.
For the following example, we will suppose TVB REST server runs on *http://localhost:9090*

Accessing the client API entrypoint:
------------------------------------

If the TVB REST server you want to access runs at another address, change the parameter in the bellow TVBClient instantiation.

.. code-block:: python

    from tvb.interfaces.rest.client.tvb_client import TVBClient
    main_client = TVBClient("http://localhost:9090")
..


Start using the client API to send requests, by calling different types of methods:

- methods that return a list of DTOs

.. code-block:: python

    list_of_users = main_client.get_users()
    list_of_user_projects = main_client.get_project_list(list_of_users[0].username)
    list_of_datatypes_in_project = main_client.get_data_in_project(list_of_user_projects[0].gid)
    list_of_operations_for_datatype = main_client.get_operations_for_datatype(list_of_datatypes_in_project[0].gid)
..

- methdos that download data files locally, under a folder chosen by the client

.. code-block:: python

    main_client.retrieve_datatype(list_of_datatypes_in_project[0].gid, download_folder)
..

- methods that launch operations in the TVB server
    Such an operation requires the client to prepare the operation configuration and send it in an H5 file together with the requests.

    By using the client API, the user only needs to instantiate the proper Model class and send it as argument to the following method.
    It wraps the serialization of the Model inside the H5 and the attaching to the POST request.

    The example above launches a Fourier analyzer, we suppose the Fourier AlgorithmDTO is *list_of_operations_for_datatype[0]*.

.. code-block:: python

    project_gid = list_of_user_projects[0].gid
    operation_module = list_of_operations_for_datatype[0].module
    operation_class = list_of_operations_for_datatype[0].classname
    model = FFTAdapterModel()
    # logic to fill the model with required attributes
    operation_gid = main_client.launch_operation(project_gid, operation_module, operation_class, model)
..

- method to monitor the status of an operation

.. code-block:: python

    status = main_client.get_operation_status(operation_gid)
..