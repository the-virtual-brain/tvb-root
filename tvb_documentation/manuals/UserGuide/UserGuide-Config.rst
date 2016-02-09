
.. _configuring_TVB:

Configuring TVB
===============

The preferred method to configure |TVB| is from the web interface. See :ref:`tvb_settings_ui`.

However if |TVB| is installed on a headless server (no GUI), then the web interface might not be available remotely.
See :ref:`tvb_settings_headless`.


.. _tvb_settings_headless:

Configuring a headless TVB
--------------------------

In order to configure TVB in a headless environment, create a file named `.tvb.configuration` in the home directory
of the current OS user which is launching |TVB|.
Copy the following content and edit it to suit your needs. ::

    MAXIMUM_NR_OF_OPS_IN_RANGE=2000
    URL_WEB=http://127.0.0.1:8080/
    ADMINISTRATOR_EMAIL=jira.tvb@gmail.com
    MATLAB_EXECUTABLE=/usr/bin/octave
    MAXIMUM_NR_OF_THREADS=4
    WEB_SERVER_PORT=8080
    URL_MPLH5=ws://127.0.0.1:9000/
    LAST_CHECKED_CODE_VERSION=6507
    USR_DISK_SPACE=5242880
    DEPLOY_CLUSTER=False
    ADMINISTRATOR_NAME=admin
    LAST_CHECKED_FILE_VERSION=2
    URL_VALUE=sqlite:////home/tvb_user/TVB/tvb-database.db
    ADMINISTRATOR_PASSWORD=[[md5 of password]]
    SELECTED_DB=sqlite
    MAXIMUM_NR_OF_VERTICES_ON_SURFACE=300000
    MPLH5_SERVER_PORT=9000
    TVB_STORAGE=/home/tvb_user/TVB

Usually one would change the web server port and domain.
|TVB| will create a folder with project data named TVB (at the path specified by line starting with `TVB_STORAGE`).
By default it is located in the users home directory.
You can change the `TVB_STORAGE` to point to a different location.

Finally run the appropriate script for your platform (as described in the previous chapter), to launch |TVB| with the new settings.


Setting up a client/server configuration
----------------------------------------

This is for when you want one |TVB| instance to service many users via the web interface.
In such a setup the console interfaces of tvb are not available to remote users.

It is likely that you will have to change http ports and the path where |TVB| will store project data.
Depending on the processing power of the server you might want to adjust the maximum number of operations threads and vertices.
You may also want to adjust the maximum disk quota for each tvb user.

In this highly concurrent setup we strongly recommended to use PostgreSQL as the metadata storage of TVB.

    1. You should install PostgreSQL DB, independently of TVB. For Windows user, see the next chapter on how this can be easily achieved.
    2. Create a database called `tvb`, in PostgreSQL
    3. Choose the postgres user that TVB will use to connect. Any user with rights over `tvb` database is ok.
       These are *not* tvb accounts but db accounts.
    4. Create a DB connection URI. Postgres URI's in TVB have this general form ::

        postgresql+psycopg2://postgres:root@[postresql-server-host]:[postgres-port]/tvb?user=[user]&password=[postgres-pwd]
        # The angle bracketed expressions are placeholders that have to be replaced by values specific to your machine.

    5. Place the concrete connection URI in the |TVB| configuration using either the GUI or by editing the config file.


Installing PosgreSQL on Windows
-------------------------------

Getting PostgreSQL database up and running isn't too difficult on Windows:

    - Grab a copy for 32 or 64 bit from http://www.enterprisedb.com/products-services-training/pgdownload#windows
    - Install, noting the port number and user/pass. These will be later on needed in TVB, when writing the connection URL
    - [optional when using TVB Git repositories directly]
        * `pip install psycopg2`, with the `PATH` set correctly for your Python installation, or
        * grab and install from http://www.lfd.uci.edu/~gohlke/pythonlibs/#psycopg
        * test `python -c "import psycopg2"`, if ImportError, find libpq.dll (e.g. `c://program files/postgresql/9.3/bin`) and add it to the `PATH`.
    - open pgAdmin, right click databases, add a database named `tvb`, click ok, close pgadmin
    - stop, clean (this deletes all tvb previous data!) and start.
        * This cleanup is necessary if you have started TVB Distribution before and used sqlite DB.
        * in case you have important TVB data that should not be lost, you can always use Project Export, clean, restart, and then Project Import after the new DB backend has been set up.
    - in TVB GUI settings page, select `postgresql`, edit the port number (if not default `5432`) and DB password in the DB connection URL
    - validate db (by using the GUI button). It should be ok, if not: look at the output in terminal to see more details.


