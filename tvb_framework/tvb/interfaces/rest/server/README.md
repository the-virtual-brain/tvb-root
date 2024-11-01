# TVB REST Server
TVB REST services which can be used to interact with TVB. 

    

## Guide to setup TVB REST server locally

To be able to run TVB REST server locally you need a keycloak server for obtaining authorization tokens. Follow the steps to get started with the local setup of TVB REST Server.

Before you start 
- Make sure docker is installed and it is up and running.
- By default these ports will be used in this guide, so make sure they are free:
    - [localhost:8080](http://localhost:8080) for running TVB(GUI)
    - [localhost:9090](http://localhost:9090) for running TVB REST server
    - [localhost:8081](http://localhost:8081) for running Keycloak server
    - [localhost:8888](http://localhost:8888) for running auth server used by rest server


### 1. Start keycloak server

From the command line. Enter the following command

```
docker run -p 8081:8080 -e KEYCLOAK_ADMIN=admin -e KEYCLOAK_ADMIN_PASSWORD=admin quay.io/keycloak/keycloak:17.0.0 start-dev
```
This will start your keycloak server at [localhost:8081](http://localhost:8081). It will also create an initial admin user with username `admin` and password `admin`

### 2. Create realm

Let's create a new realm for the TVB REST server. Open your [Keycloak Admin Console](http://localhost:8081/admin) and login with the above created username and password. By doing this you will log in your default master realm

   1. Click `Master` (dropdown) on top left and then click on `Add realm`
   2. Click `Import` and select [myrealm-realm.json](./dev_resources/myrealm-realm.json) file and then click on `Create`. This will automatically create a new realm as _myrealm_ for you

### 3. Create client

Now we will create a client for the TVB REST server inside _myrealm_. Open [Keycloak Admin Console](http://localhost:8081/admin) 

   1. Click `Clients` (left-hand menu)
   2. Click `Create` (top-right corner of table)
   3. Click `Import` and select [tvb-rest.json](./dev_resources/tvb-rest.json) file and then click on `Save`

### 4. Create User

We also need a user inside _myrealm_ so let's create a user. Open [Keycloak Admin Console](http://localhost:8081/admin)

   1. Click `Users` (left-hand menu)
   2. Click `Add user` (top-right corner of table)
   3. Provide values for required fields (username, first and last name) and click on `Save`
   4. Now set the password for this user, click `Credentials` (on top of page)
   5. Set the password, you can also make it temporary to avoid resetting the password on first login
   6. Remember the username and password because you'll need this for login when using TVB REST server

### 5. Configure TVB

Now we have to configure TVB to launch the REST server on app start-up. So start your main TVB app and navigate to [TVB Settings](http://localhost:8080/settings/settings/)

   1. Link the TVB REST server to your Keycloak instance:
        1. Enter [keycloak_config.json](./dev_resources/keycloak_config.json) in the `Rest API Keycloak configuration file` field. And make sure [keycloak_config.json](./dev_resources/keycloak_config.json) is accessible to the TVB app
        
        ![This is an image](./dev_resources/keycloak_setting_rest_only.jpg)

   2. Link the TVB GUI app to your Keycloak instance:
        1. Enable the keycloak login, click on checkbox
        2. Enter the same [keycloak_config.json](./dev_resources/keycloak_config.json) file name in the `Web Keycloak configuration file` field
                

        ![This is an image](./dev_resources/keycloak_setting_rest_and_gui.jpg)

   Click `Apply` to save changes and restart TVB.


If you've enabled keycloak login then after restart TVB will open the keycloak login window and you'll have to login with the username and password you've created earlier. 

If TVB rest server started successfully then you'll be able to see the rest server documentation at [http://localhost:9090/doc/](http://localhost:9090/doc).

### 6. Install tvb-rest-client (optional)
We provide the [tvb-rest-client](https://pypi.org/project/tvb-rest-client/) package to simplify interaction with the TVB REST server. It can be installed from Pypi:

```
pip install tvb-rest-client
```

Test the rest server, by firing a simulation from command line:

```
python -m tvb.interfaces.rest.client.examples.fire_simulation --rest-url=http://localhost:9090
```
 
Or from a script, using the [tvb-rest-client](https://pypi.org/project/tvb-rest-client/) API, by retrieving the user's projects:

```
from tvb.interfaces.rest.client.tvb_client import TVBClient
tvb_client = TVBClient("http://localhost:9090")
tvb_client.browser_login()
print(tvb_client.get_project_list())
```

Any operations launched using REST server are also visibile in the TVB GUI.

