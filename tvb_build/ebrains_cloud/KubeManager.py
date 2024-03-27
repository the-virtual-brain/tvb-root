from kubernetes import client, config
from kubernetes.config import incluster_config


class KubeManager(object):
    def __init__(self, host, bearer_token, namespace):
        configuration = client.Configuration()
        configuration.host = host
        configuration.verify_ssl = True
        configuration.api_key['authorization'] = bearer_token
        api_client = client.ApiClient(configuration)
        self.v1 = client.CoreV1Api(api_client)
        self.namespace = namespace

    def get_pods(self, application):
        pods = None

        try:
            response = self.fetch_endpoints(application)
            pods = response[0].subsets[0].addresses

        except Exception as e:
            print("Failed to retrieve rancher pods for application {}".format(application), e)
            # LOGGER.error("Failed to retrieve openshift pods for application {}".format(application), e)
        return pods

    def fetch_endpoints(self, target_application):
        # todo check on which image the load_incluster_config() works in a k8s pod
        # config.load_incluster_config()
        # v1 = client.CoreV1Api()
        # print(v1.api_client)
        # v1 = self.init_client(host, bearer_token)
        response = self.v1.read_namespaced_endpoints_with_http_info(name=target_application, namespace=self.namespace)
        return response

    def create_pod(self, pod_name, namespace):
        pod_manifest = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': pod_name
            },
            'spec': {
                'containers': [{
                    'image': 'busybox',
                    'name': 'sleep',
                    "args": [
                        "/bin/sh",
                        "-c",
                        "while true;do date;sleep 5; done"
                    ]
                }]
            }
        }
        self.v1.create_namespaced_pod(body=pod_manifest, namespace=namespace)

    def delete_pod(self, pod_name, namespace):
        self.v1.delete_namespaced_pod(
            name=pod_name,
            namespace=namespace)

    @staticmethod
    def get_authorization_token():
        kube_config = incluster_config.InClusterConfigLoader(
            token_filename=incluster_config.SERVICE_TOKEN_FILENAME,
            cert_filename=incluster_config.SERVICE_CERT_FILENAME,
            try_refresh_token=True)
        kube_config.load_and_set(None)
        return kube_config.token

    @staticmethod
    def get_authorization_header():
        token = KubeManager.get_authorization_token()
        return {"Authorization": "{}".format(token)}

    @staticmethod
    def check_token(authorization_token):
        expected_token = KubeManager.get_authorization_token()
        assert authorization_token == expected_token

    # @staticmethod
    # def notify_pods(url, target_application=TvbProfile.current.web.OPENSHIFT_APPLICATION):
    #
    #     if not TvbProfile.current.web.OPENSHIFT_DEPLOY:
    #         return
    #
    #     LOGGER.info("Notify all pods with url {}".format(url))
    #     openshift_pods = KubeNotifier.get_pods(target_application)
    #     url_pattern = "http://{}:" + str(TvbProfile.current.web.SERVER_PORT) + url
    #     auth_header = KubeNotifier.get_authorization_header()
    #
    #     with ThreadPoolExecutor(max_workers=len(openshift_pods)) as executor:
    #         for pod in openshift_pods:
    #             pod_ip = pod.ip
    #             LOGGER.info("Notify pod: {}".format(pod_ip))
    #             executor.submit(requests.get, url=url_pattern.format(pod_ip), headers=auth_header)
