from kubernetes import client, config
import time
from tvb_library.tvb.basic.logger.builder import get_logger

LOGGER = get_logger(__name__)


class KubeManager:
    def __init__(self, v1, apps_v1, namespace: str):
        self.v1 = v1
        self.apps_v1 = apps_v1
        self.namespace = namespace

    @classmethod
    def from_kube_config(cls, namespace: str):
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()
        return cls(v1, apps_v1, namespace)

    @classmethod
    def from_custom_kube_config(cls, host: str, bearer_token: str, namespace: str):
        """
            @host: Rancher host: https://rancher.tc.humanbrainproject.eu/k8s/clusters/c-m-jcx2qhqn
            @bearer_token: account generated token
        """
        configuration = client.Configuration()
        configuration.host = host
        configuration.verify_ssl = True
        configuration.api_key['authorization'] = bearer_token
        api_client = client.ApiClient(configuration)
        v1 = client.CoreV1Api(api_client)
        apps_v1 = client.AppsV1Api(api_client)
        return cls(v1, apps_v1, namespace)

    def get_pods(self, application):
        pods = None
        try:
            response = self.fetch_endpoints(application)
            pods = response[0].subsets[0].addresses
            LOGGER.info('Retrieve rancher pods for application {}'.format(application))
        except Exception as e:
            LOGGER.error('Failed to retrieve rancher pods for application {}'.format(application), e)
        return pods

    def fetch_endpoints(self, target_application):
        response = self.v1.read_namespaced_endpoints_with_http_info(name=target_application, namespace=self.namespace)
        return response

    def create_pod_from_deployment(self, deployment_name: str):
        try:
            deployment = self.apps_v1.read_namespaced_deployment(deployment_name, self.namespace)
            deployment.spec.replicas += 1
            pod = self.apps_v1.replace_namespaced_deployment(deployment_name, self.namespace, deployment)
            time.sleep(0.05)
            LOGGER.info('Created pod from application {}'.format(pod.metadata.name))
        except Exception as e:
            LOGGER.error('Failed to scale up deployment {}'.format(deployment_name), e)

    def delete_pod_from_deployment(self, deployment_name: str):
        deployment = self.apps_v1.read_namespaced_deployment(deployment_name, self.namespace)
        try:
            deployment.spec.replicas -= 1
            pod = self.apps_v1.replace_namespaced_deployment(deployment_name, self.namespace, deployment)
            LOGGER.info('Deleted pod from application {}'.format(pod.metadata.name))
        except Exception as e:
            LOGGER.error('Failed to scale down {}'.format(deployment_name), e)
