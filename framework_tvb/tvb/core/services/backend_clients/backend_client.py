from abc import abstractmethod


class BackendClient(object):
    """
    Interface for a backend client that runs operations asynchronously on a specific environment
    """

    @staticmethod
    @abstractmethod
    def execute(operation_id, user_name_label, adapter_instance):
        """
        Start operation asynchronously
        """

    @staticmethod
    @abstractmethod
    def stop_operation(operation_id):
        """
        Stop the thread for a given operation id
        """
