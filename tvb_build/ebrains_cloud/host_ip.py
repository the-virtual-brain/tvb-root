import socket


class HostIp:
    @staticmethod
    def get_host_current_host_ip():
        """
        :return: current host ip address
        """
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        return ip_addr
