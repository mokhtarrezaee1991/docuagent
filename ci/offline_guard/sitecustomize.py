"""Blockiert externe Python-Socket-Verbindungen nur im CI-Testprozess.

Aktivierung erfolgt explizit per PYTHONPATH, auch fuer Kindprozesse.
Loopback und lokale IPC bleiben fuer TestClient/Streamlit verfuegbar.
Dies ist eine Schutzschicht und keine Betriebssystem-Netzwerksandbox.
"""

import ipaddress
import socket


def _check_host(host):
    if host is None or host in ("localhost", b"localhost"):
        return
    if isinstance(host, bytes):
        host = host.decode("ascii")
    try:
        if ipaddress.ip_address(host.split("%", 1)[0]).is_loopback:
            return
    except (ValueError, AttributeError):
        pass
    raise OSError("External networking is disabled during offline tests")


_original_getaddrinfo = socket.getaddrinfo
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_sendto = socket.socket.sendto


def _getaddrinfo(host, *args, **kwargs):
    _check_host(host)
    return _original_getaddrinfo(host, *args, **kwargs)


def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check_host(address[0])
    return _original_connect(self, address)


def _connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check_host(address[0])
    return _original_connect_ex(self, address)


def _sendto(self, data, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check_host(args[-1][0])
    return _original_sendto(self, data, *args)


socket.getaddrinfo = _getaddrinfo
socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.socket.sendto = _sendto
NETWORK_GUARD_ACTIVE = True
