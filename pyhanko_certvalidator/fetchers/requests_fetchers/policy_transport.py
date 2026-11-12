"""
Policy-aware HTTP transport for the ``requests`` fetcher backend.

The admission policy is enforced at two layers:

1. :func:`~pyhanko_certvalidator.fetchers._guarded_http.guarded_http_request`
   validates the URL (and every redirect target) and pre-resolves the host
   through the policy's resolver before anything is sent.
2. A custom urllib3 connection class repeats the resolution and the IP-level
   check at actual connect time, so neither a reused connection pool nor a
   different code path inside urllib3 can open a socket to an unapproved
   address.

Redirects are handled manually with the library's own redirect machinery
switched off, and response bodies are streamed with a hard size cap.
"""

import socket
from typing import Mapping, Optional

import requests
import urllib3
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import (
    HTTPConnectionPool,
    HTTPSConnectionPool,
)
from urllib3.poolmanager import PoolManager

from ..._asyncio_compat import to_thread
from .._guarded_http import GuardedResponse, GuardedTransport
from ..network_policy import (
    NetworkAdmissionPolicy,
    NetworkPolicyError,
    PolicyDenialReason,
)

__all__ = ['PolicyAwareRequestsSession', 'RequestsGuardedTransport']


def _connect_to_approved_address(
    connection,
    policy: NetworkAdmissionPolicy,
    dns_host: str,
    port: int,
    timeout,
):
    """
    Replacement for urllib3's default socket establishment.

    The hostname is resolved through the policy's resolver (never through
    urllib3's internal ``getaddrinfo`` call) and every candidate address is
    checked against the policy before a socket is opened.
    """

    connect_timeout = getattr(timeout, 'connect_timeout', timeout)
    approved = policy.resolve_and_check(dns_host, port)
    last_error: Optional[OSError] = None
    for address in approved:
        sock = socket.socket(
            family=socket.AF_INET
            if address.version == 4
            else socket.AF_INET6,
            type=socket.SOCK_STREAM,
        )
        try:
            sock.settimeout(connect_timeout)
            sock.setsockopt(
                socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
            )
            if address.version == 4:
                sock.connect((str(address), port))
            else:
                sock.connect((str(address), port, 0, 0))
            return sock
        except OSError as e:
            last_error = e
            sock.close()
    raise NetworkPolicyError(
        f"Unable to establish a connection to any approved address for "
        f"host {dns_host!r} (last error: {last_error})",
        reason=PolicyDenialReason.NO_USABLE_ADDRESS,
    )


def _build_connection_classes(policy: NetworkAdmissionPolicy):
    class _PolicyHTTPConnection(HTTPConnection):
        def _new_conn(self):
            return _connect_to_approved_address(
                self,
                policy,
                self._dns_host,
                self.port,
                self.timeout,
            )

    class _PolicyHTTPSConnection(HTTPSConnection):
        def _new_conn(self):
            return _connect_to_approved_address(
                self,
                policy,
                self._dns_host,
                self.port,
                self.timeout,
            )

    class _PolicyHTTPConnectionPool(HTTPConnectionPool):
        ConnectionCls = _PolicyHTTPConnection

    class _PolicyHTTPSConnectionPool(HTTPSConnectionPool):
        ConnectionCls = _PolicyHTTPSConnection

    return _PolicyHTTPConnectionPool, _PolicyHTTPSConnectionPool


class _PolicyPoolManager(PoolManager):
    def __init__(self, policy: NetworkAdmissionPolicy, **kwargs):
        http_pool, https_pool = _build_connection_classes(policy)
        super().__init__(**kwargs)
        self.pool_classes_by_scheme = {
            'http': http_pool,
            'https': https_pool,
        }


class _PolicyHTTPAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, policy: NetworkAdmissionPolicy, **kwargs):
        self._policy = policy
        super().__init__(**kwargs)

    def init_poolmanager(
        self, connections, maxsize, block=False, **pool_kwargs
    ):
        self.poolmanager = _PolicyPoolManager(
            self._policy,
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


class PolicyAwareRequestsSession(requests.Session):
    """
    ``requests`` session whose sockets are constrained by an admission
    policy. The session never follows redirects on its own; the guarded
    request loop re-validates each redirect target first.

    Ambient proxy configuration (``HTTP_PROXY`` and friends) is deliberately
    ignored: routing through a proxy would make the actual socket peer the
    proxy rather than the policy-approved address, and credentials embedded
    in proxy URLs would bypass the userinfo rules.
    """

    def __init__(self, policy: NetworkAdmissionPolicy):
        super().__init__()
        adapter = _PolicyHTTPAdapter(policy)
        self.mount('http://', adapter)
        self.mount('https://', adapter)
        # Redirects are always issued with allow_redirects=False; the
        # library's own redirect machinery is never used.
        # Do not pick up proxies, .netrc or other environment configuration.
        self.trust_env = False

    def request(self, *args, **kwargs):
        # Belt-and-braces: never inherit a proxy, even if one was passed
        # explicitly by an inner layer.
        kwargs.setdefault('proxies', {})
        return super().request(*args, **kwargs)


class RequestsGuardedTransport(GuardedTransport):
    """Blocking ``requests`` transport exposed as an awaitable interface."""

    def __init__(self, policy: NetworkAdmissionPolicy):
        self.policy = policy
        self._session: Optional[PolicyAwareRequestsSession] = None

    def _get_session(self) -> PolicyAwareRequestsSession:
        if self._session is None:
            self._session = PolicyAwareRequestsSession(self.policy)
        return self._session

    async def send(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        data: Optional[bytes],
        timeout: float,
        max_response_size: int,
    ) -> GuardedResponse:
        return await to_thread(
            self._send_blocking,
            method=method,
            url=url,
            headers=dict(headers),
            data=data,
            timeout=timeout,
            max_response_size=max_response_size,
        )

    def _send_blocking(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        data: Optional[bytes],
        timeout: float,
        max_response_size: int,
    ) -> GuardedResponse:
        session = self._get_session()
        # stream=True keeps the body off the wire until we can enforce the
        # size limit incrementally.
        with session.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        ) as response:
            content = self._read_limited(response, url, max_response_size)
            return GuardedResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=content,
                url=str(response.url),
            )

    @staticmethod
    def _read_limited(response, url: str, max_response_size: int) -> bytes:
        chunks = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_response_size:
                    raise NetworkPolicyError(
                        f"Response body from {url} exceeds the maximum "
                        f"allowed size of {max_response_size} bytes",
                        reason=PolicyDenialReason.RESPONSE_TOO_LARGE,
                        url=url,
                    )
                chunks.append(chunk)
        finally:
            response.close()
        return b''.join(chunks)

    async def close(self) -> None:
        session = self._session
        if session is not None:
            await to_thread(session.close)
            self._session = None
