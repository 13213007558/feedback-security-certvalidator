"""
Policy-aware HTTP transport for the ``aiohttp`` fetcher backend.

The admission policy is enforced both before the request (by the shared
guarded request loop) and at the connector level: the custom resolver below
is the only component allowed to turn host names into addresses, and it
discards every address the policy refuses. Because aiohttp consults the
resolver whenever it opens a new connection (including after pool eviction),
connection pooling can never bypass the policy.
"""

import socket
from typing import Mapping, Optional

import aiohttp

from .._guarded_http import GuardedResponse, GuardedTransport
from ..network_policy import (
    NetworkAdmissionPolicy,
    NetworkPolicyError,
    PolicyDenialReason,
)

__all__ = [
    'PolicyAwareResolver',
    'make_policy_aware_session',
    'AIOHttpGuardedTransport',
]


class PolicyAwareResolver(aiohttp.abc.AbstractResolver):
    """
    aiohttp resolver adapter that delegates to the policy's
    :class:`~pyhanko_certvalidator.fetchers.network_policy.HostResolver`
    and filters the results through the IP-level rules.
    """

    def __init__(self, policy: NetworkAdmissionPolicy):
        self.policy = policy

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ):
        results = []
        try:
            addresses = await self.policy.resolve_and_check_async(
                host, port, url=None
            )
        except NetworkPolicyError:
            raise
        for address in addresses:
            if family == socket.AF_INET and address.version != 4:
                continue
            if family == socket.AF_INET6 and address.version != 6:
                continue
            results.append(
                {
                    'hostname': host,
                    'host': str(address),
                    'port': port,
                    'family': socket.AF_INET
                    if address.version == 4
                    else socket.AF_INET6,
                    'proto': socket.IPPROTO_TCP,
                    'flags': socket.AI_NUMERICHOST | socket.AI_NUMERICSERV,
                }
            )
        if not results:
            raise NetworkPolicyError(
                f"No admissible address of the requested address family "
                f"available for host {host!r}",
                reason=PolicyDenialReason.NO_USABLE_ADDRESS,
            )
        return results

    async def close(self) -> None:
        resolver = self.policy.resolver
        if resolver is not None:
            await resolver.close()


def _make_connector(policy: NetworkAdmissionPolicy) -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(
        resolver=PolicyAwareResolver(policy),
        # Every connection attempt must go through the policy-aware resolver,
        # including destinations whose addresses are already cached elsewhere.
        ttl_dns_cache=0,
    )


def make_policy_aware_session(
    policy: NetworkAdmissionPolicy,
) -> aiohttp.ClientSession:
    """
    Build a plain :class:`aiohttp.ClientSession` whose connector is bound to
    the admission policy. A factory (rather than a ``ClientSession``
    subclass) keeps the session compatible with aiohttp's own lifecycle
    expectations.
    """

    return aiohttp.ClientSession(connector=_make_connector(policy))


class AIOHttpGuardedTransport(GuardedTransport):
    def __init__(
        self,
        policy: NetworkAdmissionPolicy,
        session_provider=None,
    ):
        self.policy = policy
        # When a fetcher already owns a LazySession, reuse its single
        # policy-aware session instead of opening a second connection pool.
        self._session_provider = session_provider
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session_provider is not None:
            return await self._session_provider()
        if self._session is None:
            self._session = make_policy_aware_session(self.policy)
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
        session = await self._get_session()
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.request(
            method=method,
            url=url,
            headers=dict(headers),
            data=data,
            timeout=client_timeout,
            allow_redirects=False,
        ) as response:
            content = await self._read_limited(
                response, url, max_response_size
            )
            return GuardedResponse(
                status_code=response.status,
                headers=response.headers,
                content=content,
                url=str(response.url),
            )

    @staticmethod
    async def _read_limited(
        response: aiohttp.ClientResponse,
        url: str,
        max_response_size: int,
    ) -> bytes:
        chunks = []
        total = 0
        async for chunk in response.content.iter_chunked(65536):
            total += len(chunk)
            if total > max_response_size:
                response.close()
                raise NetworkPolicyError(
                    f"Response body from {url} exceeds the maximum allowed "
                    f"size of {max_response_size} bytes",
                    reason=PolicyDenialReason.RESPONSE_TOO_LARGE,
                    url=url,
                )
            chunks.append(chunk)
        return b''.join(chunks)

    async def close(self) -> None:
        session = self._session
        if session is not None:
            await session.close()
            self._session = None
