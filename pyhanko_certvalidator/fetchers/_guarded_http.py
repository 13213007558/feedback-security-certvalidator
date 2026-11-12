"""
Shared, backend-agnostic HTTP plumbing used by both the ``requests`` and
``aiohttp`` fetcher backends when a :class:`NetworkAdmissionPolicy` is in
effect.

All policy-sensitive behaviour lives here so that both backends make the
same decisions. Backend-specific code only supplies a
:class:`GuardedTransport` that opens a single, already-authorised HTTP
exchange (no automatic redirects) and streams the response body while the
size limit is enforced.
"""

import abc
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urljoin

from .network_policy import NetworkAdmissionPolicy, NetworkPolicyError

__all__ = [
    'GuardedResponse',
    'GuardedTransport',
    'guarded_http_request',
    'REDIRECT_STATUSES',
]

# Redirect status codes that change the request method to GET.
_REDIRECT_TO_GET = frozenset([301, 302, 303])
# Redirect status codes that preserve the request method.
_REDIRECT_PRESERVE = frozenset([307, 308])
REDIRECT_STATUSES = _REDIRECT_TO_GET | _REDIRECT_PRESERVE


@dataclass
class GuardedResponse:
    """Minimal response representation shared by both backends."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes
    url: str


class GuardedTransport(abc.ABC):
    """
    One-hop HTTP transport.

    Implementations must:

    * connect only to the supplied, already-policy-approved addresses;
    * leave redirect handling to :func:`guarded_http_request` (i.e. disable
      the underlying library's automatic redirect handling);
    * raise :class:`NetworkPolicyError` themselves if a connection-level
      policy check fails, so a pooled connection can never bypass the
      policy.
    """

    @abc.abstractmethod
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
        raise NotImplementedError

    async def close(self) -> None:
        pass


async def guarded_http_request(
    transport: GuardedTransport,
    policy: NetworkAdmissionPolicy,
    url: str,
    *,
    method: str = 'GET',
    headers: Optional[Mapping[str, str]] = None,
    data: Optional[bytes] = None,
) -> GuardedResponse:
    """
    Execute an HTTP request while enforcing ``policy`` on the initial URL,
    every redirect target and every response body.

    No socket is opened until both the URL-level rules and the IP-level
    rules (applied to the addresses produced by the policy's resolver) have
    passed.
    """

    request_headers = dict(headers or {})
    current_url = url
    current_method = method.upper()
    current_data = data
    redirect_count = 0

    while True:
        # Re-evaluated from scratch on every iteration: initial URL *and*
        # each redirect target.
        scheme, hostname, port = policy.check_url(current_url)
        await policy.resolve_and_check_async(
            hostname, port, url=current_url
        )

        response = await transport.send(
            method=current_method,
            url=current_url,
            headers=request_headers,
            data=current_data,
            timeout=policy.per_request_timeout,
            max_response_size=policy.max_response_size,
        )

        if response.status_code not in REDIRECT_STATUSES:
            policy.check_response_size(len(response.content), url=current_url)
            return response

        location = _header_lookup(response.headers, 'location')
        if location is None:
            policy.check_response_size(len(response.content), url=current_url)
            return response

        redirect_count += 1
        policy.check_redirect_count(redirect_count, url=current_url)
        # Resolve relative redirects against the current URL; the joined
        # URL is then fully re-validated at the top of the next iteration.
        current_url = urljoin(current_url, location)
        if response.status_code in _REDIRECT_TO_GET:
            current_method = 'GET'
            current_data = None
            request_headers = {
                key: value
                for key, value in request_headers.items()
                if key.lower() != 'content-type'
            }


def _header_lookup(headers: Mapping[str, str], name: str) -> Optional[str]:
    # Both requests' CaseInsensitiveDict and aiohttp's CIMultiDict support
    # case-insensitive ``get``; fall back to a manual scan for plain dicts.
    value = headers.get(name)
    if value is not None:
        return value
    name_lower = name.lower()
    for key, candidate in headers.items():
        if key.lower() == name_lower:
            return candidate
    return None
