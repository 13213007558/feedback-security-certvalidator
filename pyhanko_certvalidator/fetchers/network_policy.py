"""
Optional, backend-agnostic network admission policy for the built-in
fetcher backends.

The URLs used to download AIA certificates, CRLs and OCSP responses are
attested to by the certificate being validated, and therefore cannot be
trusted to point at well-behaved public hosts. A malicious (or merely
misconfigured) certificate can direct the validator at internal services,
cloud metadata endpoints or privileged ports.

The :class:`NetworkAdmissionPolicy` in this module centralises the admission
decision so that the ``requests`` and ``aiohttp`` fetcher backends apply the
exact same rules:

* the URL's scheme, host, port and ``userinfo`` component are checked before
  any network activity takes place;
* every redirect target is re-checked from scratch, up to a configurable
  maximum number of redirects;
* DNS name resolution goes through a pluggable resolver, and every resolved
  address is checked against the IP-level rules before a socket is opened.
  A hostname that passes the name-based rules can therefore never smuggle
  requests to an internal address via DNS rebinding;
* ``localhost`` aliases, IPv4-mapped IPv6 addresses (e.g.
  ``::ffff:127.0.0.1``) and other non-global addresses are recognised and
  handled explicitly rather than treated as ordinary public addresses;
* response bodies and timeouts are bounded.

Denials raise :class:`NetworkPolicyError`, which is *not* a subclass of the
fetch-error hierarchy, so it cannot be mistaken for missing validation
material and swallowed by tolerant ("soft fail") revocation logic.
"""

import abc
import enum
import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from urllib.parse import urlsplit

__all__ = [
    'PolicyDenialReason',
    'NetworkPolicyError',
    'HostResolver',
    'DefaultHostResolver',
    'NetworkAdmissionPolicy',
    'DEFAULT_MAX_RESPONSE_SIZE',
    'DEFAULT_MAX_REDIRECTS',
    'DEFAULT_REQUEST_TIMEOUT',
]

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESPONSE_SIZE = 10 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_REQUEST_TIMEOUT = 10

# Hostnames that unconditionally resolve to the local host. We refuse these
# even if the name-based rules would otherwise permit them, so that entries
# such as "localhost" in an allow-list cannot be used to reach loopback.
_LOCALHOST_ALIASES = frozenset(
    [
        'localhost',
        'localhost.localdomain',
        'local',
        'ip6-localhost',
        'ip6-loopback',
        'ip6-allnodes',
        'ip6-allrouters',
        'ip6-localnet',
    ]
)


class PolicyDenialReason(enum.Enum):
    """Machine-readable classification of a network policy denial."""

    UNSUPPORTED_SCHEME = 'unsupported_scheme'
    USERINFO_NOT_ALLOWED = 'userinfo_not_allowed'
    INVALID_URL = 'invalid_url'
    INVALID_HOST = 'invalid_host'
    HOST_NOT_ALLOWED = 'host_not_allowed'
    IP_NOT_ALLOWED = 'ip_not_allowed'
    PORT_NOT_ALLOWED = 'port_not_allowed'
    TOO_MANY_REDIRECTS = 'too_many_redirects'
    RESPONSE_TOO_LARGE = 'response_too_large'
    DNS_ERROR = 'dns_error'
    NO_USABLE_ADDRESS = 'no_usable_address'


class NetworkPolicyError(Exception):
    """
    Raised when the :class:`NetworkAdmissionPolicy` refuses a request or a
    response.

    This exception deliberately does not inherit from
    :class:`~pyhanko_certvalidator.errors.CertificateFetchError` or the
    revocation fetch-error hierarchy: a policy violation is a security
    decision, not a case of "the material happened to be unavailable".
    """

    def __init__(
        self,
        message: str,
        *,
        reason: PolicyDenialReason,
        url: Optional[str] = None,
        redirect_count: Optional[int] = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.url = url
        self.redirect_count = redirect_count

    def __str__(self):  # pragma: nocover
        base = super().__str__()
        if self.url is not None:
            return f'{base} (url={self.url}, reason={self.reason.value})'
        return f'{base} (reason={self.reason.value})'


@dataclass(frozen=True)
class _ResolvedName:
    hostname: str
    port: int
    addresses: Tuple[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], ...]


class HostResolver(abc.ABC):
    """
    Pluggable name resolution interface.

    Both backends resolve names through this abstraction so that admission
    decisions never depend on whatever resolver the underlying HTTP library
    happens to use internally. Tests can substitute a fake resolver to
    simulate DNS rebinding without touching the network.
    """

    @abc.abstractmethod
    def resolve_all(
        self, hostname: str, port: int
    ) -> Tuple[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], ...]:
        raise NotImplementedError

    async def resolve_all_async(
        self, hostname: str, port: int
    ) -> Tuple[Union[ipaddress.IPv4Address, ipaddress.IPv6Address], ...]:
        # Local import to avoid pulling in the compatibility shim when only
        # the synchronous interface is used.
        from .._asyncio_compat import to_thread

        return await to_thread(self.resolve_all, hostname, port)

    async def close(self) -> None:
        pass


class DefaultHostResolver(HostResolver):
    """Resolver backed by the standard library's :func:`socket.getaddrinfo`."""

    def resolve_all(self, hostname, port):
        try:
            infos = socket.getaddrinfo(
                hostname, port, type=socket.SOCK_STREAM
            )
        except socket.gaierror as e:
            raise NetworkPolicyError(
                f"Name resolution for host '{hostname}' failed: {e.strerror or e}",
                reason=PolicyDenialReason.DNS_ERROR,
            ) from e
        addresses: List[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]] = []
        seen = set()
        for family, _type, _proto, _canonname, sockaddr in infos:
            try:
                addr = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            addr = _canonicalise_ip(addr)
            if addr not in seen:
                seen.add(addr)
                addresses.append(addr)
        if not addresses:
            raise NetworkPolicyError(
                f"Name resolution for host '{hostname}' returned no usable "
                f"addresses",
                reason=PolicyDenialReason.NO_USABLE_ADDRESS,
            )
        return tuple(addresses)


def _canonicalise_ip(
    addr: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
) -> Union[ipaddress.IPv4Address, ipaddress.IPv6Address]:
    # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) so that they
    # are evaluated against the same rules as the IPv4 address they encode,
    # instead of being mistaken for an ordinary IPv6 address.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


_SCHEME_DEFAULT_PORTS = {'http': 80, 'https': 443}


@dataclass(frozen=True)
class NetworkAdmissionPolicy:
    """
    Declarative network admission policy applied by the built-in fetcher
    backends.

    All fields are optional; a policy with no name or IP allow-list entries
    permits only requests to public, non-loopback addresses on the default
    HTTP(S) ports.

    :param allowed_schemes:
        URL schemes that may be used, lower-cased (defaults to
        ``('http', 'https')``).
    :param allowed_hosts:
        Exact host names (case-insensitive) that are always admitted at the
        name-based stage. Names are matched after stripping a trailing dot
        and lower-casing. Use :attr:`allowed_host_suffixes` for subdomain
        matching.
    :param allowed_host_suffixes:
        DNS suffixes (case-insensitive) that are admitted, including the
        suffix itself. For example, ``'.example.com'`` admits both
        ``example.com`` and ``foo.example.com``.
    :param denied_hosts:
        Host names that are always refused even if they match an allow-list
        entry.
    :param allowed_ip_networks:
        IP networks (strings or :mod:`ipaddress` network objects) whose
        addresses may be contacted. If empty, no explicit IP allow-list is
        applied at the network level (the built-in denials still hold).
    :param denied_ip_networks:
        IP networks whose addresses are always refused. These take
        precedence over :attr:`allowed_ip_networks`.
    :param allow_private_addresses:
        Admit RFC 1918 / unique-local / link-local addresses. Defaults to
        ``False``.
    :param allow_loopback:
        Admit loopback addresses and ``localhost`` aliases. Defaults to
        ``False``.
    :param allow_non_global_addresses:
        Admit any non-global address (private, loopback, reserved,
        multicast, unspecified, ...). Defaults to ``False``.
    :param allowed_ports:
        Explicit set of permitted destination ports. ``None`` (the default)
        restricts requests to the canonical port for each allowed scheme
        (80 for HTTP, 443 for HTTPS); an explicit tuple further limits or
        extends the permitted ports. An empty tuple forbids every port.
    :param allow_userinfo:
        Permit ``user:pass@`` style userinfo components in URLs. Defaults to
        ``False``.
    :param max_redirects:
        Maximum number of redirects that may be followed. Each redirect
        target is re-evaluated against the full policy.
    :param max_response_size:
        Maximum accepted response body size in bytes. Bodies exceeding this
        limit are aborted.
    :param per_request_timeout:
        Timeout in seconds applied to each individual HTTP request.
    :param resolver:
        Custom :class:`HostResolver`. Defaults to :class:`DefaultHostResolver`.
    """

    allowed_schemes: Tuple[str, ...] = ('http', 'https')
    allowed_hosts: Tuple[str, ...] = ()
    allowed_host_suffixes: Tuple[str, ...] = ()
    denied_hosts: Tuple[str, ...] = ()
    allowed_ip_networks: Tuple[Union[ipaddress.IPv4Network, ipaddress.IPv6Network], ...] = ()
    denied_ip_networks: Tuple[Union[ipaddress.IPv4Network, ipaddress.IPv6Network], ...] = ()
    allow_private_addresses: bool = False
    allow_loopback: bool = False
    allow_non_global_addresses: bool = False
    allowed_ports: Optional[Tuple[int, ...]] = None
    allow_userinfo: bool = False
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE
    per_request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    resolver: Optional[HostResolver] = field(default=None, compare=False)

    def __post_init__(self):
        if self.max_redirects < 0:
            raise ValueError("max_redirects must be non-negative")
        if self.max_response_size <= 0:
            raise ValueError("max_response_size must be positive")
        if self.per_request_timeout <= 0:
            raise ValueError("per_request_timeout must be positive")
        if not self.allowed_schemes:
            raise ValueError("at least one allowed scheme is required")
        if self.allowed_ports is not None:
            for port in self.allowed_ports:
                if not 1 <= port <= 65535:
                    raise ValueError(f"invalid port {port}")
        # Frozen dataclass + normalisation: use object.__setattr__ once.
        object.__setattr__(
            self,
            'allowed_schemes',
            tuple(s.casefold() for s in self.allowed_schemes),
        )
        object.__setattr__(
            self,
            'allowed_hosts',
            tuple(_normalise_hostname(h) for h in self.allowed_hosts),
        )
        object.__setattr__(
            self,
            'allowed_host_suffixes',
            tuple(_normalise_suffix(s) for s in self.allowed_host_suffixes),
        )
        object.__setattr__(
            self,
            'denied_hosts',
            tuple(_normalise_hostname(h) for h in self.denied_hosts),
        )
        object.__setattr__(
            self,
            'allowed_ip_networks',
            tuple(_as_network(n) for n in self.allowed_ip_networks),
        )
        object.__setattr__(
            self,
            'denied_ip_networks',
            tuple(_as_network(n) for n in self.denied_ip_networks),
        )
        if self.resolver is None:
            object.__setattr__(self, 'resolver', DefaultHostResolver())

    # ------------------------------------------------------------------
    # URL-level admission
    # ------------------------------------------------------------------

    def check_url(self, url: str) -> Tuple[str, str, int]:
        """
        Validate everything that can be decided without DNS: scheme,
        userinfo, hostname shape and port.

        :return:
            A ``(scheme, hostname, port)`` triple for use by the transport.
        :raises NetworkPolicyError:
            if the URL is refused.
        """

        try:
            parts = urlsplit(url)
        except ValueError as e:
            raise NetworkPolicyError(
                f"URL {url!r} could not be parsed: {e}",
                reason=PolicyDenialReason.INVALID_URL,
                url=url,
            ) from e

        scheme = parts.scheme.casefold()
        if scheme not in self.allowed_schemes:
            raise NetworkPolicyError(
                f"Scheme {parts.scheme!r} is not permitted; allowed schemes "
                f"are {sorted(self.allowed_schemes)}",
                reason=PolicyDenialReason.UNSUPPORTED_SCHEME,
                url=url,
            )

        if (parts.username is not None or parts.password is not None) and not \
                self.allow_userinfo:
            raise NetworkPolicyError(
                "URLs containing userinfo (user:pass@host) are not permitted",
                reason=PolicyDenialReason.USERINFO_NOT_ALLOWED,
                url=url,
            )

        raw_host = parts.hostname
        if raw_host is None or not raw_host:
            raise NetworkPolicyError(
                "URL does not contain a valid host component",
                reason=PolicyDenialReason.INVALID_HOST,
                url=url,
            )

        port = parts.port
        if port is None:
            try:
                port = _SCHEME_DEFAULT_PORTS[scheme]
            except KeyError:  # pragma: nocover
                raise NetworkPolicyError(
                    f"No default port known for scheme {scheme!r}",
                    reason=PolicyDenialReason.PORT_NOT_ALLOWED,
                    url=url,
                ) from None
        if self.allowed_ports is None:
            # Default: only the canonical port for the URL's scheme.
            if port != _SCHEME_DEFAULT_PORTS[scheme]:
                raise NetworkPolicyError(
                    f"Port {port} is not the default port for scheme "
                    f"{scheme!r}; supply allowed_ports to permit additional "
                    f"ports",
                    reason=PolicyDenialReason.PORT_NOT_ALLOWED,
                    url=url,
                )
        elif port not in self.allowed_ports:
            raise NetworkPolicyError(
                f"Port {port} is not permitted; allowed ports are "
                f"{sorted(self.allowed_ports)}",
                reason=PolicyDenialReason.PORT_NOT_ALLOWED,
                url=url,
            )

        hostname = self._check_hostname(raw_host, url)
        return scheme, hostname, port

    def _check_hostname(self, raw_host: str, url: str) -> str:
        # Literal IP address in the URL?
        try:
            literal = ipaddress.ip_address(raw_host)
        except ValueError:
            literal = None
        if literal is not None:
            canonical = _canonicalise_ip(literal)
            self.check_ip_address(canonical, url=url)
            return str(canonical)

        # Bracketed non-IP hosts (e.g. "[example.com]") or other malformed
        # shapes are rejected up front.
        if raw_host.startswith('[') or raw_host.endswith(']'):
            raise NetworkPolicyError(
                f"Host {raw_host!r} is not a valid bracketed IPv6 literal",
                reason=PolicyDenialReason.INVALID_HOST,
                url=url,
            )
        hostname = _normalise_hostname(raw_host)
        if not hostname or any(
            not label for label in hostname.split('.')
        ):
            raise NetworkPolicyError(
                f"Host {raw_host!r} is not a valid hostname",
                reason=PolicyDenialReason.INVALID_HOST,
                url=url,
            )

        if hostname in self.denied_hosts:
            raise NetworkPolicyError(
                f"Host {hostname!r} is explicitly denied",
                reason=PolicyDenialReason.HOST_NOT_ALLOWED,
                url=url,
            )

        is_localhost_alias = (
            hostname in _LOCALHOST_ALIASES
            or hostname.endswith('.localhost')
        )
        if is_localhost_alias and not self.allow_loopback:
            raise NetworkPolicyError(
                f"Host {hostname!r} is a localhost alias and loopback access "
                f"is disabled",
                reason=PolicyDenialReason.HOST_NOT_ALLOWED,
                url=url,
            )

        name_allowed = (
            hostname in self.allowed_hosts
            or self._matches_suffix(hostname)
        )
        if not name_allowed and not self.allowed_hosts and not \
                self.allowed_host_suffixes:
            # No name allow-list configured: fall through to IP-level
            # enforcement after resolution.
            return hostname
        if not name_allowed:
            raise NetworkPolicyError(
                f"Host {hostname!r} does not match any allowed host or "
                f"suffix",
                reason=PolicyDenialReason.HOST_NOT_ALLOWED,
                url=url,
            )
        return hostname

    def _matches_suffix(self, hostname: str) -> bool:
        for suffix in self.allowed_host_suffixes:
            if hostname == suffix or hostname.endswith('.' + suffix):
                return True
        return False

    # ------------------------------------------------------------------
    # IP-level admission
    # ------------------------------------------------------------------

    def check_ip_address(
        self,
        address: Union[ipaddress.IPv4Address, ipaddress.IPv6Address, str],
        *,
        url: Optional[str] = None,
    ) -> Union[ipaddress.IPv4Address, ipaddress.IPv6Address]:
        """
        Apply the IP-level rules to a single resolved address.

        Explicit deny entries win over everything else; the built-in
        non-global-address denials win over the explicit allow-list.
        """

        if isinstance(address, str):
            try:
                addr = _canonicalise_ip(ipaddress.ip_address(address))
            except ValueError as e:
                raise NetworkPolicyError(
                    f"{address!r} is not a valid IP address",
                    reason=PolicyDenialReason.INVALID_HOST,
                    url=url,
                ) from e
        else:
            addr = _canonicalise_ip(address)

        if any(addr in network for network in self.denied_ip_networks):
            raise NetworkPolicyError(
                f"Resolved address {addr} is explicitly denied",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )

        explicit_allow = bool(self.allowed_ip_networks) and any(
            addr in network for network in self.allowed_ip_networks
        )
        if self.allowed_ip_networks and not explicit_allow:
            raise NetworkPolicyError(
                f"Resolved address {addr} is not part of any allowed IP "
                f"network",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )

        if addr.is_loopback and not (
            self.allow_loopback or explicit_allow
        ):
            raise NetworkPolicyError(
                f"Resolved address {addr} is a loopback address",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        if addr.is_multicast:
            raise NetworkPolicyError(
                f"Resolved address {addr} is a multicast address",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        if addr.is_unspecified:
            raise NetworkPolicyError(
                f"Resolved address {addr} is an unspecified address",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        if addr.is_reserved:
            raise NetworkPolicyError(
                f"Resolved address {addr} is a reserved address",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        if (addr.is_private or addr.is_link_local) and not (
            self.allow_private_addresses
            or self.allow_non_global_addresses
            or explicit_allow
        ):
            raise NetworkPolicyError(
                f"Resolved address {addr} is a private or link-local address",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        if not addr.is_global and not (
            self.allow_non_global_addresses
            or self.allow_private_addresses
            or explicit_allow
        ):
            raise NetworkPolicyError(
                f"Resolved address {addr} is not a globally routable address",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        return addr

    def resolve_and_check(
        self, hostname: str, port: int, *, url: Optional[str] = None
    ) -> Tuple[
        Union[ipaddress.IPv4Address, ipaddress.IPv6Address], ...
    ]:
        """
        Resolve ``hostname`` through the configured resolver and return the
        subset of resolved addresses that pass the IP-level rules.

        Literal IP hosts bypass DNS. A :class:`NetworkPolicyError` is raised
        when no admissible address remains, so callers never open a socket
        without an approved address.
        """

        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None:
            return (self.check_ip_address(literal, url=url),)

        assert self.resolver is not None
        candidates = self.resolver.resolve_all(hostname, port)
        admitted: List[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]] = []
        for candidate in candidates:
            try:
                admitted.append(self.check_ip_address(candidate, url=url))
            except NetworkPolicyError as e:
                logger.warning(
                    "Rejecting resolved address %s for %s: %s",
                    candidate, url or hostname, e,
                )
        if not admitted:
            raise NetworkPolicyError(
                f"None of the addresses resolved for host {hostname!r} are "
                f"admissible under this policy",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        return tuple(admitted)

    async def resolve_and_check_async(
        self, hostname: str, port: int, *, url: Optional[str] = None
    ):
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None:
            return (self.check_ip_address(literal, url=url),)

        assert self.resolver is not None
        candidates = await self.resolver.resolve_all_async(hostname, port)
        admitted = []
        for candidate in candidates:
            try:
                admitted.append(self.check_ip_address(candidate, url=url))
            except NetworkPolicyError as e:
                logger.warning(
                    "Rejecting resolved address %s for %s: %s",
                    candidate, url or hostname, e,
                )
        if not admitted:
            raise NetworkPolicyError(
                f"None of the addresses resolved for host {hostname!r} are "
                f"admissible under this policy",
                reason=PolicyDenialReason.IP_NOT_ALLOWED,
                url=url,
            )
        return tuple(admitted)

    def check_redirect_count(self, redirect_count: int, *, url: str) -> None:
        if redirect_count > self.max_redirects:
            raise NetworkPolicyError(
                f"Redirect limit of {self.max_redirects} exceeded while "
                f"fetching {url}",
                reason=PolicyDenialReason.TOO_MANY_REDIRECTS,
                url=url,
                redirect_count=redirect_count,
            )

    def check_response_size(self, size: int, *, url: str) -> None:
        if size > self.max_response_size:
            raise NetworkPolicyError(
                f"Response body of {size} bytes from {url} exceeds the "
                f"maximum allowed size of {self.max_response_size} bytes",
                reason=PolicyDenialReason.RESPONSE_TOO_LARGE,
                url=url,
            )


def _normalise_hostname(hostname: str) -> str:
    # urlsplit already strips brackets from IPv6 literals; here we deal with
    # ordinary DNS names. IDNA handling is delegated to the HTTP libraries at
    # connect time, we only need a canonical comparison form.
    host = hostname.strip().rstrip('.').casefold()
    return host


def _normalise_suffix(suffix: str) -> str:
    normalised = _normalise_hostname(suffix)
    return normalised.lstrip('.')


def _as_network(
    value: Union[str, ipaddress.IPv4Network, ipaddress.IPv6Network],
) -> Union[ipaddress.IPv4Network, ipaddress.IPv6Network]:
    if isinstance(value, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
        return value
    return ipaddress.ip_network(value, strict=False)
