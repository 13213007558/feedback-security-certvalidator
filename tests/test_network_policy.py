"""
Tests for the optional network admission policy and its enforcement in both
built-in fetcher backends.

No test in this module contacts a real network address: name resolution is
served by an in-memory fake resolver, and the requests transport is exercised
with a fake socket, while the aiohttp transport is driven through the shared
guarded engine with a stubbed session.
"""

import ipaddress
import socket

import pytest

from pyhanko_certvalidator.errors import (
    CertificateFetchError,
    CRLFetchError,
    OCSPFetchError,
)
from pyhanko_certvalidator.fetchers import (
    NetworkAdmissionPolicy,
    NetworkPolicyError,
    PolicyDenialReason,
)
from pyhanko_certvalidator.fetchers._guarded_http import (
    GuardedResponse,
    GuardedTransport,
    guarded_http_request,
)
from pyhanko_certvalidator.fetchers.aiohttp_fetchers.cert_fetch_client import (
    AIOHttpCertificateFetcher,
)
from pyhanko_certvalidator.fetchers.aiohttp_fetchers.policy_transport import (
    AIOHttpGuardedTransport,
    PolicyAwareResolver,
)
from pyhanko_certvalidator.fetchers.aiohttp_fetchers.util import LazySession
from pyhanko_certvalidator.fetchers.network_policy import HostResolver
from pyhanko_certvalidator.fetchers.requests_fetchers.cert_fetch_client import (
    RequestsCertificateFetcher,
)
from pyhanko_certvalidator.fetchers.requests_fetchers.policy_transport import (
    PolicyAwareRequestsSession,
)
from pyhanko_certvalidator.fetchers.requests_fetchers.util import (
    RequestsFetcherMixin,
)

PUBLIC_IPV4 = '93.184.216.34'
PUBLIC_IPV4_B = '93.184.216.35'


class FakeResolver(HostResolver):
    """Deterministic in-memory resolver; never touches the network."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)
        self.calls = []

    def resolve_all(self, hostname, port):
        self.calls.append((hostname, port))
        if hostname not in self._mapping:
            raise NetworkPolicyError(
                f"no fake record for {hostname}",
                reason=PolicyDenialReason.DNS_ERROR,
            )
        return tuple(ipaddress.ip_address(a) for a in self._mapping[hostname])


class ScriptedTransport(GuardedTransport):
    """Transport returning scripted responses and recording every call."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.contacted = []

    async def send(
        self, *, method, url, headers, data, timeout, max_response_size
    ):
        self.contacted.append((method, url))
        status, headers_out, body, location = self.routes[url]
        headers_out = dict(headers_out)
        if location is not None:
            headers_out['Location'] = location
        return GuardedResponse(status, headers_out, body, url)


def make_policy(**kwargs):
    resolver = kwargs.pop('resolver', None)
    kwargs.setdefault('allowed_hosts', ('crt.example.com',))
    return NetworkAdmissionPolicy(resolver=resolver, **kwargs)


# ---------------------------------------------------------------------------
# URL-level rules
# ---------------------------------------------------------------------------

def test_default_policy_rejects_localhost_and_private():
    policy = make_policy(resolver=FakeResolver({}))
    for host in ('localhost', 'ip6-localhost', 'foo.localhost'):
        with pytest.raises(NetworkPolicyError) as exc:
            policy.check_url(f'http://{host}/path')
        assert exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED


def test_localhost_suffix_is_alias():
    policy = make_policy(
        allowed_host_suffixes=('.localhost',),
        resolver=FakeResolver({}),
    )
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://anything.localhost/')
    assert exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED


def test_userinfo_is_rejected_by_default():
    policy = make_policy(resolver=FakeResolver({}))
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://user:secret@crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.USERINFO_NOT_ALLOWED


def test_userinfo_can_be_explicitly_permitted():
    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = make_policy(allow_userinfo=True, resolver=resolver)
    scheme, host, port = policy.check_url(
        'http://user:secret@crt.example.com/'
    )
    assert (host, port) == ('crt.example.com', 80)


def test_unsupported_scheme_rejected():
    policy = make_policy(resolver=FakeResolver({}))
    for url in ('file:///etc/passwd', 'ftp://crt.example.com/',
                'gopher://crt.example.com/'):
        with pytest.raises(NetworkPolicyError) as exc:
            policy.check_url(url)
        assert exc.value.reason == PolicyDenialReason.UNSUPPORTED_SCHEME


def test_invalid_host_forms_rejected():
    policy = make_policy(resolver=FakeResolver({}))
    for url in (
        'http://[not-an-ip]/',
        'http:///',
        'http://bad..example.com/',
    ):
        with pytest.raises(NetworkPolicyError) as exc:
            policy.check_url(url)
        assert exc.value.reason in (
            PolicyDenialReason.INVALID_HOST,
            PolicyDenialReason.INVALID_URL,
        )


def test_port_rules_including_default_ports():
    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = make_policy(
        allowed_ports=(443,), resolver=resolver,
    )
    policy.check_url('https://crt.example.com/')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.PORT_NOT_ALLOWED
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('https://crt.example.com:8443/')
    assert exc.value.reason == PolicyDenialReason.PORT_NOT_ALLOWED


def test_default_ports_only_when_allowed_ports_unset():
    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = make_policy(resolver=resolver)
    policy.check_url('https://crt.example.com/')
    policy.check_url('http://crt.example.com/')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('https://crt.example.com:8443/')
    assert exc.value.reason == PolicyDenialReason.PORT_NOT_ALLOWED


def test_explicit_allowed_ports_can_extend_beyond_defaults():
    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = make_policy(allowed_ports=(443, 8443), resolver=resolver)
    policy.check_url('https://crt.example.com:8443/')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.PORT_NOT_ALLOWED


def test_empty_allowed_ports_forbids_every_port():
    policy = make_policy(allowed_ports=(), resolver=FakeResolver({}))
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('https://crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.PORT_NOT_ALLOWED


def test_host_allowlist_and_suffix_matching():
    resolver = FakeResolver({
        'crt.example.com': [PUBLIC_IPV4],
        'sub.crt.example.com': [PUBLIC_IPV4],
        'evil.com': [PUBLIC_IPV4],
    })
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        allowed_host_suffixes=('.trusted.example.org',),
        resolver=resolver,
    )
    policy.check_url('http://crt.example.com/')
    policy.check_url('http://sub.trusted.example.org/')
    policy.check_url('http://trusted.example.org/')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://sub.crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://evil.com/')
    assert exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED


def test_denied_host_takes_precedence_over_allowlist():
    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = make_policy(
        denied_hosts=('crt.example.com',), resolver=resolver,
    )
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED


# ---------------------------------------------------------------------------
# IP-level rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'address',
    [
        '127.0.0.1',
        '10.0.0.1',
        '172.16.4.9',
        '192.168.1.1',
        '169.254.169.254',
        '0.0.0.0',
        '224.0.0.1',
        '::1',
        'fe80::1',
        'fc00::1',
        '::ffff:127.0.0.1',
        '::ffff:10.0.0.1',
    ],
)
def test_non_global_and_mapped_addresses_rejected(address):
    policy = make_policy(resolver=FakeResolver({}))
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_ip_address(address)
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED


def test_ipv4_mapped_address_is_evaluated_as_embedded_ipv4():
    policy = make_policy(resolver=FakeResolver({}))
    # A mapped *public* address unwraps to its IPv4 form and is admitted.
    result = policy.check_ip_address('::ffff:93.184.216.34')
    assert isinstance(result, ipaddress.IPv4Address)
    assert str(result) == PUBLIC_IPV4


def test_explicit_ip_allow_and_deny_networks():
    policy = NetworkAdmissionPolicy(
        allowed_ip_networks=('10.5.0.0/16',),
        denied_ip_networks=('10.5.9.0/24',),
        allow_private_addresses=True,
        resolver=FakeResolver({}),
    )
    policy.check_ip_address('10.5.3.3')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_ip_address('10.5.9.9')
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_ip_address('10.6.0.1')
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED


def test_private_addresses_can_be_enabled():
    resolver = FakeResolver({'intranet.example': ['10.20.30.40']})
    policy = make_policy(
        allowed_hosts=('intranet.example',),
        allow_private_addresses=True,
        resolver=resolver,
    )
    addresses = policy.resolve_and_check('intranet.example', 80)
    assert addresses == (ipaddress.ip_address('10.20.30.40'),)


def test_resolve_filters_and_rebinds():
    resolver = FakeResolver({
        'rebind.example.com': [PUBLIC_IPV4, '169.254.169.254'],
        'all-internal.example.com': ['10.0.0.1', '169.254.169.254'],
    })
    policy = make_policy(
        allowed_hosts=('rebind.example.com', 'all-internal.example.com'),
        resolver=resolver,
    )
    admitted = policy.resolve_and_check('rebind.example.com', 443)
    assert admitted == (ipaddress.ip_address(PUBLIC_IPV4),)
    with pytest.raises(NetworkPolicyError) as exc:
        policy.resolve_and_check('all-internal.example.com', 443)
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED


def test_literal_ip_url_is_checked_directly():
    policy = make_policy(resolver=FakeResolver({}))
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_url('http://169.254.169.254/latest/meta-data/')
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED


def test_response_size_limit():
    policy = make_policy(
        max_response_size=10, resolver=FakeResolver({})
    )
    policy.check_response_size(10, url='http://crt.example.com/')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_response_size(11, url='http://crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.RESPONSE_TOO_LARGE


def test_redirect_count_limit():
    policy = make_policy(
        max_redirects=2, resolver=FakeResolver({})
    )
    policy.check_redirect_count(2, url='http://crt.example.com/')
    with pytest.raises(NetworkPolicyError) as exc:
        policy.check_redirect_count(3, url='http://crt.example.com/')
    assert exc.value.reason == PolicyDenialReason.TOO_MANY_REDIRECTS


def test_invalid_constructor_arguments():
    with pytest.raises(ValueError):
        NetworkAdmissionPolicy(max_redirects=-1)
    with pytest.raises(ValueError):
        NetworkAdmissionPolicy(max_response_size=0)
    with pytest.raises(ValueError):
        NetworkAdmissionPolicy(allowed_ports=(0,))
    with pytest.raises(ValueError):
        NetworkAdmissionPolicy(allowed_schemes=())


# ---------------------------------------------------------------------------
# Shared guarded engine (backend-agnostic)
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_policy():
    resolver = FakeResolver({
        'crt.example.com': [PUBLIC_IPV4],
        'intranet.internal': ['10.0.0.7'],
    })
    return NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        allowed_host_suffixes=('.example.com',),
        denied_hosts=('forbidden.example.com',),
        resolver=resolver,
        max_redirects=2,
        max_response_size=100,
    )


@pytest.mark.asyncio
async def test_engine_allows_legitimate_download(engine_policy):
    transport = ScriptedTransport({
        'http://crt.example.com/aia.cer': (
            200, {'Content-Type': 'application/pkix-cert'}, b'cert', None),
    })
    response = await guarded_http_request(
        transport, engine_policy, 'http://crt.example.com/aia.cer'
    )
    assert response.content == b'cert'
    assert transport.contacted == [
        ('GET', 'http://crt.example.com/aia.cer')
    ]


@pytest.mark.asyncio
async def test_engine_rechecks_every_redirect(engine_policy):
    transport = ScriptedTransport({
        'http://crt.example.com/start': (
            302, {}, b'', 'http://intranet.internal/secret'),
    })
    with pytest.raises(NetworkPolicyError) as exc:
        await guarded_http_request(
            transport, engine_policy, 'http://crt.example.com/start'
        )
    assert exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED
    # The internal peer is never contacted.
    assert transport.contacted == [
        ('GET', 'http://crt.example.com/start')
    ]


@pytest.mark.asyncio
async def test_engine_redirect_to_internal_ip_blocked_before_connect(
    engine_policy,
):
    transport = ScriptedTransport({
        'http://crt.example.com/start': (
            302, {}, b'', 'http://169.254.169.254/metadata'),
    })
    with pytest.raises(NetworkPolicyError) as exc:
        await guarded_http_request(
            transport, engine_policy, 'http://crt.example.com/start'
        )
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    assert len(transport.contacted) == 1


@pytest.mark.asyncio
async def test_engine_follows_allowed_redirect_chain(engine_policy):
    transport = ScriptedTransport({
        'http://crt.example.com/r1': (
            302, {}, b'', 'http://crt.example.com/r2'),
        'http://crt.example.com/r2': (
            200, {}, b'final', None),
    })
    response = await guarded_http_request(
        transport, engine_policy, 'http://crt.example.com/r1'
    )
    assert response.status_code == 200
    assert response.content == b'final'
    assert len(transport.contacted) == 2


@pytest.mark.asyncio
async def test_engine_enforces_redirect_limit(engine_policy):
    transport = ScriptedTransport({
        'http://crt.example.com/r': (
            302, {}, b'', 'http://crt.example.com/r2'),
        'http://crt.example.com/r2': (
            302, {}, b'', 'http://crt.example.com/r3'),
        'http://crt.example.com/r3': (
            302, {}, b'', 'http://crt.example.com/r4'),
    })
    with pytest.raises(NetworkPolicyError) as exc:
        await guarded_http_request(
            transport, engine_policy, 'http://crt.example.com/r'
        )
    assert exc.value.reason == PolicyDenialReason.TOO_MANY_REDIRECTS


@pytest.mark.asyncio
async def test_engine_denial_never_calls_transport(engine_policy):
    transport = ScriptedTransport({})
    with pytest.raises(NetworkPolicyError):
        await guarded_http_request(
            transport, engine_policy, 'http://intranet.internal/x'
        )
    assert transport.contacted == []


@pytest.mark.asyncio
async def test_engine_post_redirect_switches_to_get(engine_policy):
    transport = ScriptedTransport({
        'http://crt.example.com/post': (
            303, {}, b'', 'http://crt.example.com/result'),
        'http://crt.example.com/result': (200, {}, b'ok', None),
    })
    response = await guarded_http_request(
        transport,
        engine_policy,
        'http://crt.example.com/post',
        method='POST',
        data=b'payload',
    )
    assert response.content == b'ok'
    methods = [method for method, _ in transport.contacted]
    assert methods == ['POST', 'GET']


# ---------------------------------------------------------------------------
# requests transport: connection-level enforcement with a fake socket
# ---------------------------------------------------------------------------

class FakeSocket(socket.socket):
    """Real socket subclass that records peers but never connects."""

    connections = []

    def connect(self, address):
        FakeSocket.connections.append(address)

    def connect_ex(self, address):
        FakeSocket.connections.append(address)
        return 0


@pytest.fixture
def requests_routes():
    return {
        'http://crt.example.com/good.cer': (
            200,
            {'Content-Type': 'application/pkix-cert'},
            b'payload',
            None,
        ),
        'http://crt.example.com/redir-internal': (
            302,
            {'Location': 'http://10.0.0.9/secret'},
            b'',
            'http://10.0.0.9/secret',
        ),
        'http://crt.example.com/redir-ok': (
            302,
            {'Location': 'http://crt.example.com/good.cer'},
            b'',
            'http://crt.example.com/good.cer',
        ),
    }


class _FakeRequestsResponse:
    def __init__(self, status_code, headers, body, url):
        self.status_code = status_code
        self.headers = headers
        self._body = body
        self.url = url

    def iter_content(self, chunk_size=65536):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index:index + chunk_size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def stubbed_requests_session(monkeypatch, requests_routes):
    """Replace the blocking session.request with an in-memory scripted one."""

    def fake_request(session, method, url, **kwargs):
        key = url
        assert key in requests_routes, (
            f'unapproved URL reached requests transport: {key}'
        )
        status, headers, body, _ = requests_routes[key]
        return _FakeRequestsResponse(status, dict(headers), body, url)

    monkeypatch.setattr(
        PolicyAwareRequestsSession, 'request', fake_request
    )
    return requests_routes


def _requests_policy():
    resolver = FakeResolver({
        'crt.example.com': [PUBLIC_IPV4],
        '10.0.0.9': ['10.0.0.9'],
    })
    return NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',), resolver=resolver
    )


def test_requests_session_ignores_ambient_proxy():
    policy = _requests_policy()
    session = PolicyAwareRequestsSession(policy)
    assert session.trust_env is False
    adapter = session.get_adapter('http://crt.example.com/')
    assert (
        adapter.poolmanager.pool_classes_by_scheme['http']
        .ConnectionCls.__name__
        == '_PolicyHTTPConnection'
    )


def test_requests_connection_helper_uses_only_approved_peers(monkeypatch):
    from pyhanko_certvalidator.fetchers.requests_fetchers.policy_transport import (
        _connect_to_approved_address,
    )

    FakeSocket.connections = []
    monkeypatch.setattr(socket, 'socket', FakeSocket)
    policy = _requests_policy()

    class _DummyConnection:
        _dns_host = 'crt.example.com'
        port = 80

    sock = _connect_to_approved_address(
        _DummyConnection(), policy, 'crt.example.com', 80, 5
    )
    assert FakeSocket.connections == [(PUBLIC_IPV4, 80)]
    sock.close()

    with pytest.raises(NetworkPolicyError) as exc:
        _connect_to_approved_address(
            _DummyConnection(), policy, '10.0.0.9', 80, 5
        )
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    # No socket was ever opened to the internal peer.
    assert all(peer[0] != '10.0.0.9' for peer in FakeSocket.connections)


@pytest.mark.asyncio
async def test_requests_legitimate_fetch_uses_approved_peer(
    stubbed_requests_session,
):
    probe = RequestsFetcherMixin(network_policy=_requests_policy())
    response = await probe._get(
        'http://crt.example.com/good.cer',
        acceptable_content_types=('application/pkix-cert',),
    )
    assert response.content == b'payload'
    await probe._close_guarded_transport()


@pytest.mark.asyncio
async def test_requests_internal_redirect_never_reaches_transport(
    stubbed_requests_session,
):
    probe = RequestsFetcherMixin(network_policy=_requests_policy())
    with pytest.raises(NetworkPolicyError) as exc:
        await probe._get(
            'http://crt.example.com/redir-internal',
            acceptable_content_types=('application/pkix-cert',),
        )
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    await probe._close_guarded_transport()


@pytest.mark.asyncio
async def test_requests_allowed_redirect_reuses_policy_checks(
    stubbed_requests_session,
):
    probe = RequestsFetcherMixin(network_policy=_requests_policy())
    response = await probe._get(
        'http://crt.example.com/redir-ok',
        acceptable_content_types=('application/pkix-cert',),
    )
    assert response.content == b'payload'
    await probe._close_guarded_transport()


@pytest.mark.asyncio
async def test_requests_oversized_body_aborted(monkeypatch):
    routes = {
        'http://crt.example.com/big': (
            200, {'Content-Type': 'application/pkix-cert'}, b'x' * 500,
            None,
        ),
    }

    def fake_request(session, method, url, **kwargs):
        status, headers, body, _ = routes[url]
        return _FakeRequestsResponse(status, headers, body, url)

    monkeypatch.setattr(
        PolicyAwareRequestsSession, 'request', fake_request
    )
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        resolver=FakeResolver({'crt.example.com': [PUBLIC_IPV4]}),
        max_response_size=100,
    )
    probe = RequestsFetcherMixin(network_policy=policy)
    with pytest.raises(NetworkPolicyError) as exc:
        await probe._get(
            'http://crt.example.com/big',
            acceptable_content_types=('application/pkix-cert',),
        )
    assert exc.value.reason == PolicyDenialReason.RESPONSE_TOO_LARGE
    await probe._close_guarded_transport()


# ---------------------------------------------------------------------------
# aiohttp transport: policy-aware resolver + guarded engine
# ---------------------------------------------------------------------------

class _FakeContent:
    def __init__(self, body):
        self._body = body

    async def iter_chunked(self, size):
        for index in range(0, len(self._body), size):
            yield self._body[index:index + size]


class _FakeResponse:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.content = _FakeContent(body)
        self.url = 'http://stub/'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()

    def close(self):
        pass


def _install_aiohttp_routes(session, routes):
    def fake_request(method, url, **kwargs):
        key = str(url)
        assert key in routes, f'unapproved URL reached aiohttp transport: {key}'
        status, headers, body = routes[key]
        response = _FakeResponse(status, headers, body)

        class _ContextManager:
            async def __aenter__(self_inner):
                return response

            async def __aexit__(self_inner, *args):
                response.close()

        return _ContextManager()

    session.request = fake_request


@pytest.mark.asyncio
async def test_aiohttp_session_uses_policy_aware_connector():
    policy = _requests_policy()
    transport = AIOHttpGuardedTransport(policy)
    session = await transport._get_session()
    assert isinstance(session.connector._resolver, PolicyAwareResolver)
    await transport.close()


@pytest.mark.asyncio
async def test_policy_aware_resolver_filters_addresses():
    resolver = FakeResolver({
        'crt.example.com': [PUBLIC_IPV4, PUBLIC_IPV4_B],
        'evil.example.com': ['169.254.169.254'],
    })
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com', 'evil.example.com'),
        resolver=resolver,
    )
    aware = PolicyAwareResolver(policy)
    results = await aware.resolve('crt.example.com', 443)
    assert {item['host'] for item in results} == {
        PUBLIC_IPV4, PUBLIC_IPV4_B
    }
    with pytest.raises(NetworkPolicyError) as exc:
        await aware.resolve('evil.example.com', 443)
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    await aware.close()


@pytest.mark.asyncio
async def test_aiohttp_and_requests_take_same_decision():
    resolver = FakeResolver({
        'crt.example.com': [PUBLIC_IPV4],
        'intranet.internal': ['10.0.0.7'],
    })
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        resolver=resolver,
        max_redirects=1,
        max_response_size=100,
    )

    routes = {
        'http://crt.example.com/redir': (
            302, {'Location': 'http://intranet.internal/secret'}, b'',
        ),
    }
    aio_transport = AIOHttpGuardedTransport(policy)
    session = await aio_transport._get_session()
    _install_aiohttp_routes(session, routes)
    with pytest.raises(NetworkPolicyError) as aio_exc:
        await guarded_http_request(
            aio_transport, policy, 'http://crt.example.com/redir'
        )
    await aio_transport.close()

    scripted = ScriptedTransport({
        'http://crt.example.com/redir': (
            302, {}, b'', 'http://intranet.internal/secret'),
    })
    with pytest.raises(NetworkPolicyError) as shared_exc:
        await guarded_http_request(
            scripted, policy, 'http://crt.example.com/redir'
        )
    assert aio_exc.value.reason == shared_exc.value.reason
    assert aio_exc.value.reason == PolicyDenialReason.HOST_NOT_ALLOWED


@pytest.mark.asyncio
async def test_aiohttp_oversized_body_aborted():
    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        resolver=resolver,
        max_response_size=100,
    )
    transport = AIOHttpGuardedTransport(policy)
    session = await transport._get_session()
    _install_aiohttp_routes(session, {
        'http://crt.example.com/big': (200, {}, b'x' * 500),
    })
    with pytest.raises(NetworkPolicyError) as exc:
        await guarded_http_request(
            transport, policy, 'http://crt.example.com/big'
        )
    assert exc.value.reason == PolicyDenialReason.RESPONSE_TOO_LARGE
    await transport.close()


def test_aiohttp_rejects_raw_session_with_policy():
    raw_session = object()
    with pytest.raises(ValueError):
        AIOHttpCertificateFetcher(
            raw_session, network_policy=_requests_policy()
        )


# ---------------------------------------------------------------------------
# Cache boundaries and error propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_isolated_between_policy_instances(
    stubbed_requests_session,
):
    strict = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        resolver=FakeResolver({'crt.example.com': [PUBLIC_IPV4]}),
    )
    broad = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        allow_private_addresses=True,
        resolver=FakeResolver({
            'crt.example.com': [PUBLIC_IPV4],
            '10.0.0.9': ['10.0.0.9'],
        }),
    )
    strict_probe = RequestsFetcherMixin(network_policy=strict)
    broad_probe = RequestsFetcherMixin(network_policy=broad)
    await strict_probe._get(
        'http://crt.example.com/good.cer',
        acceptable_content_types=('application/pkix-cert',),
    )
    # Each fetcher owns its own cache; the broad policy cannot read a result
    # produced under the strict policy, nor vice versa.
    assert strict_probe._iter_results() != broad_probe._iter_results()
    with pytest.raises(KeyError):
        broad_probe.get_results_for_tag('http://crt.example.com/good.cer')
    await strict_probe._close_guarded_transport()
    await broad_probe._close_guarded_transport()


@pytest.mark.asyncio
async def test_denial_is_cached_as_error_not_silently_retried(
    monkeypatch,
):
    calls = []

    def fake_request(session, method, url, **kwargs):
        calls.append(url)
        # Only the first hop is ever reachable; the redirect target would
        # raise KeyError, but the policy must block it before transport.
        if url.endswith('redir-internal'):
            return _FakeRequestsResponse(
                302,
                {'Location': 'http://10.0.0.9/secret'},
                b'',
                url,
            )
        raise AssertionError(f'unapproved URL reached transport: {url}')

    monkeypatch.setattr(
        PolicyAwareRequestsSession, 'request', fake_request
    )
    policy = _requests_policy()
    fetcher = RequestsCertificateFetcher(network_policy=policy)
    for _ in range(2):
        with pytest.raises(NetworkPolicyError):
            await fetcher.fetch_certs(
                'http://crt.example.com/redir-internal', 'certificate'
            )
    # The first hop is contacted once; the second call is served from the
    # cached error without any new network activity.
    assert calls == ['http://crt.example.com/redir-internal']
    await fetcher._close_guarded_transport()


@pytest.mark.asyncio
async def test_policy_error_is_not_wrapped_as_material_missing(
    monkeypatch,
):
    def fake_request(session, method, url, **kwargs):
        if url.endswith('redir-internal'):
            return _FakeRequestsResponse(
                302,
                {'Location': 'http://10.0.0.9/secret'},
                b'',
                url,
            )
        # Ordinary missing material.
        return _FakeRequestsResponse(404, {}, b'', url)

    monkeypatch.setattr(
        PolicyAwareRequestsSession, 'request', fake_request
    )
    policy = _requests_policy()
    fetcher = RequestsCertificateFetcher(network_policy=policy)
    # NetworkPolicyError must propagate unwrapped, rather than being turned
    # into a CertificateFetchError ("material unavailable").
    with pytest.raises(NetworkPolicyError):
        await fetcher.fetch_certs(
            'http://crt.example.com/redir-internal', 'certificate'
        )

    # Sanity check: ordinary HTTP failures still use the fetch error type.
    with pytest.raises(CertificateFetchError):
        await fetcher.fetch_certs(
            'http://crt.example.com/missing', 'certificate'
        )


@pytest.mark.asyncio
async def test_policy_error_not_subclass_of_fetch_errors():
    assert not issubclass(NetworkPolicyError, CertificateFetchError)
    assert not issubclass(NetworkPolicyError, CRLFetchError)
    assert not issubclass(NetworkPolicyError, OCSPFetchError)


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------

def test_backends_thread_policy_through_fetchers():
    from pyhanko_certvalidator.fetchers.aiohttp_fetchers import (
        AIOHttpFetcherBackend,
    )
    from pyhanko_certvalidator.fetchers.requests_fetchers import (
        RequestsFetcherBackend,
    )

    policy = _requests_policy()
    requests_backend = RequestsFetcherBackend(network_policy=policy)
    fetchers = requests_backend.get_fetchers()
    assert fetchers.cert_fetcher.network_policy is policy
    assert fetchers.crl_fetcher.network_policy is policy
    assert fetchers.ocsp_fetcher.network_policy is policy

    aio_backend = AIOHttpFetcherBackend(network_policy=policy)
    aio_fetchers = aio_backend.get_fetchers()
    assert aio_fetchers.cert_fetcher.network_policy is policy
    assert isinstance(aio_backend.session, LazySession)


def test_aiohttp_backend_rejects_policy_with_caller_session():
    from pyhanko_certvalidator.fetchers.aiohttp_fetchers import (
        AIOHttpFetcherBackend,
    )

    class _StandInSession:
        pass

    with pytest.raises(ValueError):
        AIOHttpFetcherBackend(
            session=_StandInSession(),
            network_policy=_requests_policy(),
        )


def test_backend_without_policy_keeps_legacy_behaviour():
    from pyhanko_certvalidator.fetchers.requests_fetchers import (
        RequestsFetcherBackend,
    )

    backend = RequestsFetcherBackend(per_request_timeout=7)
    fetchers = backend.get_fetchers()
    assert fetchers.cert_fetcher.network_policy is None
    assert fetchers.cert_fetcher.per_request_timeout == 7


# ---------------------------------------------------------------------------
# Strict validation: security errors must not masquerade as missing material
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_network_policy_error_passes_through_strict_revocation_layer(
    monkeypatch,
):
    """
    The revocation layer (``_check_revocation``) only catches
    OCSPFetchError/CRLFetchError. Because NetworkPolicyError is neither, a
    policy denial during an OCSP/CRL fetch escapes the strict validation
    machinery instead of being downgraded to "material missing" and
    tolerated.
    """
    from pyhanko_certvalidator.fetchers.requests_fetchers.crl_client import (
        RequestsCRLFetcher,
    )
    from pyhanko_certvalidator.fetchers.requests_fetchers.ocsp_client import (
        RequestsOCSPFetcher,
    )

    def fake_request(session, method, url, **kwargs):
        raise AssertionError('denied URL must never reach transport')

    monkeypatch.setattr(
        PolicyAwareRequestsSession, 'request', fake_request
    )
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        resolver=FakeResolver({'crt.example.com': [PUBLIC_IPV4]}),
    )

    ocsp = RequestsOCSPFetcher(network_policy=policy)
    with pytest.raises(NetworkPolicyError):
        await ocsp._post(
            'http://169.254.169.254/ocsp',
            b'request',
            content_type='application/ocsp-request',
            acceptable_content_types=('application/ocsp-response',),
        )

    crl = RequestsCRLFetcher(network_policy=policy)
    with pytest.raises(NetworkPolicyError):
        await crl._get(
            'http://10.0.0.8/crl.pem',
            acceptable_content_types=('application/pkix-crl',),
        )


@pytest.mark.asyncio
async def test_aiohttp_cert_fetcher_denies_internal_redirect():
    from pyhanko_certvalidator.fetchers.aiohttp_fetchers.policy_transport import (
        AIOHttpGuardedTransport,
    )

    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',),
        resolver=FakeResolver({
            'crt.example.com': [PUBLIC_IPV4],
            '10.0.0.9': ['10.0.0.9'],
        }),
    )
    transport = AIOHttpGuardedTransport(policy)
    session = await transport._get_session()
    _install_aiohttp_routes(session, {
        'http://crt.example.com/redir': (
            302, {'Location': 'http://10.0.0.9/secret'}, b'',
        ),
    })
    # The guarded engine refuses the internal redirect before any certificate
    # parsing happens; NetworkPolicyError is not a ValueError/aiohttp error,
    # so the fetcher's wrapping clause leaves it untouched.
    with pytest.raises(NetworkPolicyError) as exc:
        await guarded_http_request(
            transport, policy, 'http://crt.example.com/redir'
        )
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    await transport.close()


@pytest.mark.parametrize(
    'url,expected_reason',
    [
        ('https://crt.example.com/ok', None),
        ('http://169.254.169.254/meta', PolicyDenialReason.IP_NOT_ALLOWED),
        ('http://localhost/x', PolicyDenialReason.HOST_NOT_ALLOWED),
        ('file:///etc/passwd', PolicyDenialReason.UNSUPPORTED_SCHEME),
        ('https://crt.example.com:9999/x', PolicyDenialReason.PORT_NOT_ALLOWED),
        ('https://user:p@crt.example.com/',
         PolicyDenialReason.USERINFO_NOT_ALLOWED),
    ],
)
def test_requests_and_aiohttp_backends_share_decision(url, expected_reason):
    """
    Both backends construct their fetchers from the same immutable policy
    object and delegate admission to the shared engine, so the URL-level
    decision must be identical regardless of the HTTP library in use.
    """
    from pyhanko_certvalidator.fetchers.aiohttp_fetchers import (
        AIOHttpFetcherBackend,
    )
    from pyhanko_certvalidator.fetchers.requests_fetchers import (
        RequestsFetcherBackend,
    )

    resolver = FakeResolver({'crt.example.com': [PUBLIC_IPV4]})
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('crt.example.com',), resolver=resolver
    )
    requests_fetcher = (
        RequestsFetcherBackend(network_policy=policy)
        .get_fetchers()
        .cert_fetcher
    )
    aio_fetcher = (
        AIOHttpFetcherBackend(network_policy=policy)
        .get_fetchers()
        .cert_fetcher
    )

    def decision(fetcher):
        try:
            fetcher.network_policy.check_url(url)
            return None
        except NetworkPolicyError as exc:
            return exc.reason

    assert decision(requests_fetcher) == expected_reason
    assert decision(aio_fetcher) == expected_reason


class FlippingResolver(HostResolver):
    """Alternates a hostname between a public and an internal address."""

    def __init__(self):
        self._flipped = False

    def resolve_all(self, hostname, port):
        self._flipped = not self._flipped
        if self._flipped:
            return (ipaddress.ip_address(PUBLIC_IPV4),)
        return (ipaddress.ip_address('10.0.0.66'),)


@pytest.mark.asyncio
async def test_aiohttp_rechecks_address_when_connection_is_reopened():
    """
    aiohttp consults the connector's resolver on every new connection, so a
    DNS rebind that flips an allowed name to an internal address after a
    pooled connection is evicted cannot bypass the policy.
    """
    policy = NetworkAdmissionPolicy(
        allowed_hosts=('rebind.example.com',),
        resolver=FlippingResolver(),
    )
    resolver = PolicyAwareResolver(policy)
    first = await resolver.resolve('rebind.example.com', 443)
    assert first[0]['host'] == PUBLIC_IPV4
    with pytest.raises(NetworkPolicyError) as exc:
        await resolver.resolve('rebind.example.com', 443)
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
    await resolver.close()


def test_requests_rechecks_address_on_each_new_socket(monkeypatch):
    """
    The requests connection class resolves and re-validates the host inside
    _new_conn, i.e. whenever urllib3 opens a fresh socket rather than reusing
    a pooled one. A rebinding name is therefore re-checked on every connect.
    """
    from pyhanko_certvalidator.fetchers.requests_fetchers.policy_transport import (
        _connect_to_approved_address,
    )

    policy = NetworkAdmissionPolicy(
        allowed_hosts=('rebind.example.com',),
        resolver=FlippingResolver(),
    )

    class _Connection:
        _dns_host = 'rebind.example.com'
        port = 443

    FakeSocket.connections = []
    monkeypatch.setattr(socket, 'socket', FakeSocket)
    sock = _connect_to_approved_address(
        _Connection(), policy, 'rebind.example.com', 443, 5
    )
    assert FakeSocket.connections == [(PUBLIC_IPV4, 443)]
    sock.close()
    with pytest.raises(NetworkPolicyError) as exc:
        _connect_to_approved_address(
            _Connection(), policy, 'rebind.example.com', 443, 5
        )
    assert exc.value.reason == PolicyDenialReason.IP_NOT_ALLOWED
