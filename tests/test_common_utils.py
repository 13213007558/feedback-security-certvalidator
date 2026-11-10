import os

from asn1crypto import x509

from pyhanko_certvalidator.fetchers.common_utils import (
    enumerate_delivery_point_urls,
    unpack_cert_content,
)

TESTS_ROOT = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(TESTS_ROOT, 'fixtures')


def test_unpack_cert_content_pkcs7_with_binary_octet_stream_alias():
    with open(
        os.path.join(FIXTURES_DIR, 'certs_to_unpack/acserprorfbv5.p7b'), 'rb'
    ) as f:
        pkcs7_bytes = f.read()

    certs_returned = unpack_cert_content(
        response_data=pkcs7_bytes,
        content_type="binary/octet-stream",
        permit_pem=True,
        url="http://repositorio.serpro.gov.br/cadeias/acserprorfbv5.p7b",
    )
    assert len(list(certs_returned)) == 3


def test_crl_distribution_point_enumeration_skip_ldap():
    with open(os.path.join(FIXTURES_DIR, 'PostaSrbijeCA1.der'), 'rb') as f:
        cert = x509.Certificate.load(f.read())
    (dist_point,) = cert.crl_distribution_points_value
    (url,) = enumerate_delivery_point_urls(dist_point)
    assert url.startswith('http://')


import asyncio

import pytest

from pyhanko_certvalidator import errors
from pyhanko_certvalidator.fetchers.common_utils import queue_fetch_task


class PausableFetcher:
    """Fake fetcher synchronised with event barriers (no sleeps)."""

    def __init__(self, result='RESULT', error=None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.calls = 0
        self.result = result
        self.error = error

    async def __call__(self):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.error is not None:
            raise self.error
        return self.result


async def _yield_loop(n=3):
    # scheduling barrier: let pending callbacks run (not a timing sleep)
    for _ in range(n):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_queue_fetch_task_initiator_cancelled_waiter_completes():
    results, running_jobs = {}, {}
    fetch = PausableFetcher()
    initiator = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await fetch.started.wait()
    waiter = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await _yield_loop()

    initiator.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initiator

    # the shared fetch must survive the initiator's cancellation
    assert not fetch.cancelled.is_set()
    fetch.release.set()
    assert await waiter == 'RESULT'
    assert fetch.calls == 1
    # job completed -> result cached, registry cleaned up
    assert results == {'tag': 'RESULT'}
    assert running_jobs == {}


@pytest.mark.asyncio
async def test_queue_fetch_task_waiter_cancelled_initiator_completes():
    results, running_jobs = {}, {}
    fetch = PausableFetcher()
    initiator = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await fetch.started.wait()
    waiter = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await _yield_loop()

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    fetch.release.set()
    assert await initiator == 'RESULT'
    assert fetch.calls == 1
    assert results == {'tag': 'RESULT'}
    assert running_jobs == {}


@pytest.mark.asyncio
async def test_queue_fetch_task_all_waiters_leave_no_orphan_no_cache():
    results, running_jobs = {}, {}
    fetch = PausableFetcher()
    initiator = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await fetch.started.wait()
    waiter = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await _yield_loop()

    initiator.cancel()
    waiter.cancel()
    for task in (initiator, waiter):
        with pytest.raises(asyncio.CancelledError):
            await task
    await _yield_loop()

    # last waiter leaving must cancel the shared fetch (no orphan task),
    # drop the registry entry and cache nothing
    assert fetch.cancelled.is_set()
    assert running_jobs == {}
    assert results == {}

    # a later identical request must start a fresh fetch, not block
    # on a leftover event or a cached cancellation
    retry = PausableFetcher(result='FRESH')
    retry.release.set()
    assert await queue_fetch_task(results, running_jobs, 'tag', retry) == 'FRESH'
    assert retry.calls == 1
    assert results == {'tag': 'FRESH'}
    assert running_jobs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error_type',
    [
        errors.CertificateFetchError,
        errors.CRLFetchError,
        errors.OCSPFetchError,
    ],
)
async def test_queue_fetch_task_error_propagates_to_all_waiters(error_type):
    # cert, CRL and OCSP fetchers all wrap I/O errors in their public
    # exception type inside the fetch callback; queue_fetch_task must
    # relay that exception to every waiter without re-wrapping it
    results, running_jobs = {}, {}
    fetch = PausableFetcher(error=error_type('boom'))
    fetch.release.set()
    initiator = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await fetch.started.wait()
    waiter = asyncio.ensure_future(
        queue_fetch_task(results, running_jobs, 'tag', fetch)
    )
    await _yield_loop()

    with pytest.raises(error_type, match='boom'):
        await initiator
    with pytest.raises(error_type, match='boom'):
        await waiter
    assert fetch.calls == 1
    assert isinstance(results['tag'], error_type)
    assert running_jobs == {}


@pytest.mark.asyncio
async def test_queue_fetch_task_result_shared_and_cached():
    results, running_jobs = {}, {}
    fetch = PausableFetcher()
    tasks = [
        asyncio.ensure_future(queue_fetch_task(results, running_jobs, 'tag', fetch))
        for _ in range(3)
    ]
    await fetch.started.wait()
    await _yield_loop()
    fetch.release.set()
    assert [await t for t in tasks] == ['RESULT'] * 3
    assert fetch.calls == 1
    assert results == {'tag': 'RESULT'}
    assert running_jobs == {}

    # repeat request is served from the cache without re-fetching
    other = PausableFetcher(result='SHOULD-NOT-BE-USED')
    other.release.set()
    assert await queue_fetch_task(results, running_jobs, 'tag', other) == 'RESULT'
    assert other.calls == 0
