"""
Tests for the optional revocation evidence report and the trustless offline
replay machinery.

Scenarios with fixed timestamps:

1. an expired OCSP response (nextUpdate in the past) -> stale, not evidence;
2. a revocation that is not yet effective at the validation time;
3. an OCSP response with the right serial but a different issuer;
4. missing revocation material under soft-fail and under hard-fail.
"""

import json
from datetime import timedelta

import pytest

from pyhanko_certvalidator.context import ValidationContext
from pyhanko_certvalidator.errors import (
    CRLFetchError,
    InsufficientRevinfoError,
    OCSPFetchError,
    RevokedError,
    StaleRevinfoError,
)
from pyhanko_certvalidator.evidence import (
    EvidenceOutcome,
    EvidenceReplayError,
    MaterialKind,
    MaterialSource,
    OCSPCertStatus,
    OfflineReplaySecurityError,
    RevocationDecision,
    report_from_json,
    report_to_json,
    replay_validation_offline,
)
from pyhanko_certvalidator.fetchers import (
    CertificateFetcher,
    CRLFetcher,
    FetcherBackend,
    Fetchers,
    OCSPFetcher,
)
from pyhanko_certvalidator.policy_decl import (
    CertRevTrustPolicy,
    RevocationCheckingPolicy,
    RevocationCheckingRule,
)
from pyhanko_certvalidator.validate import async_validate_path

from .evidence_helpers import (
    LEAF_SERIAL,
    VALIDATION_MOMENT,
    build_crl,
    build_delegated_responder_cert,
    build_ocsp_response,
    build_pki,
)

SOFT_FAIL_POLICY = CertRevTrustPolicy(
    RevocationCheckingPolicy.from_legacy('soft-fail')
)
HARD_FAIL_POLICY = CertRevTrustPolicy(
    RevocationCheckingPolicy.from_legacy('hard-fail')
)


class _FailingOCSPFetcher(OCSPFetcher):
    def fetched_responses(self):
        return ()

    def fetched_responses_for_cert(self, cert):
        return ()

    async def fetch(self, cert, authority):
        raise OCSPFetchError("simulated OCSP unreachable")


class _FailingCRLFetcher(CRLFetcher):
    def fetched_crls(self):
        return ()

    def fetched_crls_for_cert(self, cert):
        raise KeyError

    async def fetch(self, cert, *, use_deltas=None):
        raise CRLFetchError("simulated CRL unreachable")


class _NullCertFetcher(CertificateFetcher):
    def fetched_certs(self):
        return ()

    def fetch_cert_issuers(self, cert):
        return self

    def fetch_crl_issuers(self, certificate_list):
        return self

    def __aiter__(self):
        return iter(())

    async def __anext__(self):
        raise StopAsyncIteration


class FailingFetcherBackend(FetcherBackend):
    def get_fetchers(self) -> Fetchers:
        return Fetchers(
            ocsp_fetcher=_FailingOCSPFetcher(),
            crl_fetcher=_FailingCRLFetcher(),
            cert_fetcher=_NullCertFetcher(),
        )


def _context(pki, **kwargs):
    kwargs.setdefault('trust_roots', [pki['ca_a']])
    kwargs.setdefault('moment', VALIDATION_MOMENT)
    kwargs.setdefault('collect_evidence', True)
    kwargs.setdefault('revinfo_policy', SOFT_FAIL_POLICY)
    return ValidationContext(**kwargs)


async def _run(vc, leaf):
    paths = await vc.path_builder.async_build_paths(leaf)
    return await async_validate_path(vc, paths[0])


def _leaf_entry(report):
    entries = [c for c in report.certificates if c.is_ee_cert]
    assert len(entries) == 1
    return entries[0]


# --------------------------------------------------------------------------
# 1. expired OCSP (nextUpdate before validation time)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_ocsp_is_stale_and_hard_fails():
    pki = build_pki()
    expired_ocsp = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(days=30),
        next_update=VALIDATION_MOMENT - timedelta(days=1),
    )
    vc = _context(pki, ocsps=[expired_ocsp])
    # matched but stale revinfo is a hard failure even in legacy soft-fail
    with pytest.raises(StaleRevinfoError):
        await _run(vc, pki['leaf_a'])

    report = vc.build_evidence_report()
    data = report_to_json(report)
    report = report_from_json(json.loads(json.dumps(data)))  # JSON round-trip

    leaf_entry = _leaf_entry(report)
    (ocsp_ev,) = leaf_entry.ocsp_evidence
    assert ocsp_ev.accepted is False
    assert ocsp_ev.freshness_rating == 'stale'
    assert ocsp_ev.cert_status == OCSPCertStatus.GOOD
    assert ocsp_ev.signature_valid is True
    assert ocsp_ev.reference.source == MaterialSource.PROVIDED
    assert 'not recent enough' in ocsp_ev.selection_reason
    # the rejected material still appears in the report's material index
    assert ocsp_ev.reference.sha256 in report.materials

    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        raw_ocsps=[expired_ocsp.dump()],
    )
    # replay reproduces the same hard rejection (stale is not soft-fail)
    assert outcome.accepted is False
    assert outcome.soft_failed is False
    assert outcome.exception_type == 'StaleRevinfoError'
    assert outcome.report.fetching_allowed is False


# --------------------------------------------------------------------------
# 2. not-yet-effective revocation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_future_revocation_does_not_reject_but_is_recorded():
    pki = build_pki()
    future = VALIDATION_MOMENT + timedelta(days=10)
    future_rev_ocsp = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='revoked',
        revocation_time=future,
        revocation_reason='key_compromise',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[future_rev_ocsp])
    # a matched OCSP response that does not resolve to "good" at the
    # validation time is treated as inconclusive, which legacy soft-fail
    # still rejects when material was matched
    with pytest.raises(InsufficientRevinfoError):
        await _run(vc, pki['leaf_a'])

    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    (ocsp_ev,) = leaf_entry.ocsp_evidence
    assert ocsp_ev.cert_status == OCSPCertStatus.REVOKED
    assert ocsp_ev.revocation_time == future
    assert ocsp_ev.revocation_reason == 'key_compromise'
    # the precise timing reason is preserved rather than collapsed into
    # a generic exception
    assert 'not yet effective' in ocsp_ev.selection_reason

    # replay reproduces the same inconclusive rejection offline
    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        raw_ocsps=[future_rev_ocsp.dump()],
    )
    assert outcome.accepted is False
    assert outcome.revoked is False
    assert any(
        'not yet effective' in e.selection_reason
        for c in outcome.report.certificates
        for e in c.ocsp_evidence
    )


@pytest.mark.asyncio
async def test_effective_revocation_rejects_and_replays_as_revoked():
    pki = build_pki()
    past = VALIDATION_MOMENT - timedelta(days=2)
    rev_ocsp = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='revoked',
        revocation_time=past,
        revocation_reason='key_compromise',
        this_update=VALIDATION_MOMENT - timedelta(hours=2),
        next_update=VALIDATION_MOMENT + timedelta(hours=2),
    )
    vc = _context(pki, ocsps=[rev_ocsp])
    with pytest.raises(RevokedError):
        await _run(vc, pki['leaf_a'])

    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    assert leaf_entry.outcome == EvidenceOutcome.REVOKED
    assert leaf_entry.decision == RevocationDecision.REJECTED_REVOKED
    (ocsp_ev,) = leaf_entry.ocsp_evidence
    assert ocsp_ev.cert_status == OCSPCertStatus.REVOKED
    assert ocsp_ev.revocation_time == past

    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        raw_ocsps=[rev_ocsp.dump()],
    )
    assert outcome.revoked is True
    assert outcome.accepted is False
    assert 'rejected' in outcome.explanation


# --------------------------------------------------------------------------
# 3. same serial number, different issuer
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ocsp_for_other_issuer_same_serial_is_not_evidence():
    pki = build_pki()
    # response signed by CA B, carrying CA B's CertID; the serial equals
    # leaf A's serial, but the issuer hashes do not match CA A
    foreign_ocsp = build_ocsp_response(
        target_cert=pki['leaf_b'],
        issuer_cert=pki['ca_b'],
        issuer_key=pki['ca_b_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, trust_roots=[pki['ca_a']], ocsps=[foreign_ocsp])
    # no in-scope material -> soft-fail continues
    await _run(vc, pki['leaf_a'])

    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    (ocsp_ev,) = leaf_entry.ocsp_evidence
    assert ocsp_ev.cert_id_matched is False
    assert ocsp_ev.accepted is False
    assert 'different certificate/issuer' in ocsp_ev.selection_reason
    # correctly signed foreign material must not establish good status
    assert leaf_entry.outcome == EvidenceOutcome.UNKNOWN
    assert any(
        u.kind == MaterialKind.OCSP and 'issuer' in u.reason
        for u in leaf_entry.unavailable
    )

    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        raw_ocsps=[foreign_ocsp.dump()],
    )
    assert outcome.accepted is True
    assert outcome.soft_failed is True
    replay_leaf = _leaf_entry(outcome.report)
    assert all(not e.accepted for e in replay_leaf.ocsp_evidence)


@pytest.mark.asyncio
async def test_delegated_responder_without_authority_is_rejected():
    pki = build_pki()
    # responder cert issued by CA B, answering about CA A's leaf: the
    # signature verifies against the embedded cert, but its chain does not
    # terminate at CA A, so it is not authorised evidence
    rogue_cert, rogue_key = build_delegated_responder_cert(
        "rogue responder", pki['ca_b_name'], pki['ca_b'], pki['ca_b_key']
    )
    bad_ocsp = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
        responder_cert=rogue_cert,
        responder_key=rogue_key,
    )
    vc = _context(
        pki,
        trust_roots=[pki['ca_a']],
        other_certs=[rogue_cert],
        ocsps=[bad_ocsp],
    )
    # the response matches the cert and its signature verifies against the
    # embedded responder cert, but the responder chain does not terminate at
    # CA A; a matched-but-unauthorised response is a hard failure
    with pytest.raises(InsufficientRevinfoError):
        await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    (ocsp_ev,) = leaf_entry.ocsp_evidence
    assert ocsp_ev.signature_valid is True
    assert ocsp_ev.responder_authorised is False
    assert ocsp_ev.accepted is False
    assert 'not authorised' in ocsp_ev.selection_reason
    assert leaf_entry.outcome == EvidenceOutcome.FAILED


# --------------------------------------------------------------------------
# 4. missing material: soft-fail vs hard-fail, plus fetch-error soft-fail
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_material_soft_fail_records_unavailable():
    pki = build_pki()
    vc = _context(pki)  # no ocsps, no crls, no fetching
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    assert leaf_entry.decision == RevocationDecision.SOFT_FAIL_CONTINUED
    kinds = {u.kind for u in leaf_entry.unavailable}
    assert MaterialKind.OCSP in kinds and MaterialKind.CRL in kinds
    assert report.materials == {}

    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
    )
    assert outcome.accepted is True
    assert outcome.soft_failed is True


@pytest.mark.asyncio
async def test_fetch_error_soft_fail_is_recorded_with_provenance():
    pki = build_pki()
    vc = ValidationContext(
        trust_roots=[pki['ca_a']],
        moment=VALIDATION_MOMENT,
        collect_evidence=True,
        revinfo_policy=SOFT_FAIL_POLICY,
        allow_fetching=True,
        fetcher_backend=FailingFetcherBackend(),
    )
    # fetch failures are tolerated in soft-fail, so validation continues
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    assert leaf_entry.decision == RevocationDecision.SOFT_FAIL_CONTINUED
    sf_kinds = {s.kind for s in leaf_entry.soft_fails}
    assert MaterialKind.OCSP in sf_kinds and MaterialKind.CRL in sf_kinds
    assert all(
        'FetchError' in s.exception_type for s in leaf_entry.soft_fails
    )


@pytest.mark.asyncio
async def test_missing_material_hard_fail_rejects_and_replays_rejection():
    pki = build_pki()
    # hard-fail requires some material to be present; supply a response that
    # has no jurisdiction over CA A's leaf so the check still finds nothing
    # usable and hard-fails
    foreign_ocsp = build_ocsp_response(
        target_cert=pki['leaf_b'],
        issuer_cert=pki['ca_b'],
        issuer_key=pki['ca_b_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(
        pki,
        revinfo_policy=HARD_FAIL_POLICY,
        trust_roots=[pki['ca_a']],
        ocsps=[foreign_ocsp],
    )
    with pytest.raises(InsufficientRevinfoError):
        await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    assert leaf_entry.outcome == EvidenceOutcome.FAILED

    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=HARD_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        raw_ocsps=[foreign_ocsp.dump()],
    )
    assert outcome.accepted is False
    assert outcome.soft_failed is False
    assert 'insufficient' in outcome.explanation.lower()


# --------------------------------------------------------------------------
# CRL coverage and not-yet-effective CRL revocation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crl_coverage_and_future_crl_revocation():
    pki = build_pki()
    future = VALIDATION_MOMENT + timedelta(days=5)
    crl_future_rev = build_crl(
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
        entries=[
            {
                'serial': LEAF_SERIAL,
                'revocation_date': future,
                'reason': 'cessation_of_operation',
            }
        ],
    )
    vc = _context(pki, crls=[crl_future_rev])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    (crl_ev,) = leaf_entry.crl_evidence
    assert crl_ev.accepted is True
    assert crl_ev.cert_listed is True
    assert crl_ev.revocation_time == future
    assert crl_ev.issuer_matched is True
    assert crl_ev.signature_valid is True
    assert crl_ev.scope_reasons  # CRL coverage reasons are recorded

    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_a']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        raw_crls=[crl_future_rev.dump()],
    )
    assert outcome.accepted is True


@pytest.mark.asyncio
async def test_crl_from_other_issuer_has_no_jurisdiction():
    pki = build_pki()
    # a CRL signed by CA B cannot cover CA A's leaf, even if serial collides
    foreign_crl = build_crl(
        issuer_cert=pki['ca_b'],
        issuer_key=pki['ca_b_key'],
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
        entries=[
            {'serial': LEAF_SERIAL, 'revocation_date': VALIDATION_MOMENT}
        ],
    )
    vc = _context(pki, trust_roots=[pki['ca_a']], crls=[foreign_crl])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    (crl_ev,) = leaf_entry.crl_evidence
    assert crl_ev.issuer_matched is False
    assert crl_ev.accepted is False
    assert leaf_entry.outcome == EvidenceOutcome.UNKNOWN


# --------------------------------------------------------------------------
# replay security properties
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_requires_explicit_trust_roots():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[good])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    with pytest.raises(OfflineReplaySecurityError):
        await replay_validation_offline(
            report,
            trust_roots=[],
            revinfo_policy=SOFT_FAIL_POLICY,
            end_entity_cert=pki['leaf_a'],
            raw_ocsps=[good.dump()],
        )


@pytest.mark.asyncio
async def test_replay_rejects_policy_mismatch():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[good])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    with pytest.raises(EvidenceReplayError, match='policy'):
        await replay_validation_offline(
            report,
            trust_roots=[pki['ca_a']],
            revinfo_policy=HARD_FAIL_POLICY,
            end_entity_cert=pki['leaf_a'],
            raw_ocsps=[good.dump()],
        )


@pytest.mark.asyncio
async def test_replay_rejects_tampered_or_missing_raw_bytes():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[good])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()

    with pytest.raises(EvidenceReplayError, match='not supplied'):
        await replay_validation_offline(
            report,
            trust_roots=[pki['ca_a']],
            revinfo_policy=SOFT_FAIL_POLICY,
            end_entity_cert=pki['leaf_a'],
        )

    tampered = bytearray(good.dump())
    tampered[-1] ^= 0x01
    with pytest.raises(EvidenceReplayError, match='inconsistent'):
        await replay_validation_offline(
            report,
            trust_roots=[pki['ca_a']],
            revinfo_policy=SOFT_FAIL_POLICY,
            end_entity_cert=pki['leaf_a'],
            raw_ocsps=[bytes(tampered)],
        )


@pytest.mark.asyncio
async def test_unknown_ocsp_status_is_distinguished():
    pki = build_pki()
    unknown_ocsp = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='unknown',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[unknown_ocsp])
    # an explicit unknown is in-scope but inconclusive; legacy soft-fail
    # still rejects a matched-but-inconclusive response
    with pytest.raises(InsufficientRevinfoError):
        await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    leaf_entry = _leaf_entry(report)
    (ocsp_ev,) = leaf_entry.ocsp_evidence
    assert ocsp_ev.cert_status == OCSPCertStatus.UNKNOWN
    assert ocsp_ev.accepted is False
    # the distinct "unknown" status is preserved, not collapsed into a
    # generic good/revoked label
    assert leaf_entry.outcome == EvidenceOutcome.FAILED


@pytest.mark.asyncio
async def test_report_records_validation_time_and_fingerprints():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[good])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    assert report.validation_time == VALIDATION_MOMENT
    assert report.point_in_time_validation is True
    assert report.policy.ee_rule == (
        RevocationCheckingRule.CHECK_IF_DECLARED_SOFT.value
    )
    leaf_fp = report.certificates[-1].cert_fingerprints['sha256']
    path_fps = [f['sha256'] for f in report.path.cert_fingerprints]
    assert leaf_fp in path_fps
    leaf_entry = report.certificates[-1]
    assert (
        leaf_entry.issuer_fingerprints['sha256']
        == report.path.cert_fingerprints[-2]['sha256']
    )


@pytest.mark.asyncio
async def test_disabled_collector_changes_nothing():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = ValidationContext(
        trust_roots=[pki['ca_a']],
        ocsps=[good],
        moment=VALIDATION_MOMENT,
        revinfo_policy=SOFT_FAIL_POLICY,
    )
    assert vc.evidence_collection_enabled is False
    assert vc.evidence_collector is None
    await _run(vc, pki['leaf_a'])
    with pytest.raises(ValueError):
        vc.build_evidence_report()


# --------------------------------------------------------------------------
# provenance: cache hits and "fetched != valid"
# --------------------------------------------------------------------------


class _CachingOCSPFetcher(OCSPFetcher):
    def __init__(self, responses):
        self._responses = list(responses)
        self._served = []
        self.fetch_calls = 0

    def fetched_responses(self):
        return list(self._served)

    def fetched_responses_for_cert(self, cert):
        return list(self._served)

    async def fetch(self, cert, authority):
        self.fetch_calls += 1
        # emulate a fetcher that caches after downloading
        self._served.extend(self._responses)
        # manager calls load_multi itself; return the first raw response
        return self._responses[0]


class _CachingCRLFetcher(CRLFetcher):
    def __init__(self, crls):
        self._crls = list(crls)
        self._served = []
        self.fetch_calls = 0

    def fetched_crls(self):
        return list(self._served)

    def fetched_crls_for_cert(self, cert):
        raise KeyError

    async def fetch(self, cert, *, use_deltas=None):
        self.fetch_calls += 1
        self._served.extend(self._crls)
        return list(self._crls)


class _CachingFetcherBackend(FetcherBackend):
    def __init__(self, ocsp_responses, crls):
        self._ocsp = _CachingOCSPFetcher(ocsp_responses)
        self._crl = _CachingCRLFetcher(crls)

    def get_fetchers(self) -> Fetchers:
        return Fetchers(
            ocsp_fetcher=self._ocsp,
            crl_fetcher=self._crl,
            cert_fetcher=_NullCertFetcher(),
        )


@pytest.mark.asyncio
async def test_fetched_then_cached_provenance_and_download_not_trusted():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    backend = _CachingFetcherBackend([good], [])

    # first run: material is downloaded
    vc1 = ValidationContext(
        trust_roots=[pki['ca_a']],
        moment=VALIDATION_MOMENT,
        collect_evidence=True,
        revinfo_policy=SOFT_FAIL_POLICY,
        allow_fetching=True,
        fetcher_backend=backend,
    )
    await _run(vc1, pki['leaf_a'])
    report1 = vc1.build_evidence_report()
    ocsp_ref = [
        e.reference
        for c in report1.certificates
        for e in c.ocsp_evidence
    ][0]
    assert ocsp_ref.source == MaterialSource.FETCHED
    assert backend._ocsp.fetch_calls == 1

    # a freshly downloaded, correctly signed response still had to pass the
    # full relevance/authorisation checks before counting as evidence
    accepted = [
        e
        for c in report1.certificates
        for e in c.ocsp_evidence
        if e.accepted
    ]
    assert accepted and accepted[0].signature_valid is True

    # second run with a new context backed by the same warmed cache:
    # no new fetch, provenance is a cache hit
    vc2 = ValidationContext(
        trust_roots=[pki['ca_a']],
        moment=VALIDATION_MOMENT,
        collect_evidence=True,
        revinfo_policy=SOFT_FAIL_POLICY,
        allow_fetching=True,
        fetcher_backend=backend,
    )
    await _run(vc2, pki['leaf_a'])
    report2 = vc2.build_evidence_report()
    ocsp_ref2 = [
        e.reference
        for c in report2.certificates
        for e in c.ocsp_evidence
    ][0]
    assert ocsp_ref2.source == MaterialSource.CACHE
    assert ocsp_ref2.sha256 == ocsp_ref.sha256


@pytest.mark.asyncio
async def test_successfully_downloaded_but_invalid_material_is_not_evidence():
    pki = build_pki()
    # "download" an OCSP response that belongs to CA B; the fetch succeeds,
    # but the material can never validate CA A's leaf
    foreign = build_ocsp_response(
        target_cert=pki['leaf_b'],
        issuer_cert=pki['ca_b'],
        issuer_key=pki['ca_b_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    backend = _CachingFetcherBackend([foreign], [])
    vc = ValidationContext(
        trust_roots=[pki['ca_a']],
        moment=VALIDATION_MOMENT,
        collect_evidence=True,
        revinfo_policy=SOFT_FAIL_POLICY,
        allow_fetching=True,
        fetcher_backend=backend,
    )
    # soft-fail continues because nothing in-scope was found
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    (ocsp_ev,) = [e for c in report.certificates for e in c.ocsp_evidence]
    # the bytes were genuinely fetched ...
    assert ocsp_ev.reference.source == MaterialSource.FETCHED
    # ... but never qualified as valid status evidence
    assert ocsp_ev.accepted is False
    assert ocsp_ev.cert_id_matched is False


@pytest.mark.asyncio
async def test_multi_single_response_shares_one_material_digest():
    from asn1crypto import algos, core
    from asn1crypto import ocsp as asn1_ocsp
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    from pyhanko_certvalidator.evidence.model import OCSP_HASH
    from tests.evidence_helpers import (
        _build_leaf,
        _cert_id,
        _new_key,
    )

    pki = build_pki()
    leaf2_key = _new_key()
    leaf2 = _build_leaf(
        "evidence-leaf-2",
        leaf2_key,
        pki['ca_a_name'],
        pki['ca_a_key'],
        999,
    )
    first = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    basic = first['response_bytes']['response'].parsed
    tbs = basic['tbs_response_data']
    second = asn1_ocsp.SingleResponse(
        {
            'cert_id': _cert_id(leaf2, pki['ca_a']),
            'cert_status': asn1_ocsp.CertStatus(
                name='good', value=asn1_ocsp.StatusGood()
            ),
            'this_update': VALIDATION_MOMENT - timedelta(hours=1),
            'next_update': VALIDATION_MOMENT + timedelta(hours=1),
        }
    )
    tbs['responses'] = list(tbs['responses']) + [second]
    signature = pki['ca_a_key'].sign(
        tbs.dump(), padding.PKCS1v15(), hashes.SHA256()
    )
    multi = asn1_ocsp.OCSPResponse(
        {
            'response_status': 'successful',
            'response_bytes': asn1_ocsp.ResponseBytes(
                {
                    'response_type': 'basic_ocsp_response',
                    'response': core.ParsableOctetString(
                        asn1_ocsp.BasicOCSPResponse(
                            {
                                'tbs_response_data': tbs,
                                'signature_algorithm':
                                    algos.SignedDigestAlgorithm(
                                        {'algorithm': 'sha256_rsa'}
                                    ),
                                'signature': core.OctetBitString(signature),
                            }
                        ).dump()
                    ),
                }
            ),
        }
    )
    vc = _context(pki, ocsps=[multi])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()
    refs = {
        e.reference.sha256
        for c in report.certificates
        for e in c.ocsp_evidence
    }
    # one raw response, one digest, regardless of how many SingleResponses
    assert refs == {OCSP_HASH(multi)}
    assert OCSP_HASH(multi) in report.materials


@pytest.mark.asyncio
async def test_replay_with_wrong_trust_root_is_not_rescued_by_report():
    pki = build_pki()
    good = build_ocsp_response(
        target_cert=pki['leaf_a'],
        issuer_cert=pki['ca_a'],
        issuer_key=pki['ca_a_key'],
        status='good',
        this_update=VALIDATION_MOMENT - timedelta(hours=1),
        next_update=VALIDATION_MOMENT + timedelta(hours=1),
    )
    vc = _context(pki, ocsps=[good])
    await _run(vc, pki['leaf_a'])
    report = vc.build_evidence_report()

    # CA B is presented as the trust root; CA A's chain cannot terminate
    # there, so replay must fail even though the report claims success
    outcome = await replay_validation_offline(
        report,
        trust_roots=[pki['ca_b']],
        revinfo_policy=SOFT_FAIL_POLICY,
        end_entity_cert=pki['leaf_a'],
        intermediate_certs=[pki['ca_a']],
        raw_ocsps=[good.dump()],
    )
    assert outcome.accepted is False
    assert outcome.revalidated_path is None
