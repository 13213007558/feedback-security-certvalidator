"""
Offline replay of revocation decisions from an evidence report.

The report itself is never trusted to establish authenticity. A replay:

1. runs with network access unconditionally disabled (no fetchers);
2. requires the caller to supply the trust roots and the revocation policy
   explicitly, and verifies that the supplied policy matches the policy
   recorded in the report;
3. feeds only the raw bytes that the report says participated (and checks
   their digests), then reruns the ordinary validation logic so that every
   signature, scope, freshness and authorisation check is performed again.

The replay result explains why the original decision accepted, rejected or
merely soft-failed, rather than merely echoing the last exception.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from asn1crypto import crl, ocsp, x509

from ..context import ValidationContext
from ..errors import (
    InsufficientRevinfoError,
    PathBuildingError,
    PathValidationError,
    RevokedError,
    StaleRevinfoError,
)
from ..path import ValidationPath
from ..policy_decl import CertRevTrustPolicy
from ..registry import TrustRootList
from ..validate import async_validate_path
from .model import (
    MaterialKind,
    RevocationEvidenceReport,
    RevocationDecision,
    digest_bytes,
)


class EvidenceReplayError(Exception):
    """
    Raised when an evidence report cannot be replayed because the inputs
    supplied by the caller are inconsistent with the report.
    """


class OfflineReplaySecurityError(EvidenceReplayError):
    """
    Raised when a replay would require trusting the report's own claims
    without caller-supplied authentication material.
    """


@dataclass(frozen=True)
class ReplayOutcome:
    """
    Result of replaying one evidence report offline.
    """

    accepted: bool
    revoked: bool
    soft_failed: bool
    explanation: str
    exception_type: Optional[str] = None
    report: Optional[RevocationEvidenceReport] = None
    revalidated_path: Optional[ValidationPath] = None
    digest_mismatches: List[str] = field(default_factory=list)


def build_report_from_context(
    context: ValidationContext,
) -> RevocationEvidenceReport:
    collector = context.evidence_collector
    if collector is None:
        raise ValueError("this context does not collect evidence")
    return collector.build_report(dict(collector._finalised))


def _check_raw_material_bundle(
    report: RevocationEvidenceReport,
    raw_ocsps: Iterable[bytes],
    raw_crls: Iterable[bytes],
) -> Tuple[List[ocsp.OCSPResponse], List[crl.CertificateList], List[str]]:
    expected = {
        digest: ref
        for digest, ref in report.materials.items()
        if ref.kind in (MaterialKind.OCSP, MaterialKind.CRL)
    }
    seen: Dict[str, bytes] = {}
    mismatches: List[str] = []
    parsed_ocsps: List[ocsp.OCSPResponse] = []
    parsed_crls: List[crl.CertificateList] = []

    for data in raw_ocsps:
        digest_hex = digest_bytes(data)
        seen[digest_hex] = data
        ref = expected.get(digest_hex)
        if ref is None:
            mismatches.append(
                f"OCSP material {digest_hex} is not referenced by the report"
            )
            continue
        if ref.kind != MaterialKind.OCSP:
            mismatches.append(
                f"material {digest_hex} supplied as OCSP but report records "
                f"it as {ref.kind.value}"
            )
            continue
        parsed_ocsps.append(ocsp.OCSPResponse.load(data))

    for data in raw_crls:
        digest_hex = digest_bytes(data)
        seen[digest_hex] = data
        ref = expected.get(digest_hex)
        if ref is None:
            mismatches.append(
                f"CRL material {digest_hex} is not referenced by the report"
            )
            continue
        if ref.kind != MaterialKind.CRL:
            mismatches.append(
                f"material {digest_hex} supplied as CRL but report records "
                f"it as {ref.kind.value}"
            )
            continue
        parsed_crls.append(crl.CertificateList.load(data))

    missing = sorted(set(expected) - set(seen))
    for digest_hex in missing:
        ref = expected[digest_hex]
        mismatches.append(
            f"raw bytes for {ref.kind.value} material {digest_hex} that "
            f"participated in the original decision were not supplied"
        )
    return parsed_ocsps, parsed_crls, mismatches


async def replay_validation_offline(
    report: RevocationEvidenceReport,
    *,
    trust_roots: TrustRootList,
    revinfo_policy: CertRevTrustPolicy,
    end_entity_cert: x509.Certificate,
    intermediate_certs: Optional[Iterable[x509.Certificate]] = None,
    raw_ocsps: Optional[Iterable[bytes]] = None,
    raw_crls: Optional[Iterable[bytes]] = None,
    time_tolerance: Optional[timedelta] = None,
    algorithm_usage_policy=None,
    other_certs: Optional[Iterable[x509.Certificate]] = None,
) -> ReplayOutcome:
    """
    Replay the validation described by ``report`` without any network access.

    :param report:
        The evidence report produced by the original run.
    :param trust_roots:
        Trust roots to use. Mandatory: the report is never treated as a
        source of trust, and omitting trust roots raises
        :class:`OfflineReplaySecurityError`.
    :param revinfo_policy:
        The revocation policy to apply. It must be equivalent to the policy
        recorded in the report, otherwise :class:`EvidenceReplayError` is
        raised before any validation takes place.
    :param end_entity_cert:
        The end-entity certificate whose validation is replayed.
    :param intermediate_certs:
        Intermediate certificates needed to build the path (typically
        recovered from the original chain).
    :param raw_ocsps:
        Raw DER bytes of OCSP responses saved alongside the report.
    :param raw_crls:
        Raw DER bytes of CRLs saved alongside the report.
    :param time_tolerance:
        Time tolerance. Must match the report if supplied.
    :param algorithm_usage_policy:
        Optional algorithm usage policy.
    :param other_certs:
        Additional untrusted certificates (e.g. OCSP responder certs).
    :return:
        A :class:`ReplayOutcome` describing the replayed decision.
    """

    if not trust_roots:
        raise OfflineReplaySecurityError(
            "Offline replay requires explicit caller-supplied trust roots; "
            "the evidence report cannot be used as a trust source."
        )
    if revinfo_policy is None:
        raise OfflineReplaySecurityError(
            "Offline replay requires an explicit caller-supplied revocation "
            "policy."
        )

    tolerance = (
        time_tolerance
        if time_tolerance is not None
        else timedelta(seconds=report.policy.time_tolerance_seconds)
    )
    matches, discrepancies = report.policy.matches_policy(
        revinfo_policy, tolerance.total_seconds()
    )
    if not matches:
        raise EvidenceReplayError(
            "The supplied replay policy does not match the policy recorded "
            "in the report: " + '; '.join(discrepancies)
        )

    raw_ocsps = list(raw_ocsps or ())
    raw_crls = list(raw_crls or ())
    parsed_ocsps, parsed_crls, digest_mismatches = _check_raw_material_bundle(
        report, raw_ocsps, raw_crls
    )
    if digest_mismatches:
        raise EvidenceReplayError(
            "Raw material bundle is inconsistent with the report: "
            + '; '.join(digest_mismatches)
        )

    # Defence in depth: even if a caller wires up fetchers elsewhere, the
    # replay context is constructed with allow_fetching left at its default
    # (False) and no fetcher backend.
    extras: List[x509.Certificate] = list(intermediate_certs or ())
    extras.extend(other_certs or ())

    context = ValidationContext(
        trust_roots=trust_roots,
        other_certs=extras,
        ocsps=parsed_ocsps,
        crls=parsed_crls,
        revinfo_policy=revinfo_policy,
        moment=report.validation_time,
        best_signature_time=report.best_signature_time,
        time_tolerance=tolerance,
        algorithm_usage_policy=algorithm_usage_policy,
        allow_fetching=False,
        collect_evidence=True,
    )
    if context.fetching_allowed:  # pragma: no cover
        raise OfflineReplaySecurityError(
            "Internal error: replay context was created with fetching enabled"
        )

    revalidated_path: Optional[ValidationPath] = None
    validation_error: Optional[PathValidationError] = None
    paths = None
    try:
        paths = context.path_builder.async_build_paths_lazy(end_entity_cert)
        async for candidate in paths:
            try:
                await async_validate_path(context, candidate)
                revalidated_path = candidate
                break
            except (RevokedError, InsufficientRevinfoError, StaleRevinfoError):
                raise
            except PathValidationError as exc:
                validation_error = exc
                continue
    except PathBuildingError as exc:
        replay_report = context.build_evidence_report()
        return ReplayOutcome(
            accepted=False,
            revoked=False,
            soft_failed=False,
            explanation=(
                "No validation path could be built offline from the "
                "caller-supplied certificates and trust roots: "
                f"{type(exc).__name__}: {exc}"
            ),
            exception_type=type(exc).__name__,
            report=replay_report,
        )
    except RevokedError as exc:
        replay_report = context.build_evidence_report()
        return ReplayOutcome(
            accepted=False,
            revoked=True,
            soft_failed=False,
            explanation=_explain_revoked(report, exc),
            exception_type=type(exc).__name__,
            report=replay_report,
            revalidated_path=revalidated_path,
        )
    except (InsufficientRevinfoError, StaleRevinfoError) as exc:
        replay_report = context.build_evidence_report()
        soft = _recorded_soft_fail(report)
        if soft:
            return ReplayOutcome(
                accepted=False,
                revoked=False,
                soft_failed=True,
                explanation=_explain_soft_fail(report, exc),
                exception_type=type(exc).__name__,
                report=replay_report,
            )
        return ReplayOutcome(
            accepted=False,
            revoked=False,
            soft_failed=False,
            explanation=_explain_insufficient(report, exc),
            exception_type=type(exc).__name__,
            report=replay_report,
        )
    finally:
        close = getattr(paths, 'cancel', None) or getattr(paths, 'aclose', None)
        if close is not None:
            result = close()
            if hasattr(result, '__await__'):
                await result

    if revalidated_path is None and validation_error is not None:
        replay_report = context.build_evidence_report()
        return ReplayOutcome(
            accepted=False,
            revoked=False,
            soft_failed=False,
            explanation=(
                "Offline replay rejected the path: "
                f"{type(validation_error).__name__}: "
                f"{validation_error.failure_msg}"
            ),
            exception_type=type(validation_error).__name__,
            report=replay_report,
        )

    if revalidated_path is None:
        replay_report = context.build_evidence_report()
        return ReplayOutcome(
            accepted=False,
            revoked=False,
            soft_failed=_recorded_soft_fail(report),
            explanation=(
                "No candidate path survived offline validation; inspect the "
                "replayed report for per-certificate reasons"
            ),
            report=replay_report,
        )

    replay_report = context.build_evidence_report()
    return ReplayOutcome(
        accepted=True,
        revoked=False,
        soft_failed=_recorded_soft_fail(report),
        explanation=_explain_acceptance(report),
        report=replay_report,
        revalidated_path=revalidated_path,
    )


def _recorded_soft_fail(report: RevocationEvidenceReport) -> bool:
    return any(
        cert.decision == RevocationDecision.SOFT_FAIL_CONTINUED
        for cert in report.certificates
    )


def _explain_acceptance(report: RevocationEvidenceReport) -> str:
    parts = []
    for cert in report.certificates:
        if cert.decision == RevocationDecision.ACCEPTED:
            parts.append(cert.explanation)
    if not parts:
        return (
            "Offline replay accepted the path; the original report did not "
            "record an explicit acceptance narrative"
        )
    return (
        "The original run accepted the path because: " + ' | '.join(parts)
    )


def _explain_revoked(report: RevocationEvidenceReport, exc: RevokedError) -> str:
    for cert in report.certificates:
        if cert.decision == RevocationDecision.REJECTED_REVOKED:
            return (
                "The original run rejected the path: " + cert.explanation
            )
    return (
        "Offline replay found the certificate revoked: "
        f"{exc.reason.human_friendly} at {exc.revocation_dt.isoformat()}"
    )


def _explain_soft_fail(report: RevocationEvidenceReport, exc) -> str:
    bits = []
    for cert in report.certificates:
        if cert.soft_fails:
            sf = ', '.join(
                f"{s.kind.value} {s.exception_type}: {s.message}"
                for s in cert.soft_fails
            )
            bits.append(f"{cert.explanation} ({sf})")
        elif cert.unavailable:
            un = ', '.join(
                f"{u.kind.value}: {u.reason}" for u in cert.unavailable
            )
            bits.append(f"{cert.explanation} ({un})")
    if not bits:
        bits.append(str(getattr(exc, 'failure_msg', exc)))
    return (
        "The original run continued only under soft-fail: "
        + ' | '.join(bits)
    )


def _explain_insufficient(report: RevocationEvidenceReport, exc) -> str:
    bits = []
    for cert in report.certificates:
        if cert.unavailable:
            bits.append(
                f"{cert.explanation} ["
                + ', '.join(
                    f"{u.kind.value}: {u.reason}" for u in cert.unavailable
                )
                + "]"
            )
    lead = (
        "The original run rejected the path for insufficient revocation "
        "information"
    )
    if bits:
        return lead + ': ' + ' | '.join(bits)
    return f"{lead}: {getattr(exc, 'failure_msg', exc)}"
