"""
Data model for revocation evidence reports.

A revocation evidence report records the materials that *actually participated*
in a revocation decision, the policy under which the decision was taken, and
the outcome of the decision. It is intentionally self-describing but
trustless: a report never asserts that its own contents are authentic, and an
offline replay (see :mod:`pyhanko_certvalidator.evidence.replay`) always
re-establishes trust from caller-supplied trust roots and policy.
"""

import base64
import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from asn1crypto import crl, ocsp, x509
from asn1crypto.util import timezone

from ..policy_decl import (
    CertRevTrustPolicy,
    RevocationCheckingPolicy,
    RevocationCheckingRule,
)

__all__ = [
    'EvidenceOutcome',
    'RevocationDecision',
    'MaterialSource',
    'MaterialKind',
    'OCSPCertStatus',
    'MaterialReference',
    'OCSPEvidence',
    'CRLEvidence',
    'UnavailableMaterialEvidence',
    'SoftFailEvidence',
    'CertificateRevocationEvidence',
    'PathEvidence',
    'PolicySnapshot',
    'RevocationEvidenceReport',
    'fingerprint_cert',
    'digest_bytes',
    'raw_material_digest',
    'OCSP_HASH',
    'CRL_HASH',
    'b64e',
    'b64d',
    'dt_to_str',
    'dt_from_str',
]

REPORT_FORMAT_VERSION = 1


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def b64d(data: str) -> bytes:
    return base64.b64decode(data.encode('ascii'))


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_cert(cert: x509.Certificate) -> Dict[str, str]:
    der = cert.dump()
    return {
        'sha256': hashlib.sha256(der).hexdigest(),
        'sha1': hashlib.sha1(der).hexdigest(),
    }


def dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def dt_from_str(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    # tolerate both microsecond and second precision
    fmt = (
        '%Y-%m-%dT%H:%M:%S.%fZ'
        if '.' in value
        else '%Y-%m-%dT%H:%M:%SZ'
    )
    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)


class EvidenceOutcome(enum.Enum):
    """
    Overall revocation decision for one certificate.
    """

    GOOD = 'good'
    """
    At least one valid, in-scope piece of revocation information positively
    established that the certificate was not revoked.
    """

    REVOKED = 'revoked'
    """
    Valid, in-scope revocation information established that the certificate
    was revoked at or before the validation time.
    """

    UNKNOWN = 'unknown'
    """
    No definitive status could be established, and policy allowed processing
    to continue (soft-fail), or the available status was ``unknown``.
    """

    FAILED = 'failed'
    """
    No definitive status could be established, and policy did not allow
    processing to continue.
    """


class RevocationDecision(enum.Enum):
    """
    The decision recorded for a single certificate, after both OCSP and CRL
    processing completed.
    """

    ACCEPTED = 'accepted'
    REJECTED_REVOKED = 'rejected-revoked'
    SOFT_FAIL_CONTINUED = 'soft-fail-continued'
    REJECTED_INSUFFICIENT = 'rejected-insufficient-revinfo'


class MaterialSource(enum.Enum):
    """
    Where a piece of revocation information came from.
    """

    PROVIDED = 'provided'
    """
    Supplied directly by the caller (e.g. via the ``ocsps``/``crls``
    parameters).
    """

    CACHE = 'cache'
    """
    Served from a fetcher's in-memory cache (a previous download within the
    same or an earlier validation run).
    """

    FETCHED = 'fetched'
    """
    Downloaded over the network during the current validation run.

    .. note::
        A successful download is recorded as evidence of provenance only;
        it does not count as valid status evidence until signature, scope,
        freshness and authorisation checks have all passed.
    """


class MaterialKind(enum.Enum):
    OCSP = 'ocsp'
    CRL = 'crl'


class OCSPCertStatus(enum.Enum):
    GOOD = 'good'
    REVOKED = 'revoked'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class MaterialReference:
    """
    Stable reference to a piece of raw revocation material (an OCSP response
    or a CRL), independent of whether it turned out to be valid evidence.
    """

    kind: MaterialKind
    source: MaterialSource
    sha256: str
    """
    Hex-encoded SHA-256 digest of the DER bytes of the complete
    OCSPResponse/CertificateList value that participated in processing.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            'kind': self.kind.value,
            'source': self.source.value,
            'sha256': self.sha256,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'MaterialReference':
        return cls(
            kind=MaterialKind(data['kind']),
            source=MaterialSource(data['source']),
            sha256=data['sha256'],
        )


@dataclass(frozen=True)
class OCSPEvidence:
    """
    Record of one OCSP response that was actually processed for a given
    certificate, including the reason why it did or did not count as valid
    evidence.
    """

    reference: MaterialReference
    cert_id_matched: bool
    """
    Whether the response's CertID (issuer name hash, issuer key hash and
    serial number) matched the certificate and its issuing authority.
    """

    cert_status: Optional[OCSPCertStatus]
    """
    The OCSP cert status, if the response could be parsed far enough to
    determine it. A response signed correctly but targeted at another
    certificate or issuer leaves this as ``None`` alongside
    ``cert_id_matched=False``.
    """

    signature_valid: bool
    responder_authorised: bool
    freshness_rating: str
    """
    The raw usability rating produced by the freshness policy
    (``ok``, ``stale``, ``too-new`` or ``unclear``).
    """

    accepted: bool
    """
    Whether this response actually counted as valid status evidence.
    """

    selection_reason: str
    """
    Human-readable explanation of why the response was accepted or rejected,
    suitable for replay explanations.
    """

    this_update: Optional[datetime] = None
    next_update: Optional[datetime] = None
    revocation_time: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    responder_cert_fingerprints: Optional[Dict[str, str]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            'reference': self.reference.to_json(),
            'cert_id_matched': self.cert_id_matched,
            'cert_status': (
                self.cert_status.value if self.cert_status is not None else None
            ),
            'signature_valid': self.signature_valid,
            'responder_authorised': self.responder_authorised,
            'freshness_rating': self.freshness_rating,
            'accepted': self.accepted,
            'selection_reason': self.selection_reason,
            'this_update': dt_to_str(self.this_update),
            'next_update': dt_to_str(self.next_update),
            'revocation_time': dt_to_str(self.revocation_time),
            'revocation_reason': self.revocation_reason,
            'responder_cert_fingerprints': self.responder_cert_fingerprints,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'OCSPEvidence':
        status = data.get('cert_status')
        return cls(
            reference=MaterialReference.from_json(data['reference']),
            cert_id_matched=data['cert_id_matched'],
            cert_status=(
                OCSPCertStatus(status) if status is not None else None
            ),
            signature_valid=data['signature_valid'],
            responder_authorised=data['responder_authorised'],
            freshness_rating=data['freshness_rating'],
            accepted=data['accepted'],
            selection_reason=data['selection_reason'],
            this_update=dt_from_str(data.get('this_update')),
            next_update=dt_from_str(data.get('next_update')),
            revocation_time=dt_from_str(data.get('revocation_time')),
            revocation_reason=data.get('revocation_reason'),
            responder_cert_fingerprints=data.get('responder_cert_fingerprints'),
        )


@dataclass(frozen=True)
class CRLEvidence:
    """
    Record of one CRL that was actually processed for a given certificate.
    """

    reference: MaterialReference
    issuer_matched: bool
    """
    Whether the CRL's issuing authority (name, key and IDP/DP scope) matched
    the certificate's issuer.
    """

    scope_reasons: List[str]
    """
    Revocation reasons covered by the CRL for the certificate, after IDP
    processing. Empty if the CRL had no jurisdiction.
    """

    signature_valid: bool
    issuer_path_validated: bool
    freshness_rating: str
    cert_listed: bool
    """
    Whether the certificate's serial number appeared in the CRL (or delta).
    """

    accepted: bool
    """
    Whether this CRL actually counted as valid status evidence.
    """

    selection_reason: str

    this_update: Optional[datetime] = None
    next_update: Optional[datetime] = None
    revocation_time: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    issuer_cert_fingerprints: Optional[Dict[str, str]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            'reference': self.reference.to_json(),
            'issuer_matched': self.issuer_matched,
            'scope_reasons': list(self.scope_reasons),
            'signature_valid': self.signature_valid,
            'issuer_path_validated': self.issuer_path_validated,
            'freshness_rating': self.freshness_rating,
            'cert_listed': self.cert_listed,
            'accepted': self.accepted,
            'selection_reason': self.selection_reason,
            'this_update': dt_to_str(self.this_update),
            'next_update': dt_to_str(self.next_update),
            'revocation_time': dt_to_str(self.revocation_time),
            'revocation_reason': self.revocation_reason,
            'issuer_cert_fingerprints': self.issuer_cert_fingerprints,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'CRLEvidence':
        return cls(
            reference=MaterialReference.from_json(data['reference']),
            issuer_matched=data['issuer_matched'],
            scope_reasons=list(data.get('scope_reasons', ())),
            signature_valid=data['signature_valid'],
            issuer_path_validated=data['issuer_path_validated'],
            freshness_rating=data['freshness_rating'],
            cert_listed=data['cert_listed'],
            accepted=data['accepted'],
            selection_reason=data['selection_reason'],
            this_update=dt_from_str(data.get('this_update')),
            next_update=dt_from_str(data.get('next_update')),
            revocation_time=dt_from_str(data.get('revocation_time')),
            revocation_reason=data.get('revocation_reason'),
            issuer_cert_fingerprints=data.get('issuer_cert_fingerprints'),
        )


@dataclass(frozen=True)
class UnavailableMaterialEvidence:
    """
    Record that revocation information of a given kind was expected but could
    not be obtained or made usable.
    """

    kind: MaterialKind
    reason: str
    detail: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            'kind': self.kind.value,
            'reason': self.reason,
            'detail': self.detail,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'UnavailableMaterialEvidence':
        return cls(
            kind=MaterialKind(data['kind']),
            reason=data['reason'],
            detail=data.get('detail'),
        )


@dataclass(frozen=True)
class SoftFailEvidence:
    """
    Record of one soft-fail event: a failure that policy explicitly allowed
    the validator to ignore.
    """

    kind: MaterialKind
    exception_type: str
    message: str

    def to_json(self) -> Dict[str, Any]:
        return {
            'kind': self.kind.value,
            'exception_type': self.exception_type,
            'message': self.message,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SoftFailEvidence':
        return cls(
            kind=MaterialKind(data['kind']),
            exception_type=data['exception_type'],
            message=data['message'],
        )


@dataclass(frozen=True)
class CertificateRevocationEvidence:
    """
    Complete revocation evidence record for one certificate within a
    validated path.
    """

    cert_fingerprints: Dict[str, str]
    issuer_fingerprints: Dict[str, str]
    serial_number: int
    is_ee_cert: bool
    outcome: EvidenceOutcome
    decision: RevocationDecision
    explanation: str
    """
    Concise narrative of why the certificate was accepted, rejected or merely
    allowed to continue under soft-fail.
    """

    ocsp_evidence: List[OCSPEvidence] = field(default_factory=list)
    crl_evidence: List[CRLEvidence] = field(default_factory=list)
    unavailable: List[UnavailableMaterialEvidence] = field(default_factory=list)
    soft_fails: List[SoftFailEvidence] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            'cert_fingerprints': self.cert_fingerprints,
            'issuer_fingerprints': self.issuer_fingerprints,
            'serial_number': self.serial_number,
            'is_ee_cert': self.is_ee_cert,
            'outcome': self.outcome.value,
            'decision': self.decision.value,
            'explanation': self.explanation,
            'ocsp_evidence': [e.to_json() for e in self.ocsp_evidence],
            'crl_evidence': [e.to_json() for e in self.crl_evidence],
            'unavailable': [e.to_json() for e in self.unavailable],
            'soft_fails': [e.to_json() for e in self.soft_fails],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'CertificateRevocationEvidence':
        return cls(
            cert_fingerprints=dict(data['cert_fingerprints']),
            issuer_fingerprints=dict(data['issuer_fingerprints']),
            serial_number=int(data['serial_number']),
            is_ee_cert=bool(data['is_ee_cert']),
            outcome=EvidenceOutcome(data['outcome']),
            decision=RevocationDecision(data['decision']),
            explanation=data['explanation'],
            ocsp_evidence=[
                OCSPEvidence.from_json(e) for e in data.get('ocsp_evidence', [])
            ],
            crl_evidence=[
                CRLEvidence.from_json(e) for e in data.get('crl_evidence', [])
            ],
            unavailable=[
                UnavailableMaterialEvidence.from_json(e)
                for e in data.get('unavailable', [])
            ],
            soft_fails=[
                SoftFailEvidence.from_json(e)
                for e in data.get('soft_fails', [])
            ],
        )


@dataclass(frozen=True)
class PathEvidence:
    """
    The chain that actually participated in the decision, identified by
    fingerprints. Only certificates on a path that was validated are listed.
    """

    trust_anchor_fingerprints: Dict[str, str]
    cert_fingerprints: List[Dict[str, str]]
    """
    Fingerprints of the certificates on the path, trust anchor first and
    end-entity certificate last.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            'trust_anchor_fingerprints': self.trust_anchor_fingerprints,
            'cert_fingerprints': [dict(f) for f in self.cert_fingerprints],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PathEvidence':
        return cls(
            trust_anchor_fingerprints=dict(data['trust_anchor_fingerprints']),
            cert_fingerprints=[dict(f) for f in data['cert_fingerprints']],
        )


@dataclass(frozen=True)
class PolicySnapshot:
    """
    Machine-readable record of the revocation policy that governed the
    decision. Replay requires the caller to supply an equivalent policy
    explicitly; this snapshot is documentation, not an authorisation token.
    """

    revocation_mode_legacy: Optional[str]
    ee_rule: str
    intermediate_rule: str
    retroactive_revinfo: bool
    time_tolerance_seconds: float
    freshness_req_type: Optional[str]
    freshness_seconds: Optional[float]

    def to_json(self) -> Dict[str, Any]:
        return {
            'revocation_mode_legacy': self.revocation_mode_legacy,
            'ee_rule': self.ee_rule,
            'intermediate_rule': self.intermediate_rule,
            'retroactive_revinfo': self.retroactive_revinfo,
            'time_tolerance_seconds': self.time_tolerance_seconds,
            'freshness_req_type': self.freshness_req_type,
            'freshness_seconds': self.freshness_seconds,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PolicySnapshot':
        return cls(
            revocation_mode_legacy=data.get('revocation_mode_legacy'),
            ee_rule=data['ee_rule'],
            intermediate_rule=data['intermediate_rule'],
            retroactive_revinfo=bool(data['retroactive_revinfo']),
            time_tolerance_seconds=float(data['time_tolerance_seconds']),
            freshness_req_type=data.get('freshness_req_type'),
            freshness_seconds=data.get('freshness_seconds'),
        )

    @classmethod
    def from_policy(
        cls,
        policy: CertRevTrustPolicy,
        time_tolerance_seconds: float,
        legacy_mode: Optional[str] = None,
    ) -> 'PolicySnapshot':
        checking: RevocationCheckingPolicy = policy.revocation_checking_policy
        freshness = policy.freshness
        return cls(
            revocation_mode_legacy=legacy_mode,
            ee_rule=checking.ee_certificate_rule.value,
            intermediate_rule=checking.intermediate_ca_cert_rule.value,
            retroactive_revinfo=policy.retroactive_revinfo,
            time_tolerance_seconds=time_tolerance_seconds,
            freshness_req_type=(
                policy.freshness_req_type.name.lower()
                if policy.freshness_req_type is not None
                else None
            ),
            freshness_seconds=(
                freshness.total_seconds() if freshness is not None else None
            ),
        )

    def matches_policy(
        self,
        policy: CertRevTrustPolicy,
        time_tolerance_seconds: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check whether a caller-supplied policy is equivalent to the one
        recorded in the report.

        :return:
            A tuple ``(matches, discrepancies)``.
        """

        discrepancies: List[str] = []
        checking: RevocationCheckingPolicy = policy.revocation_checking_policy
        expected = (
            (RevocationCheckingRule(self.ee_rule), checking.ee_certificate_rule),
            (
                RevocationCheckingRule(self.intermediate_rule),
                checking.intermediate_ca_cert_rule,
            ),
        )
        for label, (recorded, actual) in zip(
            ('ee_rule', 'intermediate_rule'), expected
        ):
            if recorded != actual:
                discrepancies.append(
                    f"{label}: report recorded {recorded.value}, "
                    f"replay supplied {actual.value}"
                )
        if self.retroactive_revinfo != policy.retroactive_revinfo:
            discrepancies.append(
                "retroactive_revinfo does not match recorded policy"
            )
        if abs(self.time_tolerance_seconds - time_tolerance_seconds) > 1e-9:
            discrepancies.append(
                "time_tolerance does not match recorded policy"
            )
        recorded_freshness_type = self.freshness_req_type
        actual_freshness_type = (
            policy.freshness_req_type.name.lower()
            if policy.freshness_req_type is not None
            else None
        )
        if recorded_freshness_type != actual_freshness_type:
            discrepancies.append(
                "freshness_req_type does not match recorded policy"
            )
        actual_freshness = (
            policy.freshness.total_seconds()
            if policy.freshness is not None
            else None
        )
        if self.freshness_seconds != actual_freshness:
            discrepancies.append(
                "freshness delta does not match recorded policy"
            )
        return not discrepancies, discrepancies


@dataclass(frozen=True)
class RevocationEvidenceReport:
    """
    Top-level evidence report for one (possibly multi-certificate) path
    validation run.
    """

    format_version: int
    validation_time: datetime
    best_signature_time: datetime
    point_in_time_validation: bool
    fetching_allowed: bool
    policy: PolicySnapshot
    path: Optional[PathEvidence]
    certificates: List[CertificateRevocationEvidence]
    materials: Dict[str, MaterialReference] = field(default_factory=dict)
    """
    Index of every raw material that participated, keyed by its SHA-256 hex
    digest, including materials that were rejected. The raw bytes themselves
    are kept separately by the caller so that reports remain shareable
    without leaking key material or bulky payloads.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            'format_version': self.format_version,
            'validation_time': dt_to_str(self.validation_time),
            'best_signature_time': dt_to_str(self.best_signature_time),
            'point_in_time_validation': self.point_in_time_validation,
            'fetching_allowed': self.fetching_allowed,
            'policy': self.policy.to_json(),
            'path': None if self.path is None else self.path.to_json(),
            'certificates': [c.to_json() for c in self.certificates],
            'materials': {k: v.to_json() for k, v in self.materials.items()},
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'RevocationEvidenceReport':
        if int(data.get('format_version', -1)) != REPORT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported evidence report format version: "
                f"{data.get('format_version')!r}"
            )
        path_data = data.get('path')
        return cls(
            format_version=int(data['format_version']),
            validation_time=dt_from_str(data['validation_time']),
            best_signature_time=dt_from_str(data['best_signature_time']),
            point_in_time_validation=bool(data['point_in_time_validation']),
            fetching_allowed=bool(data['fetching_allowed']),
            policy=PolicySnapshot.from_json(data['policy']),
            path=(
                None if path_data is None else PathEvidence.from_json(path_data)
            ),
            certificates=[
                CertificateRevocationEvidence.from_json(c)
                for c in data.get('certificates', [])
            ],
            materials={
                k: MaterialReference.from_json(v)
                for k, v in data.get('materials', {}).items()
            },
        )


def container_digest_ocsp(ocsp_response: ocsp.OCSPResponse) -> str:
    return digest_bytes(ocsp_response.dump())


def container_digest_crl(certificate_list: crl.CertificateList) -> str:
    return digest_bytes(certificate_list.dump())


# short aliases used by the collector
OCSP_HASH = container_digest_ocsp
CRL_HASH = container_digest_crl


def raw_material_digest(kind: MaterialKind, data: bytes) -> str:
    return digest_bytes(data)
