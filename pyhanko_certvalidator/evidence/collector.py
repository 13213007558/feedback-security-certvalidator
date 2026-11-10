"""
Runtime collector for revocation evidence.

A :class:`EvidenceCollector` is attached to a :class:`ValidationContext` and
fed structured observations while the normal revocation checks run. It keeps
track of which raw materials participated, where they came from, which checks
they passed or failed, and how the final decision was reached. Collectors are
strictly additive: enabling evidence collection never changes validation
behaviour, and disabled contexts simply do not instantiate one.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from asn1crypto import x509

from ..path import ValidationPath
from ..policy_decl import CertRevTrustPolicy
from ..revinfo.archival import CRLContainer, OCSPContainer
from .model import (
    CRL_HASH,
    OCSP_HASH,
    CRLEvidence,
    CertificateRevocationEvidence,
    EvidenceOutcome,
    MaterialKind,
    MaterialReference,
    MaterialSource,
    OCSPCertStatus,
    OCSPEvidence,
    PathEvidence,
    PolicySnapshot,
    REPORT_FORMAT_VERSION,
    RevocationDecision,
    RevocationEvidenceReport,
    SoftFailEvidence,
    UnavailableMaterialEvidence,
    fingerprint_cert,
)


@dataclass
class _CertRecord:
    fingerprints: Dict[str, str]
    issuer_fingerprints: Dict[str, str]
    serial_number: int
    is_ee_cert: bool
    ocsp_evidence: List[OCSPEvidence] = field(default_factory=list)
    crl_evidence: List[CRLEvidence] = field(default_factory=list)
    unavailable: List[UnavailableMaterialEvidence] = field(default_factory=list)
    soft_fails: List[SoftFailEvidence] = field(default_factory=list)
    seen_materials: set = field(default_factory=set)


class EvidenceCollector:
    """
    Mutable collector that accumulates revocation evidence during a single
    validation run (i.e. for the lifetime of one
    :class:`~pyhanko_certvalidator.context.ValidationContext`).
    """

    def __init__(
        self,
        policy: CertRevTrustPolicy,
        validation_time: datetime,
        best_signature_time: datetime,
        point_in_time_validation: bool,
        time_tolerance: timedelta,
        fetching_allowed: bool,
        legacy_revocation_mode: Optional[str] = None,
    ):
        self._policy_snapshot = PolicySnapshot.from_policy(
            policy,
            time_tolerance_seconds=time_tolerance.total_seconds(),
            legacy_mode=legacy_revocation_mode,
        )
        self._validation_time = validation_time
        self._best_signature_time = best_signature_time
        self._point_in_time = point_in_time_validation
        self._fetching_allowed = fetching_allowed
        self._records: Dict[bytes, _CertRecord] = {}
        self._finalised: Dict[bytes, CertificateRevocationEvidence] = {}
        self._materials: Dict[str, MaterialReference] = {}
        self._path: Optional[PathEvidence] = None
        self._path_certs: List[x509.Certificate] = []

    # ----- material bookkeeping -------------------------------------------

    def _register_material(
        self, kind: MaterialKind, digest_hex: str, source: MaterialSource
    ) -> MaterialReference:
        try:
            ref = self._materials[digest_hex]
            # a material first seen as provided/cached keeps its original
            # provenance; a later fetch of the same bytes does not rewrite it
            return ref
        except KeyError:
            ref = MaterialReference(kind=kind, source=source, sha256=digest_hex)
            self._materials[digest_hex] = ref
            return ref

    def _record_for(
        self,
        cert: x509.Certificate,
        issuer: x509.Certificate,
        is_ee_cert: bool,
    ) -> _CertRecord:
        key = cert.sha256
        try:
            return self._records[key]
        except KeyError:
            record = _CertRecord(
                fingerprints=fingerprint_cert(cert),
                issuer_fingerprints=fingerprint_cert(issuer),
                serial_number=cert.serial_number,
                is_ee_cert=is_ee_cert,
            )
            self._records[key] = record
            return record

    # ----- path ------------------------------------------------------------

    def record_path(self, path: ValidationPath):
        """
        Record the path that was actually validated. Called once per
        successful top-level path validation; subsequent calls replace the
        previous record since only one path can end up accepted.
        """

        certs: List[x509.Certificate] = []
        anchor = path.trust_anchor
        anchor_cert = getattr(anchor, 'certificate', None)
        if isinstance(anchor_cert, x509.Certificate):
            certs.append(anchor_cert)
        certs.extend(c for c in path.iter_certs(include_root=False))
        self._path_certs = certs
        anchor_fp = (
            fingerprint_cert(anchor_cert)
            if isinstance(anchor_cert, x509.Certificate)
            else {'sha256': '', 'sha1': ''}
        )
        self._path = PathEvidence(
            trust_anchor_fingerprints=anchor_fp,
            cert_fingerprints=[fingerprint_cert(c) for c in certs],
        )

    # ----- OCSP observations ----------------------------------------------

    def record_ocsp_attempt(
        self,
        cert: x509.Certificate,
        issuer: x509.Certificate,
        is_ee_cert: bool,
        container: OCSPContainer,
        source: MaterialSource,
        *,
        cert_id_matched: bool,
        cert_status: Optional[OCSPCertStatus],
        signature_valid: bool,
        responder_authorised: bool,
        freshness_rating: str,
        accepted: bool,
        selection_reason: str,
        responder_cert: Optional[x509.Certificate] = None,
    ):
        digest_hex = OCSP_HASH(container.ocsp_response_data)
        ref = self._register_material(MaterialKind.OCSP, digest_hex, source)
        record = self._record_for(cert, issuer, is_ee_cert)
        if digest_hex in record.seen_materials:
            return
        record.seen_materials.add(digest_hex)

        single = container.extract_single_response()
        this_update = next_update = revocation_time = None
        revocation_reason = None
        if single is not None:
            this_update = single['this_update'].native
            next_update = single['next_update'].native
            if cert_status == OCSPCertStatus.REVOKED:
                revoked_info = single['cert_status'].chosen
                revocation_time = revoked_info['revocation_time'].native
                reason = revoked_info['revocation_reason']
                revocation_reason = (
                    reason.native if reason.native is not None else 'unspecified'
                )

        record.ocsp_evidence.append(
            OCSPEvidence(
                reference=ref,
                cert_id_matched=cert_id_matched,
                cert_status=cert_status,
                signature_valid=signature_valid,
                responder_authorised=responder_authorised,
                freshness_rating=freshness_rating,
                accepted=accepted,
                selection_reason=selection_reason,
                this_update=this_update,
                next_update=next_update,
                revocation_time=revocation_time,
                revocation_reason=revocation_reason,
                responder_cert_fingerprints=(
                    None
                    if responder_cert is None
                    else fingerprint_cert(responder_cert)
                ),
            )
        )

    # ----- CRL observations ------------------------------------------------

    def record_crl_attempt(
        self,
        cert: x509.Certificate,
        issuer: x509.Certificate,
        is_ee_cert: bool,
        container: CRLContainer,
        source: MaterialSource,
        *,
        issuer_matched: bool,
        scope_reasons: List[str],
        signature_valid: bool,
        issuer_path_validated: bool,
        freshness_rating: str,
        cert_listed: bool,
        accepted: bool,
        selection_reason: str,
        revocation_time: Optional[datetime] = None,
        revocation_reason: Optional[str] = None,
        crl_issuer_cert: Optional[x509.Certificate] = None,
    ):
        digest_hex = CRL_HASH(container.crl_data)
        ref = self._register_material(MaterialKind.CRL, digest_hex, source)
        record = self._record_for(cert, issuer, is_ee_cert)
        if digest_hex in record.seen_materials:
            return
        record.seen_materials.add(digest_hex)

        tbs = container.crl_data['tbs_cert_list']
        record.crl_evidence.append(
            CRLEvidence(
                reference=ref,
                issuer_matched=issuer_matched,
                scope_reasons=list(scope_reasons),
                signature_valid=signature_valid,
                issuer_path_validated=issuer_path_validated,
                freshness_rating=freshness_rating,
                cert_listed=cert_listed,
                accepted=accepted,
                selection_reason=selection_reason,
                this_update=tbs['this_update'].native,
                next_update=tbs['next_update'].native,
                revocation_time=revocation_time,
                revocation_reason=revocation_reason,
                issuer_cert_fingerprints=(
                    None
                    if crl_issuer_cert is None
                    else fingerprint_cert(crl_issuer_cert)
                ),
            )
        )

    # ----- unavailability / soft-fail --------------------------------------

    def record_unavailable(
        self,
        cert: x509.Certificate,
        issuer: x509.Certificate,
        is_ee_cert: bool,
        kind: MaterialKind,
        reason: str,
        detail: Optional[str] = None,
    ):
        record = self._record_for(cert, issuer, is_ee_cert)
        record.unavailable.append(
            UnavailableMaterialEvidence(kind=kind, reason=reason, detail=detail)
        )

    def record_soft_fail(
        self,
        cert: x509.Certificate,
        issuer: x509.Certificate,
        is_ee_cert: bool,
        kind: MaterialKind,
        exception: BaseException,
    ):
        record = self._record_for(cert, issuer, is_ee_cert)
        record.soft_fails.append(
            SoftFailEvidence(
                kind=kind,
                exception_type=type(exception).__name__,
                message=str(exception.args[0]) if exception.args else '',
            )
        )

    # ----- finalisation ----------------------------------------------------

    def finalize_cert(
        self,
        cert: x509.Certificate,
        issuer: x509.Certificate,
        is_ee_cert: bool,
        outcome: EvidenceOutcome,
        decision: RevocationDecision,
        explanation: str,
    ) -> CertificateRevocationEvidence:
        record = self._record_for(cert, issuer, is_ee_cert)
        entry = CertificateRevocationEvidence(
            cert_fingerprints=record.fingerprints,
            issuer_fingerprints=record.issuer_fingerprints,
            serial_number=record.serial_number,
            is_ee_cert=record.is_ee_cert,
            outcome=outcome,
            decision=decision,
            explanation=explanation,
            ocsp_evidence=list(record.ocsp_evidence),
            crl_evidence=list(record.crl_evidence),
            unavailable=list(record.unavailable),
            soft_fails=list(record.soft_fails),
        )
        self._finalised[cert.sha256] = entry
        return entry

    def take_finalised(
        self, cert_sha256: bytes
    ) -> Optional[CertificateRevocationEvidence]:
        return self._finalised.get(cert_sha256)

    def build_report(
        self,
        per_cert_outcomes: Dict[bytes, CertificateRevocationEvidence],
    ) -> RevocationEvidenceReport:
        ordered: List[CertificateRevocationEvidence] = []
        if self._path_certs:
            for path_cert in self._path_certs:
                entry = per_cert_outcomes.get(path_cert.sha256)
                if entry is not None:
                    ordered.append(entry)
            seen = {e.cert_fingerprints['sha256'] for e in ordered}
            for digest, entry in per_cert_outcomes.items():
                if entry.cert_fingerprints['sha256'] not in seen:
                    ordered.append(entry)
        else:
            ordered = list(per_cert_outcomes.values())

        return RevocationEvidenceReport(
            format_version=REPORT_FORMAT_VERSION,
            validation_time=self._validation_time,
            best_signature_time=self._best_signature_time,
            point_in_time_validation=self._point_in_time,
            fetching_allowed=self._fetching_allowed,
            policy=self._policy_snapshot,
            path=self._path,
            certificates=ordered,
            materials=dict(self._materials),
        )

    @property
    def validation_time(self) -> datetime:
        return self._validation_time
