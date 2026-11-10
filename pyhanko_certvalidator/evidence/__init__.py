"""
Optional revocation evidence reporting and offline replay support.

.. versionadded:: 0.27.0
"""

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
    b64d,
    b64e,
    digest_bytes,
    dt_from_str,
    dt_to_str,
    fingerprint_cert,
)

# NOTE: collector and replay pull in validation machinery and are imported
# lazily to avoid circular imports (the revinfo manager only needs model).
__all__ = [
    'EvidenceCollector',
    'EvidenceOutcome',
    'RevocationDecision',
    'MaterialKind',
    'MaterialSource',
    'MaterialReference',
    'OCSPCertStatus',
    'OCSPEvidence',
    'CRLEvidence',
    'UnavailableMaterialEvidence',
    'SoftFailEvidence',
    'CertificateRevocationEvidence',
    'PathEvidence',
    'PolicySnapshot',
    'RevocationEvidenceReport',
    'ReplayOutcome',
    'EvidenceReplayError',
    'OfflineReplaySecurityError',
    'REPORT_FORMAT_VERSION',
    'replay_validation_offline',
    'build_report_from_context',
    'report_to_json',
    'report_from_json',
    'fingerprint_cert',
    'digest_bytes',
    'b64e',
    'b64d',
    'dt_to_str',
    'dt_from_str',
    'OCSP_HASH',
    'CRL_HASH',
]


def __getattr__(name):
    if name in (
        'EvidenceCollector',
        'replay_validation_offline',
        'build_report_from_context',
        'ReplayOutcome',
        'EvidenceReplayError',
        'OfflineReplaySecurityError',
    ):
        if name == 'EvidenceCollector':
            from .collector import EvidenceCollector as value
        else:
            from . import replay as _replay

            value = getattr(_replay, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def report_to_json(report: RevocationEvidenceReport) -> dict:
    """Serialize an evidence report to JSON-compatible data."""
    return report.to_json()


def report_from_json(data: dict) -> RevocationEvidenceReport:
    """Parse an evidence report from JSON-compatible data."""
    return RevocationEvidenceReport.from_json(data)
