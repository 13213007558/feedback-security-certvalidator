"""
Helpers for revocation evidence tests.

These helpers generate a small two-CA PKI entirely in memory with fixed
validity windows, together with hand-built OCSP responses and CRLs. Nothing
here is imported by the library itself; it exists so that evidence tests can
exercise precise timing scenarios (expired OCSP, not-yet-effective
revocation, same serial under a different issuer, missing material).
"""

from datetime import datetime
from typing import List, Optional, Tuple

from asn1crypto import algos, core, crl as asn1_crl, ocsp as asn1_ocsp, x509
from cryptography import x509 as cx509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import (
    AuthorityInformationAccessOID,
    ExtendedKeyUsageOID,
    NameOID,
)

UTC = timezone = __import__('datetime').timezone.utc

# fixed timeline so tests are deterministic
NOT_BEFORE = datetime(2020, 1, 1, tzinfo=UTC)
NOT_AFTER = datetime(2030, 1, 1, tzinfo=UTC)
VALIDATION_MOMENT = datetime(2021, 6, 1, 12, 0, 0, tzinfo=UTC)
LEAF_SERIAL = 123456789


def _new_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _name(cn: str) -> cx509.Name:
    return cx509.Name([cx509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _build_ca(cn: str, key) -> Tuple[x509.Certificate, cx509.Name]:
    subject = issuer = _name(cn)
    builder = (
        cx509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(cx509.BasicConstraints(ca=True, path_length=None), True)
        .add_extension(
            cx509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_encipherment=False,
                encipher_only=False,
                decipher_only=False,
            ),
            True,
        )
        .add_extension(
            cx509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            False,
        )
    )
    signed = builder.sign(key, hashes.SHA256())
    return (
        x509.Certificate.load(signed.public_bytes(serialization.Encoding.DER)),
        subject,
    )


def _build_leaf(
    cn: str,
    leaf_key,
    issuer_name: cx509.Name,
    issuer_key,
    serial: int,
    *,
    with_ocsp_aia: bool = True,
    with_crl_dp: bool = True,
) -> x509.Certificate:
    builder = (
        cx509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(issuer_name)
        .public_key(leaf_key.public_key())
        .serial_number(serial)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(cx509.BasicConstraints(ca=False, path_length=None), True)
        .add_extension(
            cx509.KeyUsage(
                digital_signature=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_encipherment=False,
                encipher_only=False,
                decipher_only=False,
            ),
            True,
        )
    )
    if with_ocsp_aia:
        aia = cx509.AuthorityInformationAccess(
            [
                cx509.AccessDescription(
                    access_method=AuthorityInformationAccessOID.OCSP,
                    access_location=cx509.UniformResourceIdentifier(
                        "http://ocsp.example.test"
                    ),
                )
            ]
        )
        builder = builder.add_extension(aia, False)
    if with_crl_dp:
        dp = cx509.DistributionPoint(
            full_name=[
                cx509.UniformResourceIdentifier(
                    "http://crl.example-test/ca.crl"
                )
            ],
            relative_name=None,
            reasons=None,
            crl_issuer=None,
        )
        builder = builder.add_extension(
            cx509.CRLDistributionPoints([dp]), False
        )
    signed = builder.sign(issuer_key, hashes.SHA256())
    return x509.Certificate.load(signed.public_bytes(serialization.Encoding.DER))


def build_pki(
    leaf_serial: int = LEAF_SERIAL,
    *,
    with_ocsp_aia: bool = True,
    with_crl_dp: bool = True,
) -> dict:
    """
    Build two independent CA trees. Both leaves share the same serial number
    but have different issuers, which is what makes a "same serial, different
    issuer" OCSP response unambiguously out of scope.
    """

    ca_a_key = _new_key()
    ca_b_key = _new_key()
    leaf_a_key = _new_key()
    leaf_b_key = _new_key()

    ca_a, ca_a_name = _build_ca("Evidence Test CA A", ca_a_key)
    ca_b, ca_b_name = _build_ca("Evidence Test CA B", ca_b_key)
    leaf_a = _build_leaf(
        "evidence-leaf-a",
        leaf_a_key,
        ca_a_name,
        ca_a_key,
        leaf_serial,
        with_ocsp_aia=with_ocsp_aia,
        with_crl_dp=with_crl_dp,
    )
    leaf_b = _build_leaf(
        "evidence-leaf-b",
        leaf_b_key,
        ca_b_name,
        ca_b_key,
        leaf_serial,
        with_ocsp_aia=with_ocsp_aia,
        with_crl_dp=with_crl_dp,
    )
    return {
        'ca_a': ca_a,
        'ca_a_name': ca_a_name,
        'ca_a_key': ca_a_key,
        'ca_b': ca_b,
        'ca_b_name': ca_b_name,
        'ca_b_key': ca_b_key,
        'leaf_a': leaf_a,
        'leaf_a_key': leaf_a_key,
        'leaf_b': leaf_b,
        'leaf_b_key': leaf_b_key,
    }


# --------------------------------------------------------------------------
# OCSP
# --------------------------------------------------------------------------


def _cert_id(
    target_cert: x509.Certificate, issuer_cert: x509.Certificate, algo='sha1'
) -> asn1_ocsp.CertId:
    return asn1_ocsp.CertId(
        {
            'hash_algorithm': algos.DigestAlgorithm({'algorithm': algo}),
            'issuer_name_hash': getattr(
                target_cert.issuer, algo
            ),
            'issuer_key_hash': getattr(issuer_cert.public_key, algo),
            'serial_number': target_cert.serial_number,
        }
    )


def build_ocsp_response(
    *,
    target_cert: x509.Certificate,
    issuer_cert: x509.Certificate,
    issuer_key,
    status: str,
    this_update: datetime,
    next_update: Optional[datetime],
    revocation_time: Optional[datetime] = None,
    revocation_reason: Optional[str] = None,
    produced_at: Optional[datetime] = None,
    responder_cert: Optional[x509.Certificate] = None,
    responder_key=None,
    responder_id_by_key: bool = False,
) -> asn1_ocsp.OCSPResponse:
    """
    Construct and sign a basic OCSP response.

    ``status`` is one of ``good``, ``revoked`` or ``unknown``.
    """

    cert_id = _cert_id(target_cert, issuer_cert)
    if status == 'good':
        cert_status = asn1_ocsp.CertStatus(
            name='good', value=asn1_ocsp.StatusGood()
        )
    elif status == 'unknown':
        cert_status = asn1_ocsp.CertStatus(
            name='unknown', value=asn1_ocsp.StatusUnknown()
        )
    elif status == 'revoked':
        if revocation_time is None:
            raise ValueError("revoked status requires revocation_time")
        reason = asn1_crl.CRLReason(revocation_reason or 'unspecified')
        revoked_info = asn1_ocsp.RevokedInfo(
            {
                'revocation_time': revocation_time,
                'revocation_reason': reason,
            }
        )
        cert_status = asn1_ocsp.CertStatus(name='revoked', value=revoked_info)
    else:
        raise ValueError(f"unknown OCSP status {status!r}")

    single = asn1_ocsp.SingleResponse(
        {
            'cert_id': cert_id,
            'cert_status': cert_status,
            'this_update': this_update,
            'next_update': next_update,
        }
    )

    if responder_cert is None:
        responder_cert = issuer_cert
        responder_key = issuer_key
        responder_name = issuer_cert.subject
        responder_sk = None
    else:
        responder_name = responder_cert.subject
        responder_sk = responder_cert.key_identifier

    if responder_id_by_key and responder_sk is not None:
        responder_id = asn1_ocsp.ResponderId(
            name='by_key', value=core.OctetString(responder_sk)
        )
    else:
        responder_id = asn1_ocsp.ResponderId(
            name='by_name', value=responder_name
        )

    tbs = asn1_ocsp.ResponseData(
        {
            'version': 0,
            'responder_id': responder_id,
            'produced_at': produced_at or this_update,
            'responses': [single],
        }
    )
    signature = responder_key.sign(
        tbs.dump(), padding.PKCS1v15(), hashes.SHA256()
    )
    basic = asn1_ocsp.BasicOCSPResponse(
        {
            'tbs_response_data': tbs,
            'signature_algorithm': algos.SignedDigestAlgorithm(
                {'algorithm': 'sha256_rsa'}
            ),
            'signature': core.OctetBitString(signature),
        }
    )
    return asn1_ocsp.OCSPResponse(
        {
            'response_status': 'successful',
            'response_bytes': asn1_ocsp.ResponseBytes(
                {
                    'response_type': 'basic_ocsp_response',
                    'response': core.ParsableOctetString(basic.dump()),
                }
            ),
        }
    )


def build_delegated_responder_cert(
    cn: str, issuer_name: cx509.Name, issuer_cert: x509.Certificate, issuer_key
) -> Tuple[x509.Certificate, object]:
    key = _new_key()
    builder = (
        cx509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(999)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(
            cx509.ExtendedKeyUsage([ExtendedKeyUsageOID.OCSP_SIGNING]),
            critical=True,
        )
        .add_extension(
            cx509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            False,
        )
        .add_extension(
            cx509.AuthorityKeyIdentifier.from_issuer_public_key(
                issuer_key.public_key()
            ),
            False,
        )
    )
    signed = builder.sign(issuer_key, hashes.SHA256())
    return (
        x509.Certificate.load(signed.public_bytes(serialization.Encoding.DER)),
        key,
    )


# --------------------------------------------------------------------------
# CRL
# --------------------------------------------------------------------------


def build_crl(
    *,
    issuer_cert: x509.Certificate,
    issuer_key,
    this_update: datetime,
    next_update: Optional[datetime],
    entries: Optional[List[dict]] = None,
) -> asn1_crl.CertificateList:
    """
    Build a v2 CRL. Each entry in ``entries`` is a dict with keys
    ``serial`` (int), ``revocation_date`` (datetime) and optionally
    ``reason`` (str).
    """

    revoked = []
    for entry in entries or ():
        rev_cert = asn1_crl.RevokedCertificate(
            {
                'user_certificate': entry['serial'],
                'revocation_date': asn1_crl.Time(
                    name='utc_time'
                    if entry['revocation_date'].year < 2050
                    else 'general_time',
                    value=entry['revocation_date'],
                ),
            }
        )
        if entry.get('reason'):
            reason_ext = asn1_crl.CRLEntryExtension(
                {
                    'extn_id': 'crl_reason',
                    'critical': False,
                    'extn_value': entry['reason'],
                }
            )
            rev_cert['crl_entry_extensions'] = [reason_ext]
        revoked.append(rev_cert)

    tbs_kwargs = {
        'version': 2,
        'signature': algos.SignedDigestAlgorithm(
            {'algorithm': 'sha256_rsa'}
        ),
        'issuer': issuer_cert.subject,
        'this_update': asn1_crl.Time(
            name=(
                'utc_time' if this_update.year < 2050 else 'general_time'
            ),
            value=this_update,
        ),
    }
    if next_update is not None:
        tbs_kwargs['next_update'] = asn1_crl.Time(
            name=(
                'utc_time' if next_update.year < 2050 else 'general_time'
            ),
            value=next_update,
        )
    if revoked:
        tbs_kwargs['revoked_certificates'] = revoked
    tbs = asn1_crl.TbsCertList(tbs_kwargs)
    signature = issuer_key.sign(
        tbs.dump(), padding.PKCS1v15(), hashes.SHA256()
    )
    return asn1_crl.CertificateList(
        {
            'tbs_cert_list': tbs,
            'signature_algorithm': algos.SignedDigestAlgorithm(
                {'algorithm': 'sha256_rsa'}
            ),
            'signature': core.OctetBitString(signature),
        }
    )
