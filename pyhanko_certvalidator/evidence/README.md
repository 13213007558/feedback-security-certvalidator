# Revocation evidence reporting

This subpackage optionally records *why* a revocation decision came out the
way it did, and supports replaying that decision offline.

## Recording

Enable collection on the validation context; validation behaviour is
otherwise unchanged:

```python
context = ValidationContext(
    trust_roots=roots,
    ocsps=ocsp_responses,
    crls=crls,
    moment=validation_time,
    collect_evidence=True,
)
await async_validate_path(context, path)
report = context.build_evidence_report()
jsonable = report.to_json()
```

The report only contains material that actually participated in a decision:

- OCSP responses distinguish `good` / `revoked` / `unknown`, and record
  CertID matching, signature validity, responder authorisation, freshness,
  `thisUpdate`/`nextUpdate`, revocation time/reason and the selection reason.
- CRLs record issuer/IDP scope, the covered revocation reasons, signature and
  issuer-path validity, freshness, whether the serial is listed, and any
  revocation time/reason.
- Each material carries provenance: `provided`, `cache` or `fetched`.
  A successful download is only provenance; it never counts as valid status
  evidence until every check passes.
- Correctly signed material that targets another certificate or issuer is
  recorded as out of scope and never treated as valid evidence.
- Missing material and policy-permitted soft-fails are recorded separately.

Raw OCSP/CRL bytes are not embedded; the report stores their SHA-256 digests.
Keep the bytes alongside the report if you intend to replay.

## Offline replay

Replay never trusts the report's own claims. Networking is disabled, and the
caller must explicitly supply trust roots and an equivalent revocation
policy; mismatched policy or raw-byte digests raise an error before
validation runs.

```python
outcome = await replay_validation_offline(
    report,
    trust_roots=roots,                 # caller-supplied, mandatory
    revinfo_policy=policy,             # must match the recorded policy
    end_entity_cert=leaf,
    intermediate_certs=intermediates,
    raw_ocsps=saved_ocsp_bytes,
    raw_crls=saved_crl_bytes,
)
```

`outcome` explains whether the original run accepted, rejected for
revocation, rejected for insufficient information, or merely continued under
soft-fail, citing the per-material reasons rather than the last exception.
