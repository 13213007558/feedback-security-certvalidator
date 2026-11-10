import pytest
from freezegun import freeze_time

from pyhanko_certvalidator.context import ValidationContext
from pyhanko_certvalidator.errors import PathValidationError
from pyhanko_certvalidator.policy_decl import (
    DisallowWeakAlgorithmsPolicy,
    PKIXValidationParams,
)
from pyhanko_certvalidator.policy_trace import (
    CertPolicyEvent,
    FinalIntersectionEvent,
    InitialisationEvent,
    PolicyCounterEvent,
    PolicyFailureEvent,
    PolicyMappingEvent,
    PolicyTraceRecorder,
)
from pyhanko_certvalidator.validate import async_validate_path
from tests.common import load_nist_cert

NIST_POLICY_1 = '2.16.840.1.101.3.2.1.48.1'
NIST_POLICY_2 = '2.16.840.1.101.3.2.1.48.2'


def _nist_context(*other_cert_names):
    root = load_nist_cert('TrustAnchorRootCertificate.crt')
    return ValidationContext(
        trust_roots=[root],
        other_certs=[load_nist_cert(name) for name in other_cert_names],
        revocation_mode='soft-fail',
        # adjust default algo policy to pass NIST tests
        algorithm_usage_policy=DisallowWeakAlgorithmsPolicy(
            weak_hash_algos={'md2', 'md5'}, dsa_key_size_threshold=1024
        ),
    )


async def _build_path(context, ee_name):
    ee = load_nist_cert(ee_name)
    (path,) = await context.path_builder.async_build_paths(ee)
    return ee, path


def _events(trace, cls):
    return [e for e in trace.events if isinstance(e, cls)]


@freeze_time('2022-05-01')
@pytest.mark.asyncio
async def test_trace_policy_mapping_success():
    # trust anchor -> Mapping1to2CACert (maps P1 -> P2) -> EE (asserts P2),
    # caller requires P1: mapping connects the two policy domains
    context = _nist_context('Mapping1to2CACert.crt')
    ee, path = await _build_path(context, 'ValidPolicyMappingTest1EE.crt')
    trace = PolicyTraceRecorder()
    params = PKIXValidationParams(
        user_initial_policy_set=frozenset([NIST_POLICY_1])
    )
    await async_validate_path(
        context, path, parameters=params, policy_trace=trace
    )

    init = _events(trace, InitialisationEvent)[0]
    assert init.acceptable_policies == frozenset([NIST_POLICY_1])
    assert init.explicit_policy == 3

    # the mapping was applied while policy_mapping > 0
    (mapping_event,) = _events(trace, PolicyMappingEvent)
    assert mapping_event.mappings == ((NIST_POLICY_1, NIST_POLICY_2),)
    assert mapping_event.mapping_uninhibited
    (remapped,) = mapping_event.delta.remapped
    assert remapped.valid_policy == NIST_POLICY_1
    assert remapped.old_expected_policy_set == frozenset([NIST_POLICY_1])
    assert remapped.new_expected_policy_set == frozenset([NIST_POLICY_2])

    # final intersection is visible and non-empty
    (fin,) = _events(trace, FinalIntersectionEvent)
    assert not fin.intersection_empty
    (origin,) = fin.qualified_policies
    assert origin.user_domain_policy_id == NIST_POLICY_1
    assert origin.issuer_domain_policy_id == NIST_POLICY_2
    assert origin.node_path == ('any_policy', NIST_POLICY_1, NIST_POLICY_2)

    # the qualified policy on the path can be traced back to the event
    # that introduced the corresponding node (the EE cert's processing)
    assert origin.introduced_in_event is not None
    source = trace.origin_of(origin.introduced_in_event)
    assert isinstance(source, CertPolicyEvent)
    assert source.index == 2
    assert source.cert_sha256 == ee.sha256.hex()
    assert any(n.valid_policy == NIST_POLICY_2 for n in source.delta.added)

    qps = path.qualified_policies()
    assert any(qp.user_domain_policy_id == NIST_POLICY_1 for qp in qps)

    assert trace.failure is None
    report = trace.render()
    assert f'maps {NIST_POLICY_1} -> {NIST_POLICY_2}' in report
    assert f'accepted: {NIST_POLICY_1}' in report


@freeze_time('2022-05-01')
@pytest.mark.asyncio
async def test_trace_policy_mapping_inhibited():
    # same chain, but the caller forbids policy mapping: the P1 node is
    # deleted, the tree is pruned to NULL and the EE cert is rejected
    context = _nist_context('Mapping1to2CACert.crt')
    ee, path = await _build_path(context, 'ValidPolicyMappingTest1EE.crt')
    trace = PolicyTraceRecorder()
    params = PKIXValidationParams(initial_policy_mapping_inhibit=True)
    with pytest.raises(PathValidationError, match='no valid set of policies'):
        await async_validate_path(
            context, path, parameters=params, policy_trace=trace
        )

    # the trace shows the mapping being suppressed, not silently skipped
    (mapping_event,) = _events(trace, PolicyMappingEvent)
    assert mapping_event.mappings == ((NIST_POLICY_1, NIST_POLICY_2),)
    assert not mapping_event.mapping_uninhibited
    assert mapping_event.policy_mapping_counter == 0
    removed_policies = {n.valid_policy for n in mapping_event.delta.removed}
    assert NIST_POLICY_1 in removed_policies
    assert mapping_event.tree_after is None

    # the failure is pinned to the EE certificate
    failure = trace.failure
    assert isinstance(failure, PolicyFailureEvent)
    assert failure.cert_sha256 == ee.sha256.hex()
    assert failure.index == 2

    # the last valid state shows the NULL tree and the zeroed counters
    last = trace.last_valid_state
    assert last.valid_policy_tree is None
    assert last.policy_mapping == 0
    assert last.explicit_policy == 0

    report = trace.render()
    assert 'mapping INHIBITED' in report
    assert 'FAILURE' in report
    assert ee.sha256.hex() in report


@freeze_time('2022-05-01')
@pytest.mark.asyncio
async def test_trace_inhibit_any_policy():
    # inhibitAnyPolicy0CACert sets inhibitAnyPolicy=0, so the anyPolicy
    # asserted by the EE cert must not be processed
    context = _nist_context('inhibitAnyPolicy0CACert.crt')
    ee, path = await _build_path(context, 'InvalidinhibitAnyPolicyTest1EE.crt')
    trace = PolicyTraceRecorder()
    with pytest.raises(PathValidationError, match='no valid set of policies'):
        await async_validate_path(context, path, policy_trace=trace)

    # the inhibitAnyPolicy extension drove the counter to zero
    iap_events = [
        e
        for e in _events(trace, PolicyCounterEvent)
        if e.counter == 'inhibit_any_policy'
    ]
    assert any(
        e.new_value == 0 and 'inhibitAnyPolicy' in e.reason
        for e in iap_events
    )

    # the EE cert's anyPolicy was left unprocessed and the tree pruned
    (ee_event,) = [
        e
        for e in _events(trace, CertPolicyEvent)
        if e.cert_sha256 == ee.sha256.hex()
    ]
    assert ee_event.asserted_policies == frozenset(['any_policy'])
    assert not ee_event.any_policy_uninhibited
    assert ee_event.tree_after is None
    assert any(
        n.valid_policy == NIST_POLICY_1 for n in ee_event.delta.removed
    )

    failure = trace.failure
    assert failure.cert_sha256 == ee.sha256.hex()
    assert 'INHIBITED' in trace.render()


@freeze_time('2022-05-01')
@pytest.mark.asyncio
async def test_trace_require_explicit_policy():
    # requireExplicitPolicy4CACert imposes requireExplicitPolicy=4;
    # the EE cert has no certificatePolicies, so the tree goes NULL and
    # explicit_policy reaches zero during wrap-up
    context = _nist_context(
        'requireExplicitPolicy4CACert.crt',
        'requireExplicitPolicy4subCACert.crt',
        'requireExplicitPolicy4subsubCACert.crt',
        'requireExplicitPolicy4subsubsubCACert.crt',
    )
    ee, path = await _build_path(
        context, 'InvalidrequireExplicitPolicyTest3EE.crt'
    )
    trace = PolicyTraceRecorder()
    with pytest.raises(PathValidationError, match='no valid set of policies'):
        await async_validate_path(context, path, policy_trace=trace)

    counter_events = _events(trace, PolicyCounterEvent)
    # the requireExplicitPolicy=4 constraint on the first CA cert
    assert any(
        e.counter == 'explicit_policy'
        and e.new_value == 4
        and 'requireExplicitPolicy=4' in e.reason
        for e in counter_events
    )
    # non-self-issued intermediates decrement the counter one by one
    decrements = [
        e
        for e in counter_events
        if e.counter == 'explicit_policy' and 'non-self-issued' in e.reason
    ]
    assert [e.new_value for e in decrements] == [5, 3, 2, 1]
    # the leaf cert does not trigger 6.1.3 h; wrap-up 6.1.4 a does
    wrap_up = [
        e
        for e in counter_events
        if e.counter == 'explicit_policy' and '6.1.4 a' in e.reason
    ]
    assert len(wrap_up) == 1
    assert wrap_up[0].old_value == 1
    assert wrap_up[0].new_value == 0

    # the EE cert has no certificatePolicies -> tree set to NULL
    (ee_event,) = [
        e
        for e in _events(trace, CertPolicyEvent)
        if e.cert_sha256 == ee.sha256.hex()
    ]
    assert ee_event.tree_nulled
    assert ee_event.asserted_policies is None

    # the final intersection with the caller's requirements is empty,
    # and the failure is recorded against the EE certificate
    (fin,) = _events(trace, FinalIntersectionEvent)
    assert fin.intersection_empty
    assert fin.explicit_policy == 0
    failure = trace.failure
    assert failure.cert_sha256 == ee.sha256.hex()

    report = trace.render()
    assert 'intersection is EMPTY' in report
    assert 'requireExplicitPolicy=4' in report


@freeze_time('2022-05-01')
@pytest.mark.asyncio
async def test_trace_self_issued_counters_not_decremented():
    # trust anchor -> RE2CA (requireExplicitPolicy=2)
    #   -> RE2SelfIssuedCA (self-issued) -> EE (no policies)
    # the self-issued certificate must not advance the counters
    context = _nist_context(
        'requireExplicitPolicy2CACert.crt',
        'requireExplicitPolicy2SelfIssuedCACert.crt',
    )
    ee, path = await _build_path(
        context, 'ValidSelfIssuedrequireExplicitPolicyTest6EE.crt'
    )
    trace = PolicyTraceRecorder()
    await async_validate_path(context, path, policy_trace=trace)

    self_issued_cert = load_nist_cert(
        'requireExplicitPolicy2SelfIssuedCACert.crt'
    )
    skip_events = [
        e
        for e in _events(trace, PolicyCounterEvent)
        if e.cert_sha256 == self_issued_cert.sha256.hex()
    ]
    # all three counters are explicitly recorded as NOT decremented
    assert {e.counter for e in skip_events} == {
        'explicit_policy',
        'policy_mapping',
        'inhibit_any_policy',
    }
    assert all(e.old_value == e.new_value for e in skip_events)
    assert all('self-issued' in e.reason for e in skip_events)

    # the self-issued cert processing is flagged in its cert event
    (si_event,) = [
        e
        for e in _events(trace, CertPolicyEvent)
        if e.cert_sha256 == self_issued_cert.sha256.hex()
    ]
    assert si_event.self_issued
    assert not si_event.is_leaf

    # validation succeeds; the trace ends without a failure event
    assert trace.failure is None
    last = trace.last_valid_state
    # explicit_policy: 4 -> 3 (cert 1) -> 2 (REP=2) -> 1 (wrap-up);
    # the self-issued cert contributed no decrement
    assert last.explicit_policy == 1
