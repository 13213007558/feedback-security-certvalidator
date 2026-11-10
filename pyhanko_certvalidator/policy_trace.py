"""
Optional policy processing trace for RFC 5280 path validation.

.. versionadded:: 0.26.9

This module implements an *optional* observer that records how the
valid policy tree and the policy-related state variables
(``explicit_policy``, ``inhibit_any_policy``, ``policy_mapping``)
evolve while a certification path is being validated.

The trace is recorded by the actual validation algorithm as it runs
(see :func:`pyhanko_certvalidator.validate.async_validate_path`), not by
a separate reimplementation, so the events in the trace always reflect
the real reason why a path was accepted or rejected.

Typical usage::

    from pyhanko_certvalidator.policy_trace import PolicyTraceRecorder

    trace = PolicyTraceRecorder()
    try:
        await async_validate_path(
            validation_context, path, parameters=params, policy_trace=trace
        )
    except PathValidationError:
        print(trace.render())
        raise

On failure, the raised
:class:`~pyhanko_certvalidator.errors.PathValidationError`
also carries the recorder in its ``policy_trace`` attribute, and
:attr:`PolicyTraceRecorder.last_valid_state` together with
:attr:`PolicyTraceRecorder.failure` pinpoint the certificate (by SHA-256
fingerprint) and the constraint level that caused the rejection.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from asn1crypto import x509

from .path import QualifiedPolicy
from .policy_tree import PolicyTreeNode, PolicyTreeRoot

__all__ = [
    'PolicyTreeNodeSnapshot',
    'PolicyTreeSnapshot',
    'RemappedNode',
    'PolicyTreeDelta',
    'InitialisationEvent',
    'CertPolicyEvent',
    'PolicyMappingEvent',
    'PolicyCounterEvent',
    'QualifiedPolicyOrigin',
    'FinalIntersectionEvent',
    'PolicyFailureEvent',
    'PolicyTraceEvent',
    'PolicyTraceState',
    'PolicyTraceRecorder',
]


def _fmt_policies(policies: Iterable[str]) -> str:
    return '{' + ', '.join(sorted(policies)) + '}'


@dataclass(frozen=True)
class PolicyTreeNodeSnapshot:
    """
    Immutable snapshot of a single node in the valid policy tree.
    """

    node_id: int
    """
    Identity of the live tree node this snapshot was taken from.
    Unique within a trace; allows nodes to be followed across events.
    """

    depth: int
    """
    Depth of the node in the valid policy tree (0 = child of the root,
    i.e. a policy valid in the trust anchor's domain).
    """

    valid_policy: str
    """
    The valid_policy of the node (a policy OID or ``any_policy``).
    """

    expected_policy_set: FrozenSet[str]
    """
    The expected_policy_set of the node at the time of the snapshot.
    """

    parent_node_id: Optional[int]
    """
    The :attr:`node_id` of the parent node, or ``None`` for the
    (virtual) root's children.
    """


@dataclass(frozen=True)
class PolicyTreeSnapshot:
    """
    Immutable snapshot of an entire valid policy tree.
    """

    nodes: Tuple[PolicyTreeNodeSnapshot, ...]

    def at_depth(self, depth: int) -> List[PolicyTreeNodeSnapshot]:
        return [node for node in self.nodes if node.depth == depth]

    def render(self, indent: str = '') -> str:
        """
        Render the tree as indented lines, one node per line.
        """
        if not self.nodes:
            return f'{indent}(empty tree)'
        by_parent: Dict[Optional[int], List[PolicyTreeNodeSnapshot]] = {}
        for node in self.nodes:
            by_parent.setdefault(node.parent_node_id, []).append(node)
        lines: List[str] = []

        def _emit(parent_id: Optional[int], level: int):
            for node in sorted(
                by_parent.get(parent_id, []),
                key=lambda n: (n.depth, n.valid_policy),
            ):
                lines.append(
                    '%s%s%s (expected: %s)'
                    % (
                        indent,
                        '  ' * level,
                        node.valid_policy,
                        _fmt_policies(node.expected_policy_set),
                    )
                )
                _emit(node.node_id, level + 1)

        _emit(None, 0)
        return '\n'.join(lines)


@dataclass(frozen=True)
class RemappedNode:
    """
    Record of a node whose expected_policy_set was rewritten in place
    by a policy mapping (RFC 5280, 6.1.4 (b)(1) style update).
    """

    node_id: int
    depth: int
    valid_policy: str
    old_expected_policy_set: FrozenSet[str]
    new_expected_policy_set: FrozenSet[str]


@dataclass(frozen=True)
class PolicyTreeDelta:
    """
    Structural difference between two consecutive tree snapshots.
    """

    added: Tuple[PolicyTreeNodeSnapshot, ...]
    """
    Nodes that were added to the tree.
    """

    removed: Tuple[PolicyTreeNodeSnapshot, ...]
    """
    Nodes that were removed from the tree (pruned or deleted).
    """

    remapped: Tuple[RemappedNode, ...]
    """
    Nodes whose expected_policy_set changed in place.
    """

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.remapped)


class PolicyTraceEvent:
    """
    Base class for all policy trace events. Events are stored in
    validation order in :attr:`PolicyTraceRecorder.events`; the index
    of an event in that list is its sequence number.
    """


@dataclass(frozen=True)
class InitialisationEvent(PolicyTraceEvent):
    """
    Initial state of the policy-related variables (RFC 5280, 6.1.2).
    """

    path_length: int
    explicit_policy: int
    inhibit_any_policy: int
    policy_mapping: int
    acceptable_policies: FrozenSet[str]
    """
    Caller's initial policy set (after merging with trust anchor
    qualifiers, if applicable).
    """

    initial_tree: PolicyTreeSnapshot


@dataclass(frozen=True)
class CertPolicyEvent(PolicyTraceEvent):
    """
    Result of processing the certificatePolicies extension of one
    certificate in the path (RFC 5280, 6.1.3 (d)-(f)).
    """

    index: int
    """
    1-based position of the certificate in the path (trust anchor = 0).
    """

    cert_sha256: str
    """
    Hex-encoded SHA-256 fingerprint of the certificate.
    """

    subject: str
    """
    Human-readable subject name of the certificate.
    """

    self_issued: bool
    is_leaf: bool
    asserted_policies: Optional[FrozenSet[str]]
    """
    Policy OIDs asserted by the certificate, or ``None`` if the
    certificate has no certificatePolicies extension (in which case the
    valid policy tree is set to NULL, cf. RFC 5280, 6.1.3 (e)).
    """

    any_policy_uninhibited: bool
    """
    Whether anyPolicy was processed for this certificate
    (RFC 5280, 6.1.3 (d)(2)).
    """

    delta: PolicyTreeDelta
    """
    How the valid policy tree changed while processing this certificate.
    """

    tree_after: Optional[PolicyTreeSnapshot]
    """
    The valid policy tree after processing, or ``None`` if it is NULL.
    """

    tree_nulled: bool
    """
    True if the tree was set to NULL because the certificate does not
    contain a certificatePolicies extension.
    """


@dataclass(frozen=True)
class PolicyMappingEvent(PolicyTraceEvent):
    """
    Result of processing the policyMappings extension of one
    certificate (RFC 5280, 6.1.4 (a)-(b), prepared in 6.1.3).
    """

    index: int
    cert_sha256: str
    subject: str
    mappings: Tuple[Tuple[str, str], ...]
    """
    (issuer_domain_policy, subject_domain_policy) pairs declared by the
    certificate.
    """

    mapping_uninhibited: bool
    """
    Whether policy mapping was permitted at this point in the path
    (i.e. the policy_mapping state variable was nonzero).
    """

    policy_mapping_counter: int
    """
    Value of the policy_mapping state variable when the mapping
    was processed.
    """

    delta: PolicyTreeDelta
    tree_after: Optional[PolicyTreeSnapshot]


@dataclass(frozen=True)
class PolicyCounterEvent(PolicyTraceEvent):
    """
    Change (or deliberate non-change) of one of the policy-related
    state variables.
    """

    index: Optional[int]
    """
    1-based position of the certificate that triggered the change,
    or ``None`` during wrap-up processing (RFC 5280, 6.1.4).
    """

    cert_sha256: Optional[str]
    subject: Optional[str]
    counter: str
    """
    One of ``explicit_policy``, ``inhibit_any_policy``,
    ``policy_mapping``.
    """

    old_value: int
    new_value: int
    reason: str
    """
    Human-readable justification, with a reference to the relevant
    RFC 5280 step.
    """


@dataclass(frozen=True)
class QualifiedPolicyOrigin:
    """
    Provenance record linking a :class:`~.path.QualifiedPolicy` back to
    the trace.
    """

    user_domain_policy_id: str
    issuer_domain_policy_id: str
    node_path: Tuple[str, ...]
    """
    Chain of valid_policy values from the root's child down to the
    leaf node that gave rise to this qualified policy.
    """

    introduced_in_event: Optional[int]
    """
    Sequence number (index into :attr:`PolicyTraceRecorder.events`) of
    the event whose tree delta introduced the leaf node, if known.
    """


@dataclass(frozen=True)
class FinalIntersectionEvent(PolicyTraceEvent):
    """
    Final intersection of the valid policy tree with the caller's
    initial policy set (RFC 5280, 6.1.4 (g)).
    """

    acceptable_policies: FrozenSet[str]
    explicit_policy: int
    """
    Value of the explicit_policy state variable at the end of
    wrap-up processing.
    """

    tree_before: Optional[PolicyTreeSnapshot]
    delta: PolicyTreeDelta
    """
    Effect of pruning unacceptable policies and expanding anyPolicy
    wildcards.
    """

    qualified_policies: Tuple[QualifiedPolicyOrigin, ...]
    intersection_empty: bool


@dataclass(frozen=True)
class PolicyFailureEvent(PolicyTraceEvent):
    """
    Policy-related validation failure. Always the last event in
    a trace when validation fails.
    """

    index: Optional[int]
    cert_sha256: Optional[str]
    """
    Fingerprint of the certificate whose processing caused the failure.
    """

    subject: Optional[str]
    message: str


@dataclass(frozen=True)
class PolicyTraceState:
    """
    Last recorded valid state of the policy processing variables.
    """

    explicit_policy: int
    inhibit_any_policy: int
    policy_mapping: int
    valid_policy_tree: Optional[PolicyTreeSnapshot]

    def render(self, indent: str = '') -> str:
        tree = self.valid_policy_tree
        lines = [
            '%sexplicit_policy=%d, inhibit_any_policy=%d, policy_mapping=%d'
            % (
                indent,
                self.explicit_policy,
                self.inhibit_any_policy,
                self.policy_mapping,
            )
        ]
        if tree is None:
            lines.append(f'{indent}valid policy tree: NULL')
        else:
            lines.append(f'{indent}valid policy tree:')
            lines.append(tree.render(indent + '  '))
        return '\n'.join(lines)


def _cert_info(
    cert: Optional[x509.Certificate],
) -> Tuple[Optional[str], Optional[str]]:
    if cert is None:
        return None, None
    return cert.sha256.hex(), cert.subject.human_friendly


class PolicyTraceRecorder:
    """
    Observer that records a policy processing trace during path
    validation. Pass an instance to
    :func:`pyhanko_certvalidator.validate.async_validate_path` (or its
    synchronous sibling) via the ``policy_trace`` parameter.

    The recorder is driven by the validation algorithm itself; no
    separate policy evaluation is performed.
    """

    def __init__(self):
        self.events: List[PolicyTraceEvent] = []
        self._node_first_seen: Dict[int, int] = {}
        # keep strong references to live tree nodes so that id() values
        # cannot be recycled by the allocator while the trace is alive
        self._node_refs: Dict[int, PolicyTreeNode] = {}
        self._counters: Dict[str, int] = {}
        self._last_tree: Optional[PolicyTreeSnapshot] = None

    # ---- snapshot & diff machinery (internal) ----

    def snapshot_tree(
        self, tree: Optional[PolicyTreeRoot]
    ) -> Optional[PolicyTreeSnapshot]:
        """
        Take an immutable snapshot of a valid policy tree
        (``None`` represents the NULL tree).
        """
        if tree is None:
            return None
        nodes: List[PolicyTreeNodeSnapshot] = []

        def _walk(node, depth: int, parent_id: Optional[int]):
            for child in node.children:
                child_id = id(child)
                self._node_refs[child_id] = child
                nodes.append(
                    PolicyTreeNodeSnapshot(
                        node_id=child_id,
                        depth=depth,
                        valid_policy=child.valid_policy,
                        expected_policy_set=frozenset(
                            child.expected_policy_set
                        ),
                        parent_node_id=parent_id,
                    )
                )
                _walk(child, depth + 1, child_id)

        _walk(tree, 0, None)
        return PolicyTreeSnapshot(nodes=tuple(nodes))

    @staticmethod
    def _diff(
        before: Optional[PolicyTreeSnapshot],
        after: Optional[PolicyTreeSnapshot],
    ) -> PolicyTreeDelta:
        before_nodes = {n.node_id: n for n in before.nodes} if before else {}
        after_nodes = {n.node_id: n for n in after.nodes} if after else {}
        added = tuple(
            n
            for nid, n in sorted(after_nodes.items())
            if nid not in before_nodes
        )
        removed = tuple(
            n
            for nid, n in sorted(before_nodes.items())
            if nid not in after_nodes
        )
        remapped = tuple(
            RemappedNode(
                node_id=nid,
                depth=after_nodes[nid].depth,
                valid_policy=after_nodes[nid].valid_policy,
                old_expected_policy_set=before_nodes[nid].expected_policy_set,
                new_expected_policy_set=after_nodes[nid].expected_policy_set,
            )
            for nid in sorted(before_nodes.keys() & after_nodes.keys())
            if before_nodes[nid].expected_policy_set
            != after_nodes[nid].expected_policy_set
        )
        return PolicyTreeDelta(added=added, removed=removed, remapped=remapped)

    def _register_nodes(self, snapshot: Optional[PolicyTreeSnapshot]):
        if snapshot is None:
            return
        # must be called after the corresponding event was appended
        seq = len(self.events) - 1
        for node in snapshot.nodes:
            self._node_first_seen.setdefault(node.node_id, seq)

    # ---- recording API (called by the validation algorithm) ----

    def record_initialisation(
        self,
        *,
        path_length: int,
        explicit_policy: int,
        inhibit_any_policy: int,
        policy_mapping: int,
        acceptable_policies: FrozenSet[str],
        initial_tree: Optional[PolicyTreeRoot],
    ):
        snapshot = self.snapshot_tree(initial_tree)
        self._counters = {
            'explicit_policy': explicit_policy,
            'inhibit_any_policy': inhibit_any_policy,
            'policy_mapping': policy_mapping,
        }
        self._last_tree = snapshot
        self.events.append(
            InitialisationEvent(
                path_length=path_length,
                explicit_policy=explicit_policy,
                inhibit_any_policy=inhibit_any_policy,
                policy_mapping=policy_mapping,
                acceptable_policies=acceptable_policies,
                initial_tree=snapshot,
            )
        )
        self._register_nodes(snapshot)

    def record_cert_policies(
        self,
        *,
        index: int,
        cert: x509.Certificate,
        self_issued: bool,
        is_leaf: bool,
        asserted_policies: Optional[FrozenSet[str]],
        any_policy_uninhibited: bool,
        tree_before: Optional[PolicyTreeSnapshot],
        tree_after: Optional[PolicyTreeRoot],
    ):
        after = self.snapshot_tree(tree_after)
        sha256, subject = _cert_info(cert)
        self._last_tree = after
        self.events.append(
            CertPolicyEvent(
                index=index,
                cert_sha256=sha256,
                subject=subject,
                self_issued=self_issued,
                is_leaf=is_leaf,
                asserted_policies=asserted_policies,
                any_policy_uninhibited=any_policy_uninhibited,
                delta=self._diff(tree_before, after),
                tree_after=after,
                tree_nulled=asserted_policies is None,
            )
        )
        self._register_nodes(after)

    def record_policy_mapping(
        self,
        *,
        index: int,
        cert: x509.Certificate,
        mappings: Tuple[Tuple[str, str], ...],
        mapping_uninhibited: bool,
        policy_mapping_counter: int,
        tree_before: Optional[PolicyTreeSnapshot],
        tree_after: Optional[PolicyTreeRoot],
    ):
        after = self.snapshot_tree(tree_after)
        sha256, subject = _cert_info(cert)
        self._last_tree = after
        self.events.append(
            PolicyMappingEvent(
                index=index,
                cert_sha256=sha256,
                subject=subject,
                mappings=mappings,
                mapping_uninhibited=mapping_uninhibited,
                policy_mapping_counter=policy_mapping_counter,
                delta=self._diff(tree_before, after),
                tree_after=after,
            )
        )
        self._register_nodes(after)

    def record_counter(
        self,
        *,
        index: Optional[int],
        cert: Optional[x509.Certificate],
        counter: str,
        old: int,
        new: int,
        reason: str,
    ):
        sha256, subject = _cert_info(cert)
        self._counters[counter] = new
        self.events.append(
            PolicyCounterEvent(
                index=index,
                cert_sha256=sha256,
                subject=subject,
                counter=counter,
                old_value=old,
                new_value=new,
                reason=reason,
            )
        )

    def record_intersection(
        self,
        *,
        acceptable_policies: FrozenSet[str],
        explicit_policy: int,
        tree_before: Optional[PolicyTreeSnapshot],
        tree_after: Optional[PolicyTreeRoot],
        qualified: Iterable[Tuple[QualifiedPolicy, PolicyTreeNode]],
    ):
        after = self.snapshot_tree(tree_after)
        self._last_tree = after
        origins = []
        for qual_policy, node in qualified:
            chain = []
            cursor = node
            while cursor is not None and cursor.parent is not None:
                chain.append(cursor.valid_policy)
                cursor = cursor.parent
            origins.append(
                QualifiedPolicyOrigin(
                    user_domain_policy_id=qual_policy.user_domain_policy_id,
                    issuer_domain_policy_id=qual_policy.issuer_domain_policy_id,
                    node_path=tuple(reversed(chain)),
                    introduced_in_event=self._node_first_seen.get(id(node)),
                )
            )
        self.events.append(
            FinalIntersectionEvent(
                acceptable_policies=acceptable_policies,
                explicit_policy=explicit_policy,
                tree_before=tree_before,
                delta=self._diff(tree_before, after),
                qualified_policies=tuple(origins),
                intersection_empty=not origins,
            )
        )
        self._register_nodes(after)

    def record_failure(
        self,
        *,
        index: Optional[int],
        cert: Optional[x509.Certificate],
        message: str,
    ):
        sha256, subject = _cert_info(cert)
        self.events.append(
            PolicyFailureEvent(
                index=index,
                cert_sha256=sha256,
                subject=subject,
                message=message,
            )
        )

    # ---- accessors ----

    @property
    def failure(self) -> Optional[PolicyFailureEvent]:
        """
        The failure event that ended the trace, if validation failed
        on a policy-related condition.
        """
        for event in reversed(self.events):
            if isinstance(event, PolicyFailureEvent):
                return event
        return None

    @property
    def last_valid_state(self) -> Optional[PolicyTraceState]:
        """
        The most recent recorded state of the policy processing
        variables. On failure, this is the state immediately before
        the failing condition was detected.
        """
        if not self._counters:
            return None
        return PolicyTraceState(
            explicit_policy=self._counters['explicit_policy'],
            inhibit_any_policy=self._counters['inhibit_any_policy'],
            policy_mapping=self._counters['policy_mapping'],
            valid_policy_tree=self._last_tree,
        )

    def origin_of(self, event_seq: int) -> Optional[PolicyTraceEvent]:
        """
        Return the event with the given sequence number, or ``None``
        if out of range. Useful together with
        :attr:`QualifiedPolicyOrigin.introduced_in_event`.
        """
        if 0 <= event_seq < len(self.events):
            return self.events[event_seq]
        return None

    # ---- reporting ----

    def render(self) -> str:
        """
        Render the trace as a compact human-readable report, suitable
        for deciding which trust configuration to adjust or which CA
        in the chain to ask for a re-issued certificate.
        """
        lines = ['=== Policy processing trace ===']
        for seq, event in enumerate(self.events):
            self._render_event(seq, event, lines)
        state = self.last_valid_state
        if state is not None:
            lines.append('Final recorded state:')
            lines.append(state.render('  '))
        return '\n'.join(lines)

    def _render_event(
        self, seq: int, event: PolicyTraceEvent, lines: List[str]
    ):
        if isinstance(event, InitialisationEvent):
            lines.append(
                f'[{seq}] initial state (path length {event.path_length}): '
                f'explicit_policy={event.explicit_policy}, '
                f'inhibit_any_policy={event.inhibit_any_policy}, '
                f'policy_mapping={event.policy_mapping}'
            )
            lines.append(
                '    caller-acceptable policies: '
                + _fmt_policies(event.acceptable_policies)
            )
        elif isinstance(event, CertPolicyEvent):
            kind = ['self-issued' if event.self_issued else 'not self-issued']
            if event.is_leaf:
                kind.append('leaf')
            lines.append(
                f'[{seq}] cert {event.index} ({event.subject}) '
                f'sha256={event.cert_sha256} [{", ".join(kind)}]'
            )
            if event.asserted_policies is None:
                lines.append(
                    '    no certificatePolicies extension -> '
                    'valid policy tree set to NULL (RFC 5280 6.1.3 e)'
                )
            else:
                lines.append(
                    '    asserted policies: '
                    + _fmt_policies(event.asserted_policies)
                )
                lines.append(
                    '    anyPolicy processing: '
                    + (
                        'uninhibited'
                        if event.any_policy_uninhibited
                        else 'INHIBITED (RFC 5280 6.1.3 d(2) skipped)'
                    )
                )
            self._render_delta(event.delta, lines)
            self._render_tree_state(event.tree_after, lines)
        elif isinstance(event, PolicyMappingEvent):
            lines.append(
                f'[{seq}] policy mappings in cert {event.index} '
                f'({event.subject}) sha256={event.cert_sha256}'
            )
            for issuer_pol, subject_pol in event.mappings:
                lines.append(f'    maps {issuer_pol} -> {subject_pol}')
            if event.mapping_uninhibited:
                lines.append('    mapping applied (policy_mapping > 0)')
            else:
                lines.append(
                    '    mapping INHIBITED (policy_mapping = 0): '
                    'matching nodes deleted (RFC 5280 6.1.4 b(2) analogue)'
                )
            self._render_delta(event.delta, lines)
            self._render_tree_state(event.tree_after, lines)
        elif isinstance(event, PolicyCounterEvent):
            where = (
                f'cert {event.index}' if event.index is not None else 'wrap-up'
            )
            if event.old_value == event.new_value:
                change = f'= {event.new_value} (unchanged)'
            else:
                change = f'{event.old_value} -> {event.new_value}'
            lines.append(
                f'[{seq}] {where}: {event.counter} {change}; '
                f'{event.reason}'
            )
        elif isinstance(event, FinalIntersectionEvent):
            lines.append(
                f'[{seq}] final intersection with caller policy set '
                + _fmt_policies(event.acceptable_policies)
                + f' (explicit_policy={event.explicit_policy})'
            )
            self._render_delta(event.delta, lines)
            if event.intersection_empty:
                lines.append('    intersection is EMPTY')
            else:
                for origin in event.qualified_policies:
                    src = origin.introduced_in_event
                    src_txt = f'event #{src}' if src is not None else 'unknown'
                    lines.append(
                        f'    accepted: {origin.user_domain_policy_id} '
                        f'(asserted as {origin.issuer_domain_policy_id}; '
                        f'path {" / ".join(origin.node_path)}; '
                        f'introduced at {src_txt})'
                    )
        elif isinstance(event, PolicyFailureEvent):
            where = (
                f'cert {event.index}' if event.index is not None else 'wrap-up'
            )
            lines.append(
                f'[{seq}] FAILURE at {where} ({event.subject}) '
                f'sha256={event.cert_sha256}'
            )
            lines.append(f'    {event.message}')

    @staticmethod
    def _render_delta(delta: PolicyTreeDelta, lines: List[str]):
        for node in delta.added:
            lines.append(
                f'    + {node.valid_policy} @depth {node.depth} '
                f'(expected: {_fmt_policies(node.expected_policy_set)})'
            )
        for node in delta.removed:
            lines.append(
                f'    - {node.valid_policy} @depth {node.depth} (pruned)'
            )
        for node in delta.remapped:
            lines.append(
                f'    ~ {node.valid_policy} @depth {node.depth}: expected '
                f'{_fmt_policies(node.old_expected_policy_set)} -> '
                f'{_fmt_policies(node.new_expected_policy_set)}'
            )

    @staticmethod
    def _render_tree_state(
        tree: Optional[PolicyTreeSnapshot], lines: List[str]
    ):
        if tree is None:
            lines.append('    valid policy tree now: NULL')
        else:
            lines.append('    valid policy tree now:')
            lines.append(tree.render('      '))
