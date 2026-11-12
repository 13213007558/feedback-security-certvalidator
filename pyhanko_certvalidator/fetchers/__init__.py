from .api import *
from .network_policy import (
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_MAX_RESPONSE_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    DefaultHostResolver,
    HostResolver,
    NetworkAdmissionPolicy,
    NetworkPolicyError,
    PolicyDenialReason,
)

__all__ = [
    'Fetchers',
    'FetcherBackend',
    'OCSPFetcher',
    'CRLFetcher',
    'CertificateFetcher',
    'default_fetcher_backend',
    'NetworkAdmissionPolicy',
    'NetworkPolicyError',
    'PolicyDenialReason',
    'HostResolver',
    'DefaultHostResolver',
    'DEFAULT_MAX_REDIRECTS',
    'DEFAULT_MAX_RESPONSE_SIZE',
    'DEFAULT_REQUEST_TIMEOUT',
]


def default_fetcher_backend() -> FetcherBackend:
    """
    Instantiate a default fetcher backend that doesn't require any resource
    management, but is less efficient than a fully asynchronous fetcher
    would be.
    """

    from .requests_fetchers import RequestsFetcherBackend

    return RequestsFetcherBackend()
