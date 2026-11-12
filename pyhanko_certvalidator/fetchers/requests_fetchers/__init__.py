"""
Fetcher implementation using the ``requests`` library for backwards
compatibility. This fetcher backend doesn't take advantage of asyncio, but
has the advantage of not requiring any resource management on the caller's
part.
"""

from typing import Optional

from ..api import FetcherBackend, Fetchers
from ..network_policy import NetworkAdmissionPolicy
from .cert_fetch_client import RequestsCertificateFetcher
from .crl_client import RequestsCRLFetcher
from .ocsp_client import RequestsOCSPFetcher

__all__ = ['RequestsFetcherBackend']


class RequestsFetcherBackend(FetcherBackend):
    def __init__(
        self,
        per_request_timeout=10,
        network_policy: Optional[NetworkAdmissionPolicy] = None,
    ):
        self.per_request_timeout = per_request_timeout
        self.network_policy = network_policy

    def get_fetchers(self) -> Fetchers:
        to = self.per_request_timeout
        policy = self.network_policy
        return Fetchers(
            ocsp_fetcher=RequestsOCSPFetcher(
                per_request_timeout=to, network_policy=policy
            ),
            crl_fetcher=RequestsCRLFetcher(
                per_request_timeout=to, network_policy=policy
            ),
            cert_fetcher=RequestsCertificateFetcher(
                per_request_timeout=to, network_policy=policy
            ),
        )

    async def close(self):
        # don't need to do anything
        return
