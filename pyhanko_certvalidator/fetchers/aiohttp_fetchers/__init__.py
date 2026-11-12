from typing import Optional

import aiohttp

from ..api import FetcherBackend, Fetchers
from ..network_policy import NetworkAdmissionPolicy
from .cert_fetch_client import AIOHttpCertificateFetcher
from .crl_client import AIOHttpCRLFetcher
from .ocsp_client import AIOHttpOCSPFetcher
from .util import LazySession

__all__ = ['AIOHttpFetcherBackend']


class AIOHttpFetcherBackend(FetcherBackend):
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        per_request_timeout=10,
        network_policy: Optional[NetworkAdmissionPolicy] = None,
    ):
        if network_policy is not None and session is not None:
            raise ValueError(
                "A network admission policy cannot be combined with a "
                "caller-supplied aiohttp session: the policy needs its own "
                "policy-aware connector. Omit the session to let the "
                "backend create one."
            )
        self.session = session or LazySession(policy=network_policy)
        self.per_request_timeout = per_request_timeout
        self.network_policy = network_policy

    def get_fetchers(self) -> Fetchers:
        session = self.session
        to = self.per_request_timeout
        policy = self.network_policy
        return Fetchers(
            ocsp_fetcher=AIOHttpOCSPFetcher(
                session, per_request_timeout=to, network_policy=policy
            ),
            crl_fetcher=AIOHttpCRLFetcher(
                session, per_request_timeout=to, network_policy=policy
            ),
            cert_fetcher=AIOHttpCertificateFetcher(
                session, per_request_timeout=to, network_policy=policy
            ),
        )

    async def close(self):
        session = self.session
        # only close the session if it's a lazy session;
        # a session passed in by the caller is their own responsibility
        if isinstance(session, LazySession):
            await session.close()
