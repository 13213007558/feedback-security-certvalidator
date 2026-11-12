import asyncio
from typing import Any, Dict, Optional, Union

import aiohttp

from .._guarded_http import guarded_http_request
from ..api import DEFAULT_USER_AGENT
from ..common_utils import queue_fetch_task
from ..network_policy import NetworkAdmissionPolicy
from .policy_transport import AIOHttpGuardedTransport

__all__ = ['LazySession', 'AIOHttpMixin']


class LazySession:
    def __init__(self, policy: Optional[NetworkAdmissionPolicy] = None):
        self._session = None
        self._policy = policy

    async def get_session(self):
        session = self._session
        if session is None:
            if self._policy is not None:
                from .policy_transport import make_policy_aware_session

                self._session = session = make_policy_aware_session(
                    self._policy
                )
            else:
                self._session = session = aiohttp.ClientSession()
        return session

    async def close(self):
        session = self._session
        if session is not None:
            await session.close()


class AIOHttpMixin:
    def __init__(
        self,
        session: Union[aiohttp.ClientSession, LazySession],
        user_agent=None,
        per_request_timeout=10,
        network_policy: Optional[NetworkAdmissionPolicy] = None,
    ):
        self._session = session
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        if network_policy is not None:
            if not isinstance(session, LazySession):
                raise ValueError(
                    "A network admission policy requires a LazySession (or "
                    "no explicit session) so the connector can be bound to "
                    "the policy; raw aiohttp sessions are not supported."
                )
            self.network_policy = network_policy
            self.per_request_timeout = network_policy.per_request_timeout
            self._guarded_transport = AIOHttpGuardedTransport(
                network_policy, session_provider=self.get_session
            )
        else:
            self.network_policy = None
            self.per_request_timeout = per_request_timeout
            self._guarded_transport = None
        self.__results: Dict[Any, Any] = {}
        self.__result_events: Dict[Any, asyncio.Event] = {}
        super().__init__()

    async def get_session(self) -> aiohttp.ClientSession:
        session = self._session
        if isinstance(session, LazySession):
            return await session.get_session()
        else:
            return session

    def get_results(self):
        return {
            v for v in self.__results.values() if not isinstance(v, Exception)
        }

    def get_results_for_tag(self, tag):
        result = self.__results[tag]
        if isinstance(result, Exception):
            raise KeyError

    def _iter_results(self):
        for k, v in self.__results.items():
            if not isinstance(v, Exception):
                yield k, v

    async def _post_fetch_task(self, tag, async_fun):
        return await queue_fetch_task(
            self.__results, self.__result_events, tag, async_fun
        )

    async def _guarded_request(
        self, method, url, headers, data
    ):
        # NetworkPolicyError is intentionally not caught here or in the
        # fetcher clients: policy denials surface as security errors rather
        # than missing validation material.
        assert self.network_policy is not None
        return await guarded_http_request(
            self._guarded_transport,
            self.network_policy,
            url,
            method=method,
            headers=headers,
            data=data,
        )

    async def _close_guarded_transport(self):
        if self._guarded_transport is not None:
            await self._guarded_transport.close()
