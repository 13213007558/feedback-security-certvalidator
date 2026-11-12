from typing import Awaitable, Optional

import requests

from pyhanko_certvalidator._asyncio_compat import to_thread

from .._guarded_http import guarded_http_request
from ..api import DEFAULT_USER_AGENT
from ..common_utils import queue_fetch_task
from ..network_policy import NetworkAdmissionPolicy
from .policy_transport import RequestsGuardedTransport

__all__ = ['RequestsFetcherMixin']


class RequestsFetcherMixin:
    def __init__(
        self,
        user_agent=None,
        per_request_timeout=10,
        network_policy: Optional[NetworkAdmissionPolicy] = None,
    ):
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        if network_policy is not None:
            self.network_policy = network_policy
            self.per_request_timeout = network_policy.per_request_timeout
            self._guarded_transport = RequestsGuardedTransport(network_policy)
        else:
            self.network_policy = None
            self.per_request_timeout = per_request_timeout
            self._guarded_transport = None
        self.__results = {}
        self.__result_events = {}

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

    async def _perform_fetch(self, tag, fetch_fun):
        return await queue_fetch_task(
            self.__results, self.__result_events, tag, fetch_fun
        )

    async def _close_guarded_transport(self):
        if self._guarded_transport is not None:
            await self._guarded_transport.close()

    def _get(
        self, url, *, acceptable_content_types
    ) -> Awaitable['_GuardedResponseLike']:
        headers = {
            'Accept': ','.join(acceptable_content_types),
            'User-Agent': self.user_agent,
        }
        if self.network_policy is not None:
            return self._guarded_request('GET', url, headers, None)

        def task():
            response = requests.get(
                url=url,
                timeout=self.per_request_timeout,
                headers=headers,
            )
            if response.status_code != 200:
                raise requests.RequestException(
                    f"status code {response.status_code}"
                )
            return response

        return to_thread(task)

    def _post(
        self, url, data, *, content_type, acceptable_content_types
    ) -> Awaitable['_GuardedResponseLike']:
        headers = {
            'Accept': ','.join(acceptable_content_types),
            'User-Agent': self.user_agent,
            'Content-Type': content_type,
        }
        if self.network_policy is not None:
            return self._guarded_request('POST', url, headers, data)

        def task():
            response = requests.post(
                url=url,
                timeout=self.per_request_timeout,
                headers=headers,
                data=data,
            )
            if response.status_code != 200:
                raise requests.RequestException(
                    f"status code {response.status_code}"
                )
            return response

        return to_thread(task)

    async def _guarded_request(self, method, url, headers, data):
        # NetworkPolicyError propagates untouched: it must never be rewritten
        # as a generic "material unavailable" fetch error.
        response = await guarded_http_request(
            self._guarded_transport,
            self.network_policy,
            url,
            method=method,
            headers=headers,
            data=data,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"status code {response.status_code}"
            )
        return _GuardedResponseLike(response)


class _GuardedResponseLike:
    """
    Minimal adapter exposing the two ``requests.Response`` attributes the
    fetcher clients rely on (``.content`` and ``.headers``).
    """

    def __init__(self, guarded_response):
        self.content = guarded_response.content
        self.headers = guarded_response.headers
        self.status_code = guarded_response.status_code
        self.url = guarded_response.url
