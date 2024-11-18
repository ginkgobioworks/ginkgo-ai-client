import os
import itertools
from typing import List, Optional, Any, Literal, Dict, Iterator, Union
import time
import warnings
from concurrent import futures

from tqdm import tqdm
import requests

from ginkgo_ai_client.queries import QueryBase


class RequestError(Exception):
    """An exception raised by a request, due to the query content or a system error.

    This exception carries the original query and the result url to enable users to
    better handle failure cases.

    Parameters
    ----------
    cause: Exception
        The original exception that caused the request to fail.

    query: QueryBase (optional)
        The query that failed. This enables users to retrieve and re-try the failed
        queries in a batch query

    result_url: str (optional)
        The url where the result can be retrieved from. This enables users to get the
        result later if the failure cause was a temporary network error or an
        accidental timeout.
    """

    def __init__(
        self,
        cause: Exception,
        query: Optional[QueryBase] = None,
        result_url: Optional[str] = None,
    ):
        self.cause = cause
        self.query = query
        self.result_url = result_url
        message = self._format_error_message()
        super().__init__(message)

    def _format_error_message(self):
        cause_str = f"{self.cause.__class__.__name__}: {self.cause}"
        url_str = (
            f"\n\nThis happened while polling this url for results:\n{self.result_url}"
            if self.result_url is not None
            else ""
        )
        return f"{cause_str}\n on query: {self.query}{url_str}"


class GinkgoAIClient:
    """A client for the public Ginkgo AI models API.

    Parameters
    ----------
    api_key: str (optional)
        The API key to use for the Ginkgo AI API. If none is provided, the
        `GINKGOAI_API_KEY` environment variable will be used.

    polling_delay: float (default: 1)
        The delay between polling requests to the Ginkgo AI API, in seconds.

    Examples
    --------

    .. code-block:: python


        client = GinkgoAIClient()
        query = MaskedInferenceQuery("MPK<mask><mask>RRL", model="ginkgo-aa0-650m")
        response = client.send_request(query)
        # response["sequence"] == "MPKYLRRL"
        responses = client.send_batch_request([query_params, other_query_params])

    """

    INFERENCE_URL = "https://api.ginkgobioworks.ai/v1/transforms/run"
    BATCH_INFERENCE_URL = "https://api.ginkgobioworks.ai/v1/batches/transforms/run"

    def __init__(
        self,
        api_key: Optional[str] = None,
        polling_delay: float = 1,
    ):
        if api_key is None:
            api_key = os.environ.get("GINKGOAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Please provide an API key or set the "
                    "`GINKGOAI_API_KEY` environment variable."
                )
        self.api_key = api_key
        self.polling_delay = polling_delay

    def send_request(
        self,
        query: QueryBase,
        timeout: float = 60,
    ) -> Any:
        """Send a query to the Ginkgo AI API.

        Parameters
        ----------
        query: QueryBase
            The query to send to the Ginkgo AI API.

        timeout: float (default: 60)
            The maximum time to wait for the query to complete, in seconds.

        Returns
        -------
        Response
            The response from the Ginkgo AI API, for instance `{"sequence": "ATG..."}`.
            It will be different depending on the query, see the different docstrings
            of the helper methods ending in `*_params`.

        Raises
        ------
        RequestError
            If the request failed due to the query content or a system error. The
            exception carries the original query and (if it reached that stage) the
            url it was polling for the results.
        """

        # LAUNCH THE REQUEST

        headers = {"x-api-key": self.api_key}
        json = query.to_request_params()
        launch_response = requests.post(self.INFERENCE_URL, headers=headers, json=json)
        if not launch_response.ok:
            status_code, text = (launch_response.status_code, launch_response.text)
            # TODO: add different error types depending on exception
            cause = IOError(
                f"Request failed at launch with status {status_code}: {text}."
            )
            raise RequestError(query=query, cause=cause)

        # GET THE JOBID FROM THE RESPONSE

        launch_response = launch_response.json()
        response_has_result = "result" in launch_response
        if not (response_has_result) and ("?jobId" in launch_response["result"]):
            cause = IOError(f"Unexpected response format at launch: {launch_response}")
            raise RequestError(query=query, cause=cause)
        result_url = launch_response["result"]

        # POLL UNTIL THE JOB COMPLETES, AN ERROR OCCURS, OR WE TIME OUT.

        start_time = time.time()
        while True:
            time.sleep(self.polling_delay)
            poll_response = requests.get(result_url, headers=headers)
            assert poll_response.ok, f"Unexpected response: {poll_response}"
            poll_response = poll_response.json()
            if poll_response["status"] == "COMPLETE":
                job_result = poll_response["result"][0]
                if job_result["error"] is not None:
                    cause = IOError(f"Query returned an error: {job_result['error']}")
                    raise RequestError(query=query, cause=cause, result_url=result_url)

                # If we reach this point, we have a result, let's parse it and return it.
                response_result = poll_response["result"][0]["result"]
                return query.parse_response(response_result)
            elif poll_response["status"] in ["PENDING", "IN_PROGRESS"]:
                if time.time() - start_time > timeout:
                    cause = TimeoutError("Request timed out.")
                    raise RequestError(query=query, cause=cause, result_url=result_url)
            else:
                cause = IOError(f"Unexpected response status: {poll_response}")
                raise RequestError(query=query, cause=cause, result_url=result_url)

    def send_batch_request(
        self,
        queries: List[QueryBase],
        timeout: float = None,
        on_failed_queries: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> List[Any]:
        """Query the Ginkgo AI API in batch mode.

        Parameters
        ----------
        queries: list of dict
            The parameters of the queries (depends on the model used) used to send to the
            Ginkgo AI API. These will typically be generated using the helper methods
            in `ginkgo_ai_client.queries`.

        timeout: float (optional)
            The maximum time to wait for the batch to complete, in seconds.

        on_failed_queries: Literal["ignore", "warn", "raise"] = "ignore"
            What to do if some queries fail. The default is to ignore the failures (the
            user will have to check and handle the queries themselves). "warn" will print
            a warning and return the failed queries, "raise" will raise an exception if
            at least one query failed.

        Returns
        -------
        responses : List[Any]
            A list of responses from the Ginkgo AI API. the class of the responses
            depends on the class of the queries. If some

        Raises
        ------
        RequestException
            If the request failed due to the query content or a system error.


        Examples
        --------

        .. code-block:: python

            client = GinkgoAIClient()
            queries = [
                MaskedInferenceQuery("MPK<mask><mask>RRL", model="ginkgo-aa0-650m"),
                MaskedInferenceQuery("MES<mask><mask>YKL", model="ginkgo-aa0-650m")
            ]
            responses = client.send_batch_request(queries)

        """

        # LAUNCH THE BATCH REQUEST

        headers = {"x-api-key": self.api_key}
        request_params = [q.to_request_params() for q in queries]
        launch_response = requests.post(
            self.BATCH_INFERENCE_URL, headers=headers, json={"requests": request_params}
        )
        if not launch_response.status_code == 200:
            status_code, text = (launch_response.status_code, launch_response.text)
            raise Exception(f"Batch request failed with status {status_code}: {text}")

        # GET THE BATCHID FROM THE RESPONSE

        launch_response = launch_response.json()
        response_has_result = "result" in launch_response

        if not response_has_result or "?batchId" not in launch_response["result"]:
            raise Exception(f"Unexpected response: {launch_response}")
        ordered_job_ids = launch_response["jobIds"]
        result_url = launch_response["result"]

        # POLL UNTIL THE BATCH COMPLETES, AN ERROR OCCURS, OR WE TIME OUT.

        start_time = time.time()
        while True:
            time.sleep(self.polling_delay)
            poll_response = requests.get(result_url, headers=headers)
            if not poll_response.ok:
                cause = IOError(f"Unexpected response: {poll_response}")
                raise RequestError(
                    cause=cause,
                    result_url=result_url,
                )
            poll_response = poll_response.json()
            if poll_response["status"] == "COMPLETE":
                return self._process_batch_request_results(
                    queries=queries,
                    ordered_job_ids=ordered_job_ids,
                    response=poll_response,
                    failed_queries_action=on_failed_queries,
                )
            elif poll_response["status"] in ["PENDING", "IN_PROGRESS"]:
                if timeout is not None and (time.time() - start_time > timeout):
                    cause = TimeoutError(f"Batch request took over {timeout} seconds.")
                    raise RequestError(
                        cause=cause,
                        result_url=result_url,
                    )
            else:
                raise Exception(f"Unexpected response status: {poll_response}")

    def send_requests_by_batches(
        self,
        queries: Union[List[QueryBase], Iterator[QueryBase]],
        batch_size: int = 20,
        timeout: float = None,
        on_failed_queries: Literal["ignore", "warn", "raise"] = "ignore",
        max_concurrent: int = 3,
        show_progress: bool = True,
    ):
        # Create batch iterator
        query_iterator = iter(queries)
        batches = iter(lambda: list(itertools.islice(query_iterator, batch_size)), [])
        try:
            # will only work for lists or ranges() or iterators with a len() such as
            # our IteratorWithLength which some of our utility methods return
            n_queries = len(queries)
            total = n_queries // batch_size + (1 if n_queries % batch_size else 0)
        except Exception:
            total = None

        if show_progress:
            progress_bar = tqdm(total=total, desc="Processing batches", mininterval=1)
        else:
            progress_bar = None

        def send_batch(batch):
            return self.send_batch_request(
                queries=batch, timeout=timeout, on_failed_queries=on_failed_queries
            )

        for result in process_with_limited_concurrency(
            element_iterator=batches,
            function=send_batch,
            max_concurrent=max_concurrent,
            progress_bar=progress_bar,
        ):
            yield result

    @classmethod
    def _process_batch_request_results(
        cls,
        queries: List[QueryBase],
        ordered_job_ids: List[str],
        response: Dict,
        failed_queries_action: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> List[Any]:
        # Technical note: to understand the parsing, one should know that the
        # API returns a list of request results where each element is of the form
        # {"jobId": "...", "result": [{"error": None, "result": ...}]}
        ordered_results = sorted(
            response["requests"],
            key=lambda x: ordered_job_ids.index(x["jobId"]),
        )

        parsed_results = [
            cls._parse_batch_request_result(query, request_result["result"][0])
            for query, request_result in zip(queries, ordered_results)
        ]

        if failed_queries_action != "ignore":
            errored_results = [r for r in parsed_results if isinstance(r, RequestError)]
            n_errored = len(errored_results)
            n_queries = len(queries)
            if len(errored_results):
                msg = f"{n_errored}/{n_queries} queries in the batch failed"
                if failed_queries_action == "warn":
                    warnings.warn(msg)
                else:
                    raise IOError(msg)
        return parsed_results

    @staticmethod
    def _parse_batch_request_result(query: QueryBase, result: Dict) -> Any:
        if result["error"] is not None:
            return RequestError(
                cause=Exception(result["error"]),
                query=query,
            )
        return query.parse_response(result["result"])


def process_with_limited_concurrency(
    element_iterator, function, max_concurrent: int, progress_bar: Optional = None
):
    with futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:

        # Initialize active futures
        active_futures = []

        # Submit initial batch of futures up to max_concurrent
        for _ in range(max_concurrent):
            try:
                element = next(element_iterator)
                future = executor.submit(function, element)
                active_futures.append(future)
            except StopIteration:
                break

        # Process results and submit new batches as old ones complete
        while active_futures:
            # Wait for the first future to complete
            FC = futures.FIRST_COMPLETED

            completed, active_futures = (
                futures.wait(active_futures, return_when=FC)[0],
                list(futures.wait(active_futures, return_when=FC)[1]),
            )

            # Submit a new batch if there are any left
            try:
                element = next(element_iterator)
                future = executor.submit(function, element)
                active_futures.append(future)
            except StopIteration:
                pass

            # Yield completed results and update progress
            for future in completed:
                yield future.result()
                if progress_bar is not None:
                    progress_bar.update(1)
        if progress_bar is not None:
            progress_bar.close()
