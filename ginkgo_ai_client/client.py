import os
import requests
from typing import List, Dict, Optional
import time


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
        query_params = aa0_masked_inference("MPK<mask><mask>RRL")
        response = client.query(query_params)
        # response["sequence"] == "MPKYLRRL"
        responses = client.batch_query([query_params, other_query_params])

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

    def query(
        self,
        params: Dict,
        timeout: float = 60,
    ) -> Dict:
        """
        Parameters
        ----------
        params: dict
            The parameters of the query (depends on the model used) used to send to the
            Ginkgo AI API. These will typically be generated using the helper methods
            ending in `*_params`.

        timeout: float (default: 60)
            The maximum time to wait for the query to complete, in seconds.

        Returns
        -------
        dict
            The response from the Ginkgo AI API, for instance `{"sequence": "ATG..."}`.
            It will be different depending on the query, see the different docstrings
            of the helper methods ending in `*_params`.
        """
        headers = {"x-api-key": self.api_key}
        launch_response = requests.post(
            self.INFERENCE_URL, headers=headers, json=params
        )
        if not launch_response.ok:
            status_code, text = (launch_response.status_code, launch_response.text)
            raise Exception(
                f"Request failed with status code {status_code}: {text}. Request: {params}"
            )
        launch_response = launch_response.json()
        assert ("result" in launch_response) and (
            "?jobId" in launch_response["result"]
        ), f"Unexpected response: {launch_response}"

        # Poll until the job completes, an error occurs, or we time out.
        initial_time = time.time()
        while True:
            time.sleep(self.polling_delay)
            poll_response = requests.get(launch_response["result"], headers=headers)
            assert poll_response.ok, f"Unexpected response: {poll_response}"
            poll_response = poll_response.json()
            if poll_response["status"] == "COMPLETE":
                job_result = poll_response["result"][0]
                if job_result["error"] is not None:
                    raise IOError(f"Query returned an error: {job_result['error']}")
                return poll_response["result"][0]["result"]
            elif poll_response["status"] in ["PENDING", "IN_PROGRESS"]:
                if time.time() - initial_time > timeout:
                    raise Exception(
                        f"Timeout while waiting for request to complete.\n"
                        f"Request:{params}]\nJob:{launch_response}"
                    )
            else:
                raise Exception(f"Unexpected response status: {poll_response}")

    def batch_query(self, params_list: List[Dict], timeout: float = None) -> List[Dict]:
        """Query the Ginkgo AI API in batch mode.

        Parameters
        ----------
        params_list: list of dict
            The parameters of the queries (depends on the model used) used to send to the
            Ginkgo AI API. These will typically be generated using the helper methods
            in `ginkgo_ai_client.queries`.

        timeout: float (optional)
            The maximum time to wait for the batch to complete, in seconds.

        Returns
        -------
        list of dict
            The responses from the Ginkgo AI API. It will be different depending on the
            query, see the different docstrings in `ginkgo_ai_client.queries`.
        """
        headers = {"x-api-key": self.api_key}
        launch_response = requests.post(
            self.BATCH_INFERENCE_URL, headers=headers, json={"requests": params_list}
        )
        if not launch_response.status_code == 200:
            status_code, text = (launch_response.status_code, launch_response.text)
            raise Exception(
                f"Batch request failed with status code {status_code}: {text}"
            )
        launch_response = launch_response.json()
        assert ("result" in launch_response) and (
            "?batchId" in launch_response["result"]
        ), f"Unexpected response: {launch_response}"
        ordered_job_ids = launch_response["jobIds"]

        # Poll until the job completes, an error occurs, or we time out.
        initial_time = time.time()
        while True:
            time.sleep(self.polling_delay)
            poll_response = requests.get(launch_response["result"], headers=headers)
            assert poll_response.ok, f"Unexpected response: {poll_response}"
            poll_response = poll_response.json()
            if poll_response["status"] == "COMPLETE":
                ordered_requests = sorted(
                    poll_response["requests"],
                    key=lambda x: ordered_job_ids.index(x["jobId"]),
                )
                return [r["result"][0] for r in ordered_requests]
            elif poll_response["status"] in ["PENDING", "IN_PROGRESS"]:
                if timeout is not None and (time.time() - initial_time > timeout):
                    raise Exception(
                        f"Timeout while waiting for batch_request to complete.\n"
                        f"BatchId:{launch_response['batchId']}"
                    )
            else:
                raise Exception(f"Unexpected response status: {poll_response}")
