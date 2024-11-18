import pytest
import re
from ginkgo_ai_client.queries import MeanEmbeddingQuery


def test_that_forgetting_to_name_arguments_raises_the_better_error_message():
    expected_error_message = re.escape(
        "Invalid initialization: MeanEmbeddingQuery does not accept unnamed arguments. "
        "Please name all inputs, for instance "
        "`MeanEmbeddingQuery(field_name=value, other_field=value, ...)`."
    )
    with pytest.raises(TypeError, match=expected_error_message):
        MeanEmbeddingQuery("MLLK<mask>P", model="ginkgo-aa0-650M")
