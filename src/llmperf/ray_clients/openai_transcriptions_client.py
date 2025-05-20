import os
import time
from pathlib import Path
from typing import Any, Dict

import ray
from openai import OpenAI
import librosa

from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient

@ray.remote
class OpenAITranscriptionsClient(LLMClient):
    """Client for OpenAI Whisper API."""

    def __init__(
        self,
    ):  # select a chat-model with audio support from https://docs.stackit.cloud/stackit/en/models-licenses-319914532.html
        base_url = os.environ[
            "OPENAI_API_BASE"
        ]  # e.g. "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
        api_key = os.environ["OPENAI_API_KEY"]  # e.g. "ey..."
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        stream = True
        model = request_config.model
        audio_file = request_config.audio_file

        assert isinstance(model, str) and model, "model requied, but not set"
        assert isinstance(audio_file, str) and model, "audio_file requied, but not set"
        assert stream == True, "transcription client only supports stream mode"
        
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        input_audio_len = 0

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        try:
            input_audio_len = librosa.get_duration(path=audio_file)

            with Path(audio_file).open(mode="rb") as f:
                time_to_next_token = []
                start_time = time.monotonic()
                most_recent_received_token_time = time.monotonic()

                response = self.client.audio.transcriptions.create(
                    file=f,
                    model=model,
                    language="en",
                    response_format="json",
                    temperature=0.0,
                    stream=stream,
                )

                if stream:
                    for chunk in response:
                        choice = chunk.choices[0]
                        if choice.get("finish_reason") or choice.get("stop_reason"):
                            continue

                        tokens_received += 1

                        delta = choice["delta"]
                        if delta.get("content", None):
                            if not ttft:
                                ttft = time.monotonic() - start_time
                                time_to_next_token.append(ttft)
                            else:
                                time_to_next_token.append(
                                    time.monotonic() - most_recent_received_token_time
                                )
                            most_recent_received_token_time = time.monotonic()
                            generated_text += delta["content"]
                else:
                    # requested as single batch
                    data = response.text

                    ttft = time.monotonic() - start_time
                    time_to_next_token.append(ttft)
                    generated_text += data["content"]

                total_request_time = time.monotonic() - start_time
                output_throughput = tokens_received / total_request_time

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(
            time_to_next_token
        )  # This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = 0
        metrics[common_metrics.INPUT_AUDIO_SECONDS] = input_audio_len

        return metrics, generated_text, request_config
