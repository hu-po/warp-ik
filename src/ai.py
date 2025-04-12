import argparse
import asyncio
import base64
import glob
import json
import logging
import os
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional, Awaitable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import aiofiles

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

DATA_PATH: str = os.path.join(os.path.dirname(__file__), "data")
log.info(f"Data path: {DATA_PATH}")

TIMEOUT_ANALYSIS: int = 600 # seconds
TIMEOUT_MODEL_API: int = 30 # seconds
AI_API_MAX_RETRIES: int = 3
AI_MAX_TOKENS: int = 4096

CLAUDE_MODEL: str = "claude-3-sonnet-20240229"
GPT_MODEL: str = "gpt-4o-mini"
GEMINI_MODEL: str = "gemini-1.5-flash"
# https://docs.x.ai/docs/overview#featured-models
XAI_MODEL: str = "grok-2-vision-1212"

PROMPT_DEFAULT: str = "describe this image in as much detail as possible"

DESIRED_MODELS: List[str] = ["claude", "gpt", "gemini", "xapi"]
AVAILABLE_MODELS: List[str] = []

try:
    from anthropic import Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        AVAILABLE_MODELS.append("claude")
    else:
        log.warning("ANTHROPIC_API_KEY not set - Claude service will be unavailable")
except ImportError:
    log.warning("Anthropic module not installed - Claude service will be unavailable")

try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        AVAILABLE_MODELS.append("gpt")
    else:
        log.warning("OPENAI_API_KEY not set - GPT service will be unavailable")
    if os.getenv("XAI_API_KEY"): # xapi mirrors openai api
        AVAILABLE_MODELS.append("xapi")
    else:
        log.warning("XAI_API_KEY not set - XAI service will be unavailable")
except ImportError:
    log.warning("OpenAI module not installed - GPT service will be unavailable")

try:
    import google.generativeai as genai
    if os.getenv("GOOGLE_API_KEY"):
        AVAILABLE_MODELS.append("gemini")
    else:
        log.warning("GOOGLE_API_KEY not set - Gemini service will be unavailable")
except ImportError:
    log.warning("Google-generativeai module not installed - Gemini service will be unavailable")

ENABLED_MODELS: List[str] = [
    model for model in DESIRED_MODELS if model in AVAILABLE_MODELS
]
log.info(f"Enabled models: {ENABLED_MODELS}")

def async_retry_decorator(*, timeout: int, max_retries: int):
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
            reraise=True,
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            attempt = 1
            try:
                while True:
                    try:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        log.info(f"{func.__name__} succeeded on attempt {attempt} after {elapsed:.2f}s")
                        return result
                    except Exception as e:
                        if attempt >= max_retries:
                            raise
                        attempt += 1
                        log.warning(f"{func.__name__} failed attempt {attempt-1}: {str(e)}")
            except asyncio.TimeoutError as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                log.error(f"{func.__name__} timed out after {elapsed:.2f}s on attempt {attempt}")
                raise TimeoutError(f"{func.__name__} timed out after {timeout} seconds")
            except Exception as e:
                elapsed = (datetime.now() - start_time).total_seconds()
                log.error(f"{func.__name__} failed after {elapsed:.2f}s on attempt {attempt}: {str(e)}")
                raise
        return wrapper
    return decorator

def encode_image(image_path: str) -> str:
    if not image_path:
        return ""
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode image: {str(e)}")

AI_MODEL_MAP: Dict[str, Callable[[str, Optional[str]], Awaitable[str]]] = {}

@async_retry_decorator(timeout=TIMEOUT_MODEL_API, max_retries=AI_API_MAX_RETRIES)
async def async_claude(prompt: str, image_path: str = None) -> str:
    try:
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        base64_image = encode_image(image_path) if image_path else ""
        client = Anthropic(api_key=api_key)
        log.info("Calling Claude API")
        log.debug(f"\n---prompt - claude\n {prompt}\n---\n")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                }
            )

        response = await asyncio.to_thread(
            client.messages.create,
            model=CLAUDE_MODEL,
            max_tokens=AI_MAX_TOKENS,
            messages=messages,
        )
        response = response.content[0].text
        log.info("Claude API responded")
        log.debug(f"\n---reply - claude\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"Claude API error: {str(e)}")
        return f"Claude API error: {str(e)}"

@async_retry_decorator(timeout=TIMEOUT_MODEL_API, max_retries=AI_API_MAX_RETRIES)
async def async_gpt(prompt: str, image_path: str = None) -> str:
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
        base64_image = encode_image(image_path) if image_path else ""
        log.info("Calling GPT API")
        log.debug(f"\n---prompt - gpt\n {prompt}\n---\n")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=GPT_MODEL,
            max_tokens=AI_MAX_TOKENS,
            messages=messages,
        )
        response = response.choices[0].message.content
        log.info("GPT API responded")
        log.debug(f"\n---reply - gpt\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"GPT API error: {str(e)}")
        return f"GPT API error: {str(e)}"

@async_retry_decorator(timeout=TIMEOUT_MODEL_API, max_retries=AI_API_MAX_RETRIES)
async def async_xapi(prompt: str, image_path: str = None) -> str:
    try:
        from openai import OpenAI
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not set")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        base64_image = encode_image(image_path) if image_path else ""
        log.info("Calling XAI API")
        log.debug(f"\n---prompt - gpt\n {prompt}\n---\n")

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=GPT_MODEL,
            max_tokens=AI_MAX_TOKENS,
            messages=messages,
        )
        response = response.choices[0].message.content
        log.info("XAI API responded")
        log.debug(f"\n---reply - xai\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"XAI API error: {str(e)}")
        return f"XAI API error: {str(e)}"

@async_retry_decorator(timeout=TIMEOUT_MODEL_API, max_retries=AI_API_MAX_RETRIES)
async def async_gemini(prompt: str, image_path: str = None) -> str:
    try:
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        uploaded_file = genai.upload_file(image_path) if image_path else None
        log.info(f"Uploaded file to Gemini: {uploaded_file.uri if uploaded_file else 'None'}")

        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        log.debug(f"\n---prompt - gemini\n {prompt}\n---\n")
        content = [uploaded_file, "\n\n", prompt] if uploaded_file else [prompt]

        response = await model.generate_content_async(
            content,
            request_options={"timeout": 600},
            generation_config={"max_output_tokens": AI_MAX_TOKENS},
        )
        response = response.text
        log.info("Gemini API responded")
        log.debug(f"\n---reply - gemini\n {response}\n---\n")
        return response
    except Exception as e:
        log.error(f"Gemini API error: {str(e)}")
        return f"Gemini API error: {str(e)}"

if "claude" in ENABLED_MODELS:
    AI_MODEL_MAP["claude"] = async_claude
if "gpt" in ENABLED_MODELS:
    AI_MODEL_MAP["gpt"] = async_gpt
if "gemini" in ENABLED_MODELS:
    AI_MODEL_MAP["gemini"] = async_gemini
if "xapi" in ENABLED_MODELS:
    AI_MODEL_MAP["xapi"] = async_xapi

async def async_analysis(
    prompt: str,
    image_path: str,
    timeout: int = TIMEOUT_ANALYSIS,
) -> None:
    log.debug("Starting AI analysis")
    try:
        if not ENABLED_MODELS:
            log.error("No AI APIs enabled")
            raise ValueError("No AI APIs enabled")

        if not os.path.exists(image_path):
            log.error(f"Image not found at {image_path}")
            raise ValueError(f"Image not found at {image_path}")

        tasks = []
        ai_models = []

        for ai_model in ENABLED_MODELS:
            tasks.append(AI_MODEL_MAP[ai_model](prompt, image_path))
            ai_models.append(ai_model)

        if not tasks:
            log.error("No tasks created - check enabled models and analyses")
            raise ValueError("No enabled models found in model map")

        log.debug(f"Executing {len(tasks)} inference tasks")
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _msg = f"Analysis timed out after {timeout} seconds"
            log.error(_msg)
            return {"error": _msg}

        log.debug("Inference results:")
        results = {}
        for ai_model, resp in zip(ai_models, responses):
            results[ai_model] = resp if not isinstance(resp, Exception) else str(resp)
            if isinstance(results[ai_model], Exception):
                log.error(f"{ai_model} failed: {results[ai_model]}")
            else:
                log.debug(f"{ai_model} success: {results[ai_model]}")

        # Get base filename from image path and add analysis suffix
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(DATA_PATH, f"{base_name}-analysis.json")
        try:
            async with aiofiles.open(json_path, "w") as f:
                await f.write(json.dumps(results, indent=2))
            log.info(f"Saved JSON results to {json_path}")
        except Exception as e:
            log.error(f"Failed to save JSON results: {e}")

    except Exception as e:
        log.error(f"AI analysis error: {str(e)}", exc_info=True)
        return {"error": f"AI analysis failed: {str(e)}"}

async def async_test_model_apis() -> None:
    log.info(f"Testing enabled models: {ENABLED_MODELS}")
    for model_name in ENABLED_MODELS:
        try:
            log.info(f"Testing {model_name}")
            prompt = "ur favorite emoji"
            response = await AI_MODEL_MAP[model_name](prompt)
            log.info(f"{model_name.upper()} Response: {response}")
        except Exception as e:
            log.error(f"Error testing {model_name}: {str(e)}")

async def analyze_directory(data_dir: str, prompt: str) -> None:
    image_files = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    log.info(f"Found {len(image_files)} images in {data_dir}")
    for image_path in image_files:
        log.info(f"Analyzing {image_path}")
        await async_analysis(prompt, image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI model tests or analyze images")
    parser.add_argument("--test", action="store_true", help="Run model API tests")
    parser.add_argument("--prompt", type=str, default=PROMPT_DEFAULT, help="Prompt for image analysis")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    async def main():
        if args.test:
            await async_test_model_apis()
        await analyze_directory(DATA_PATH, args.prompt)

    asyncio.run(main())