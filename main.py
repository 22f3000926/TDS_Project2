# main.py
import os
import io
import time
import json
import asyncio
import csv
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from pypdf import PdfReader  # pip package pypdf

load_dotenv()

# CONFIG / ENV
SECRET_KEY = os.getenv("SECRET_KEY")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")  # required for LLM calls
AIPIPE_URL = os.getenv("AIPIPE_URL")

if SECRET_KEY is None:
    raise RuntimeError("SECRET_KEY environment variable not set")

app = FastAPI(title="TDS Quiz Solver")

# Limits
MAX_TOTAL_SECONDS = 180  # 3 minutes per incoming POST
MAX_PAYLOAD_BYTES = 1_000_000  # 1MB


# Request model
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


# ------------------------------
# LLM helper: ask for structured JSON answer
# ------------------------------
async def call_llm_for_json(prompt: str, system: Optional[str] = None, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN is not set in environment")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or "You are a helpful data assistant. Return a single valid JSON object only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1600,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(AIPIPE_URL, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPIPE_TOKEN}"
        }, json=payload)

    try:
        r.raise_for_status()
    except Exception as e:
        text = r.text
        raise RuntimeError(f"LLM request failed: {e}; response: {text[:1000]}")

    data = r.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected LLM response: {data}")

    content = data["choices"][0].get("message", {}).get("content", "")
    m = re.search(r"\{[\s\S]*\}$", content.strip())
    json_text = content if m is None else m.group(0)

    try:
        return json.loads(json_text)
    except:
        m2 = re.search(r"\{[\s\S]*?\}", content)
        if m2:
            try:
                return json.loads(m2.group(0))
            except:
                pass
        raise RuntimeError(f"Failed to parse JSON from LLM output. Raw output (truncated): {content[:1000]}")


# ------------------------------
# NEW — AIPipe Speech-to-Text API
# ------------------------------
async def transcribe_audio_via_aipipe(audio_bytes: bytes) -> str:
    """
    Use AIPipe's OpenAI-compatible STT endpoint:
    POST https://aipipe.org/openai/v1/audio/transcriptions
    """
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN not set")

    async with httpx.AsyncClient(timeout=60) as client:
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
            "model": (None, "gpt-4o-transcribe")
        }
        resp = await client.post(
            "https://aipipe.org/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
            files=files
        )

    try:
        resp.raise_for_status()
    except:
        raise RuntimeError(f"Transcription error: {resp.text[:500]}")

    data = resp.json()
    return data.get("text", "")


# ------------------------------
# Utility: download files
# ------------------------------
async def fetch_binary(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


def parse_csv_bytes(b: bytes, nrows: int = 200) -> List[Dict[str, str]]:
    text = b.decode(errors="replace")
    reader = csv.DictReader(text.splitlines())
    rows = []
    for i, row in enumerate(reader):
        rows.append(row)
        if i + 1 >= nrows:
            break
    return rows


def extract_text_from_pdf_bytes(b: bytes, max_chars: int = 20000) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
    except Exception:
        try:
            return b.decode(errors="replace")[:max_chars]
        except:
            return ""
    text = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except:
            t = ""
        text.append(t)
        if sum(len(x) for x in text) > max_chars:
            break
    return "\n".join(text)[:max_chars]


# ------------------------------
# Playwright Extraction Scripts
# ------------------------------
PAGE_EXTRACT_SCRIPT = r"""
() => {
  const selectors = ['#result', '.question', '.quiz', '.task', 'main', 'article', 'h1', 'h2', 'p'];
  for (const s of selectors) {
    const el = document.querySelector(s);
    if (el && el.innerText && el.innerText.trim().length > 10) {
      return { kind: 'element', selector: s, text: el.innerText.trim() };
    }
  }
  return { kind: 'body', text: document.body.innerText || '' };
}
"""

PAGE_FIND_SUBMIT = r"""
() => {
  let f = document.querySelector('form[action]');
  if (f && f.action) return f.action;

  let anchors = Array.from(document.querySelectorAll('a[href]'));
  for (const a of anchors) {
    if (/submit|answer|post|api/i.test(a.href)) return a.href;
  }

  const blocks = Array.from(document.querySelectorAll('pre, script'));
  for (const b of blocks) {
    const t = b.textContent || '';
    const m = t.match(/https?:\/\/[^\s'"]+\/[^\s'"]*/i);
    if (m) return m[0];
  }

  const t = document.body.innerText || '';
  const m2 = t.match(/https?:\/\/[^\s'"]+\/(?:submit|answer|api|post)[^\s'"]*/i);
  if (m2) return m2[0];

  return null;
}
"""


# ------------------------------
# Solve one page
# ------------------------------
# ------------------------------
# Visualization Support
# ------------------------------
def generate_visualization(viz_spec: Dict[str, Any], data: Any) -> str:
    """
    Generate a chart/visualization and return as base64 data URI.
    
    Args:
        viz_spec: Specification with keys: type, title, x_label, y_label, etc.
        data: Data to visualize (list of dicts, pandas-like structure, etc.)
    
    Returns:
        Base64 data URI string (e.g., "data:image/png;base64,iVBOR...")
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    viz_type = viz_spec.get("type", "bar").lower()
    title = viz_spec.get("title", "Chart")
    x_label = viz_spec.get("x_label", "X")
    y_label = viz_spec.get("y_label", "Y")
    
    try:
        # Handle different data formats
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dicts (most common from CSV/JSON)
            x_key = viz_spec.get("x_key") or list(data[0].keys())[0]
            y_key = viz_spec.get("y_key") or list(data[0].keys())[1]
            
            x_vals = [row.get(x_key) for row in data]
            y_vals = [float(row.get(y_key, 0)) if row.get(y_key) is not None else 0 for row in data]
            
        elif isinstance(data, dict) and "x" in data and "y" in data:
            # Direct x, y arrays
            x_vals = data["x"]
            y_vals = data["y"]
        else:
            raise ValueError("Unsupported data format for visualization")
        
        # Generate chart based on type
        if viz_type in ["bar", "column"]:
            ax.bar(range(len(x_vals)), y_vals)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=45, ha='right')
            
        elif viz_type == "line":
            ax.plot(x_vals, y_vals, marker='o')
            
        elif viz_type == "scatter":
            ax.scatter(x_vals, y_vals)
            
        elif viz_type == "pie":
            ax.pie(y_vals, labels=x_vals, autopct='%1.1f%%')
            ax.axis('equal')
            
        elif viz_type == "histogram":
            ax.hist(y_vals, bins=viz_spec.get("bins", 10))
            
        else:
            # Default to bar chart
            ax.bar(range(len(x_vals)), y_vals)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=45, ha='right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64 data URI
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Visualization generation failed: {e}")


async def check_and_generate_visualization(question_text: str, resource_summaries: List[Dict], llm_answer: Any) -> Optional[str]:
    """
    Check if the question requires a visualization and generate it if needed.
    
    Returns:
        Base64 data URI if visualization was generated, None otherwise
    """
    # Keywords that suggest visualization is needed
    viz_keywords = [
        "chart", "graph", "plot", "visualiz", "diagram",
        "bar chart", "line chart", "pie chart", "scatter plot",
        "histogram", "show trend", "display data"
    ]
    
    question_lower = question_text.lower()
    needs_viz = any(keyword in question_lower for keyword in viz_keywords)
    
    if not needs_viz:
        return None
    
    print("[solver] Detected visualization requirement")
    
    # Try to extract data from resources
    data_for_viz = None
    
    for res in resource_summaries:
        if res["type"] == "csv":
            try:
                data_for_viz = json.loads(res["snippet"])
                break
            except:
                pass
        elif res["type"] == "json":
            try:
                data_for_viz = json.loads(res["snippet"])
                if isinstance(data_for_viz, list):
                    break
            except:
                pass
    
    if not data_for_viz:
        print("[solver] No suitable data found for visualization")
        return None
    
    # Ask LLM for visualization specification
    viz_prompt = f"""
Based on this question: "{question_text[:500]}"

And this data sample: {json.dumps(data_for_viz[:20] if isinstance(data_for_viz, list) else data_for_viz)[:2000]}

Generate a JSON specification for the required visualization:
{{
    "type": "bar|line|scatter|pie|histogram",
    "title": "Chart Title",
    "x_key": "column_name_for_x_axis",
    "y_key": "column_name_for_y_axis", 
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label",
    "bins": 10  // only for histogram
}}

Return ONLY valid JSON.
"""
    
    try:
        viz_spec = await call_llm_for_json(
            viz_prompt,
            system="Return only a JSON object with visualization specification."
        )
        
        # Generate the visualization
        data_uri = generate_visualization(viz_spec, data_for_viz)
        print(f"[solver] ✓ Generated {viz_spec.get('type', 'unknown')} chart")
        return data_uri
        
    except Exception as e:
        print(f"[solver] Visualization generation failed: {e}")
        return None


# ------------------------------
# Solve one page with retry logic
# ------------------------------
async def solve_one_page(url: str, email: str, secret: str, start_ts: float, max_retries: int = 2) -> Dict[str, Any]:
    """
    Solve a single quiz page with retry logic for incorrect answers.
    
    Args:
        url: Quiz page URL to solve
        email: User email
        secret: Secret key for authentication
        start_ts: Start timestamp for timeout tracking
        max_retries: Maximum number of retry attempts for wrong answers
    
    Returns:
        Dict containing submit_url, submit_response, and llm_plan
    """
    
    # Step 1: Extract page content and resources (do this once)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
        except PlaywrightTimeoutError:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        await page.wait_for_timeout(300)

        # Extract question text
        try:
            qres = await page.evaluate(PAGE_EXTRACT_SCRIPT)
            question_text = qres.get("text", "") if isinstance(qres, dict) else str(qres)
        except:
            question_text = await page.content()

        # Find submit URL
        submit_url = await page.evaluate(PAGE_FIND_SUBMIT)

        # Find resource links
        resource_links = await page.evaluate("""() => {
            const urls = [];
            for (const a of Array.from(document.querySelectorAll('a[href]'))) {
                const href = a.href || a.getAttribute('href');
                if (!href) continue;
                if (href.match(/\\.csv$|\\.json$|\\.pdf$|\\.xlsx$|\\.wav$|\\.mp3$|\\.ogg$|\\.m4a$/i)
                    || href.toLowerCase().includes('download')
                    || href.toLowerCase().includes('audio')
                ) {
                    urls.push(href);
                }
            }
            return urls;
        }""")

        await browser.close()

    # Step 2: Download and process resources (do this once)
    resource_summaries = []
    for res_link in (resource_links or [])[:3]:  # Increased from 2 to 3
        try:
            # Check timeout before downloading each resource
            if time.monotonic() - start_ts > MAX_TOTAL_SECONDS - 30:
                print(f"[solver] Skipping resource {res_link} due to time constraint")
                break
                
            b = await fetch_binary(res_link)

            if re.search(r"\.json($|\?)", res_link, re.I):
                try:
                    j = json.loads(b.decode(errors="replace"))
                    snippet = json.dumps(j, indent=2)[:6000]  # Increased from 4000
                except:
                    snippet = b.decode(errors="replace")[:6000]
                resource_summaries.append({"url": res_link, "type": "json", "snippet": snippet})

            elif re.search(r"\.csv($|\?)", res_link, re.I):
                rows = parse_csv_bytes(b, 300)  # Increased from 200
                resource_summaries.append({"url": res_link, "type": "csv", "snippet": json.dumps(rows, indent=2)[:6000]})

            elif re.search(r"\.pdf($|\?)", res_link, re.I):
                text = extract_text_from_pdf_bytes(b, max_chars=30000)  # Increased from 20000
                resource_summaries.append({"url": res_link, "type": "pdf", "snippet": text[:6000]})

            elif re.search(r"\.(wav|mp3|m4a|ogg)$", res_link, re.I):
                text = await transcribe_audio_via_aipipe(b)
                resource_summaries.append({"url": res_link, "type": "audio", "snippet": text[:6000]})

            else:
                resource_summaries.append({"url": res_link, "type": "bin", "snippet": str(b[:200])})

        except Exception as e:
            resource_summaries.append({"url": res_link, "type": "error", "snippet": f"fetch error: {e}"})

    # Step 3: Retry loop for solving
    last_response = None
    previous_attempts = []
    
    for attempt in range(max_retries + 1):
        # Check timeout
        elapsed = time.monotonic() - start_ts
        if elapsed > MAX_TOTAL_SECONDS - 20:  # Leave 20s buffer
            raise RuntimeError(f"Timeout: {elapsed:.1f}s elapsed, stopping retries")
        
        print(f"[solver] Attempt {attempt + 1}/{max_retries + 1} for {url}")
        
        # Build prompt for LLM
        prompt_parts = [
            f"Quiz page URL: {url}",
            f"Attempt: {attempt + 1}/{max_retries + 1}",
            "",
            "Question or page text:",
            question_text[:8000]
        ]

        # Add resource summaries
        if resource_summaries:
            prompt_parts.append("\n=== AVAILABLE RESOURCES ===")
            for r in resource_summaries:
                prompt_parts.append(
                    f"\nURL: {r['url']}\nType: {r['type']}\nContent:\n{r['snippet']}\n{'-'*50}"
                )

        # Add feedback from previous attempts
        if previous_attempts:
            prompt_parts.append("\n=== PREVIOUS ATTEMPTS (LEARN FROM THESE) ===")
            for i, prev in enumerate(previous_attempts):
                prompt_parts.append(f"\nAttempt {i + 1}:")
                prompt_parts.append(f"Answer submitted: {json.dumps(prev['answer'])}")
                prompt_parts.append(f"Result: {'CORRECT' if prev.get('correct') else 'INCORRECT'}")
                if prev.get('reason'):
                    prompt_parts.append(f"Feedback: {prev['reason']}")
                prompt_parts.append(f"{'-'*50}")

        # Instructions for LLM
        prompt_parts.append(
            "\n=== INSTRUCTIONS ===\n"
            "Analyze the question, resources, and any previous attempt feedback carefully.\n"
            "If previous attempts failed, DO NOT repeat the same answer or approach.\n"
            "\n"
            "VISUALIZATION DETECTION:\n"
            "- If the question asks for a chart, graph, plot, or visualization, indicate this\n"
            "- The system will automatically generate the visualization if needed\n"
            "- You can specify viz requirements in your explanation field\n"
            "\n"
            "Return a single valid JSON object ONLY with these keys:\n"
            '- "answer" (REQUIRED): Your answer (can be boolean, number, string, object, base64 data URI for images)\n'
            '- "submit_url" (optional): Override submit URL if you found a better one\n'
            '- "explanation" (optional): Brief explanation of your reasoning\n'
            '- "confidence" (optional): Your confidence level 0-100\n'
            '- "needs_visualization" (optional): Set to true if answer should be a chart/graph\n'
            "\n"
            "The entire JSON payload must be under 1MB.\n"
            "Think carefully and provide the most accurate answer possible."
        )

        final_prompt = "\n".join(prompt_parts)[:20000]  # Increased context window

        # Call LLM
        system_prompt = (
            "You are an expert data analyst solving quiz questions. "
            "Analyze all provided data carefully. Learn from previous mistakes. "
            "Return ONLY a valid JSON object with keys: answer (required), submit_url (optional), "
            "explanation (optional), confidence (optional)."
        )
        
        try:
            plan = await call_llm_for_json(
                final_prompt,
                system=system_prompt,
                model="gpt-4o-mini"  # Consider using gpt-4o for harder problems
            )
        except Exception as e:
            if attempt < max_retries:
                print(f"[solver] LLM call failed on attempt {attempt + 1}, retrying: {e}")
                await asyncio.sleep(1)
                continue
            raise RuntimeError(f"LLM failed after {attempt + 1} attempts: {e}")

        # Determine submit URL
        llm_submit_url = plan.get("submit_url")
        final_submit_url = llm_submit_url or submit_url

        if final_submit_url and not final_submit_url.startswith("http"):
            final_submit_url = urljoin(url, final_submit_url)

        if not final_submit_url:
            raise RuntimeError("No submit URL found on page nor provided by LLM.")

        # Get answer
        answer_obj = plan.get("answer")
        
        # Check if visualization is needed and not already provided
        if answer_obj is None or (isinstance(answer_obj, str) and not answer_obj.startswith("data:")):
            viz_data_uri = await check_and_generate_visualization(question_text, resource_summaries, answer_obj)
            if viz_data_uri:
                # If LLM didn't provide an answer but we generated a viz, use that
                if answer_obj is None:
                    answer_obj = viz_data_uri
                # If LLM provided non-viz answer but viz is clearly needed, replace it
                elif any(kw in question_text.lower() for kw in ["chart", "graph", "plot", "visualiz"]):
                    answer_obj = viz_data_uri
        
        if answer_obj is None:
            if attempt < max_retries:
                print(f"[solver] LLM returned no answer on attempt {attempt + 1}, retrying")
                previous_attempts.append({
                    "answer": None,
                    "correct": False,
                    "reason": "LLM did not provide an answer"
                })
                continue
            raise RuntimeError("LLM did not provide an answer field")

        # Build submission payload
        payload = {
            "email": email,
            "secret": secret,
            "url": url,
            "answer": answer_obj
        }

        # Check payload size
        payload_bytes = json.dumps(payload).encode("utf-8")
        if len(payload_bytes) > MAX_PAYLOAD_BYTES:
            raise RuntimeError(f"Submission payload ({len(payload_bytes)} bytes) exceeds 1MB limit")

        # Submit answer
        print(f"[solver] Submitting to {final_submit_url}")
        print(f"[solver] Answer preview: {str(answer_obj)[:200]}")
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(final_submit_url, json=payload)

                try:
                    resp.raise_for_status()
                except:
                    # Even with error status, try to parse JSON response
                    try:
                        error_json = resp.json()
                        if attempt < max_retries:
                            print(f"[solver] Submit failed with {resp.status_code}: {error_json}")
                            previous_attempts.append({
                                "answer": answer_obj,
                                "correct": False,
                                "reason": f"HTTP {resp.status_code}: {error_json}"
                            })
                            await asyncio.sleep(1)
                            continue
                    except:
                        pass
                    raise RuntimeError(f"Submit returned {resp.status_code}: {resp.text[:500]}")

                try:
                    resp_json = resp.json()
                except:
                    resp_json = {"raw_text": resp.text[:1000], "status_code": resp.status_code}

        except httpx.RequestError as e:
            if attempt < max_retries:
                print(f"[solver] Network error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2)
                continue
            raise RuntimeError(f"Network error submitting answer: {e}")

        last_response = resp_json

        # Check if answer was correct
        is_correct = resp_json.get("correct", False) if isinstance(resp_json, dict) else False
        
        if is_correct:
            print(f"[solver] ✓ Answer CORRECT on attempt {attempt + 1}")
            return {
                "submit_url": final_submit_url,
                "submit_response": resp_json,
                "llm_plan": plan,
                "attempts": attempt + 1,
                "success": True
            }
        
        # Answer was incorrect
        reason = resp_json.get("reason", "No reason provided") if isinstance(resp_json, dict) else "Unknown error"
        print(f"[solver] ✗ Answer INCORRECT on attempt {attempt + 1}: {reason}")
        
        # Store this attempt for next iteration
        previous_attempts.append({
            "answer": answer_obj,
            "correct": False,
            "reason": reason
        })
        
        # Check if we got a next URL even though answer was wrong
        # (according to requirements, this can happen)
        if isinstance(resp_json, dict) and resp_json.get("url"):
            print(f"[solver] Received next URL despite incorrect answer: {resp_json['url']}")
            # Still return - the background_solve_chain will handle the next URL
            return {
                "submit_url": final_submit_url,
                "submit_response": resp_json,
                "llm_plan": plan,
                "attempts": attempt + 1,
                "success": False
            }
        
        # If this was the last attempt, return with failure
        if attempt >= max_retries:
            print(f"[solver] Max retries ({max_retries}) exhausted")
            return {
                "submit_url": final_submit_url,
                "submit_response": resp_json,
                "llm_plan": plan,
                "attempts": attempt + 1,
                "success": False,
                "all_attempts": previous_attempts
            }
        
        # Wait before retry
        await asyncio.sleep(1)
    
    # Should never reach here, but just in case
    return {
        "submit_url": final_submit_url,
        "submit_response": last_response,
        "llm_plan": plan,
        "attempts": max_retries + 1,
        "success": False
    }
# ------------------------------
# Background solver chain
# ------------------------------
async def background_solve_chain(initial_url: str, email: str, secret: str):
    start = time.monotonic()
    current = initial_url
    last_resp = None

    while current and (time.monotonic() - start) < MAX_TOTAL_SECONDS:
        try:
            result = await solve_one_page(current, email, secret, start)
        except Exception as e:
            print("[solver] error on", current, ":", repr(e))
            return {"ok": False, "error": str(e), "last_response": last_resp}

        last_resp = result.get("submit_response")

        if isinstance(last_resp, dict) and last_resp.get("url"):
            next_url = last_resp["url"]
            print("[solver] received next url:", next_url)
            current = next_url
            continue

        return {"ok": True, "final_response": last_resp}

    return {"ok": False, "error": "timeout or no more urls", "last_response": last_resp}


# ------------------------------
# Receive webhook from TDS
# ------------------------------
@app.post("/receive_requests")
async def receive_requests(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if (
        not isinstance(body, dict)
        or "email" not in body
        or "secret" not in body
        or "url" not in body
    ):
        raise HTTPException(status_code=400, detail="Missing required fields: email, secret, url")

    if body.get("secret") != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret")

    email = body["email"]
    url = body["url"]

    asyncio.create_task(background_task_wrapper(url, email, body.get("secret")))

    return {"message": "Request accepted. Solver started."}


async def background_task_wrapper(url: str, email: str, secret: str):
    try:
        res = await background_solve_chain(url, email, secret)
        print("[background_task] finished:", res)
    except Exception as e:
        print("[background_task] crashed:", repr(e))
