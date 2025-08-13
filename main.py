from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging

# Assuming these are custom modules you have defined elsewhere
from task_engine import run_python_code
from gemini import parse_question_with_llm, answer_with_data

app = FastAPI()

# --- Middleware Setup ---
# Enables Cross-Origin Resource Sharing (CORS) to allow requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Helper Functions ---
def last_n_words(s, n=25):
    """Returns the last n words of a given string."""
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_csv_empty(csv_path):
    """Checks if a CSV file is empty or does not exist."""
    return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0


# --- Route Handlers ---

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main HTML frontend page on a GET request to the root URL."""
    try:
        with open("frontend.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: frontend.html not found</h1>", status_code=404)


@app.post("/")
async def analyze(request: Request):
    """
    Handles file uploads and processes the user's question via POST request.
    This is the core logic for the AI-powered analysis.
    """
    # Create a unique folder for this request to store logs and files
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)

    # Setup logging for this specific request
    log_path = os.path.join(request_folder, "app.log")
    logger = logging.getLogger(request_id)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Log to a file
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also log to the console for real-time monitoring
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Step 1: Request folder created: %s", request_folder)

    try:
        form = await request.form()
    except Exception as e:
        logger.error("Failed to parse form data: %s", e)
        return JSONResponse(status_code=400, content={"message": "Invalid form data."})

    question_text = None
    saved_files = {}

    # Save all uploaded files and extract the question
    for field_name, value in form.items():
        if hasattr(value, "filename") and value.filename:  # It's a file
            file_path = os.path.join(request_folder, value.filename)
            async with aiofiles.open(file_path, "wb") as f:
                content = await value.read()
                await f.write(content)
            saved_files[field_name] = file_path

            # Check for the specific question file
            if value.filename == "question.txt":
                question_text = content.decode('utf-8')
        else: # It's a text field
            saved_files[field_name] = value
            if field_name == 'question':
                 question_text = value


    if not question_text:
        logger.error("No question provided in the form.")
        return JSONResponse(status_code=400, content={"message": "Question is missing."})

    logger.info("Step 2: Files received and question extracted.")

    # --- LLM Interaction and Code Execution ---
    # This section contains complex logic with multiple retries to ensure robustness.
    
    # Step 3: Get initial code from LLM
    response = None
    max_attempts = 3
    for attempt in range(max_attempts):
        logger.info("Step 3: Attempt %d to get code from LLM.", attempt + 1)
        try:
            response = await parse_question_with_llm(
                question_text=question_text,
                uploaded_files=saved_files,
                folder=request_folder
            )
            if isinstance(response, dict) and "code" in response:
                logger.info("Step 3: Successfully received code from LLM.")
                break
        except Exception as e:
            error_snippet = last_n_words(str(e), 50)
            question_text += f"\n\nPrevious attempt failed with error: {error_snippet}"
            logger.error("Step 3: Error during LLM call: %s", e)
        response = None # Reset response if it wasn't valid

    if not response:
        logger.critical("Fatal: Could not get a valid response from LLM after %d attempts.", max_attempts)
        return JSONResponse(status_code=500, content={"message": "Failed to get valid plan from LLM."})

    # Step 4: Execute the generated code
    execution_result = await run_python_code(response["code"], response.get("libraries", []), folder=request_folder)
    logger.info("Step 4: Initial code execution finished.")

    # Retry logic if execution fails or produces an empty CSV
    count = 0
    while execution_result.get("code") != 1 and count < 3:
        logger.error("Step 4: Execution failed. Retrying... (Attempt %d)", count + 1)
        error_feedback = last_n_words(str(execution_result.get("output")), 50)
        new_question_text = f"{question_text}\n\nThe previous code failed with this error: {error_feedback}. Please provide a corrected version."
        
        # Get new code
        response = await parse_question_with_llm(
            question_text=new_question_text, uploaded_files=saved_files, folder=request_folder
        )
        if not (isinstance(response, dict) and "code" in response):
            logger.error("Step 4: Could not get corrected code from LLM.")
            count += 1
            continue
            
        # Execute new code
        execution_result = await run_python_code(response["code"], response.get("libraries", []), folder=request_folder)
        logger.info("Step 4: Retried code execution finished.")
        count += 1
    
    if execution_result.get("code") != 1:
        logger.critical("Fatal: Code execution failed after multiple retries.")
        return JSONResponse(status_code=500, content={"message": "Failed to execute the generated code."})

    # Step 5 & 6: Get final answer from LLM based on data
    gpt_ans = None
    response_questions = response.get("questions")
    for attempt in range(max_attempts):
        logger.info("Step 5: Attempt %d to get final answer logic from LLM.", attempt + 1)
        try:
            gpt_ans = await answer_with_data(response_questions, folder=request_folder)
            if isinstance(gpt_ans, dict) and "code" in gpt_ans:
                logger.info("Step 5: Successfully received final answer logic.")
                break
        except Exception as e:
            logger.error("Step 5: Error during final LLM call: %s", e)
            response_questions = str(response_questions) + f"\n\nPrevious attempt failed with error: {last_n_words(str(e), 50)}"
        gpt_ans = None

    if not gpt_ans:
        logger.critical("Fatal: Could not get final answer logic from LLM.")
        return JSONResponse(status_code=500, content={"message": "Failed to get final answer logic from LLM."})

    # Step 7: Execute the final code
    final_result = await run_python_code(gpt_ans["code"], gpt_ans.get("libraries", []), folder=request_folder)
    if final_result.get("code") != 1:
        logger.error("Step 7: Final code execution failed. %s", final_result.get("output"))
        return JSONResponse(status_code=500, content={"message": "Final code execution failed.", "details": final_result.get("output")})

    logger.info("Step 7: Final code executed successfully.")

    # Step 8: Return the final result
    result_path = os.path.join(request_folder, "result.json")
    try:
        async with aiofiles.open(result_path, "r") as f:
            data = json.loads(await f.read())
        logger.info("Step 8: Sending successful result back to client.")
        return JSONResponse(content=data)
    except FileNotFoundError:
        logger.error("Step 8: result.json was not created by the executed code.")
        return JSONResponse(status_code=500, content={"message": "Result file not found."})
    except json.JSONDecodeError:
        logger.error("Step 8: result.json contains invalid JSON.")
        return JSONResponse(status_code=500, content={"message": "Result file is not valid JSON."})

