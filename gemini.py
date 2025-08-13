import os
import json
import google.generativeai as genai

# Get the API key from environment variable
api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

# It's good practice to use the most capable model available for complex tasks.
# Using "gemini-1.5-flash" as it's a strong, general-purpose model.
MODEL_NAME = "gemini-1.5-flash" 

SYSTEM_PROMPT = """
You are a data extraction and analysis assistant. 
Your job is to:
1. Write Python code that scrapes the relevant data needed to answer the user's query. If no URLs are given, then see the "uploads" folder and read the files provided there and give relevant metadata.
2. List all Python libraries that need to be installed for your code to run.
3. Identify and output the main questions that the user is asking, so they can be answered after the data is scraped.

You must respond **only** in valid JSON following the given schema:
{
  "code": "string — Python scraping code as plain text",
  "libraries": ["string — names of required libraries"],
  "questions": ["string — extracted questions"]
}
Do not include explanations, comments, or extra text outside the JSON.
"""

async def parse_question_with_llm(question_text, uploaded_files=None, urls=None, folder="uploads"):
    """
    Generates Python code to scrape or load data based on a user's question.
    """
    uploaded_files = uploaded_files or []
    urls = urls or []

    user_prompt = f"""
Question:
"{question_text}"

Uploaded files:
"{uploaded_files}"

URLs:
"{urls}"

You are a data extraction specialist.
Your task is to generate Python 3 code that loads, scrapes, or reads the data needed to answer the user's question.

1(a). Always store the final dataset in a file as `{folder}/data.csv`. If you need to store other files, also save them in this folder. Lastly, add the path and a brief description of the file in `{folder}/metadata.txt`.
1(b). Create code to collect metadata about the data you collected (e.g., storing details of the DataFrame using df.info(), df.columns, df.head(), etc.) in a `{folder}/metadata.txt` file. This metadata will help another model generate the final analysis code. Ensure the code creates any folder that doesn't exist, like `{folder}`.

2. Do not perform any analysis or answer the question in this step. Only write code to collect data and metadata.

3. The code must be self-contained and runnable without manual edits.

4. Use only Python standard libraries plus pandas, numpy, beautifulsoup4, and requests unless otherwise necessary.

5. If the data source is a webpage, download and parse it. If it’s a CSV/Excel, read it directly.

6. Do not explain the code.

7. Output only valid Python code.

8. Just scrape the data; don’t do anything fancy.

Return a JSON with:
1. The 'code' field — Python code that answers the question.
2. The 'libraries' field — list of required pip install packages.
3. Don't add libraries that come pre-installed with Python, like 'io'.
4. Your output will be executed inside a Python REPL.
5. Don't add comments.

Only return JSON in this format:
{{
  "code": "<...>",
  "libraries": ["pandas", "matplotlib"],
  "questions": ["..."]
}}

Lastly, I am saying again: do not try to solve these questions in this step.
If a JSON answer format is present in the user's question, also add it to the metadata.
"""

    model = genai.GenerativeModel(MODEL_NAME)

    response = await model.generate_content_async(
        [SYSTEM_PROMPT, user_prompt],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )
    
    return json.loads(response.text)

SYSTEM_PROMPT2 = """
You are a data analysis assistant.  
Your job is to:
1. Write Python code to solve the user's questions using the provided metadata.
2. List all Python libraries that need to be installed for the code to run.
3. Add code to save the final result to "{folder}/result.json".

Do not include explanations, comments, or extra text outside the JSON.
"""

async def answer_with_data(question_text, folder="uploads"):
    """
    Generates Python code to analyze data and answer questions, using pre-generated metadata.
    """
    metadata_path = os.path.join(folder, "metadata.txt")
    try:
        with open(metadata_path, "r") as file:
            metadata = file.read()
    except FileNotFoundError:
        metadata = "No metadata file found. Please read data from data.csv."


    user_prompt = f"""
Question:
{question_text}

Metadata:
{metadata}

Return a JSON with:
1. The 'code' field — Python code that answers the question by reading `{folder}/data.csv`.
2. The 'libraries' field — list of required pip install packages.
3. Don't add libraries that come pre-installed with Python, like 'io'.
4. Your output will be executed inside a Python REPL.
5. Don't add comments.
6. Convert any image/visualization if present into a base64 encoded PNG and add it to the final JSON result.
7. **CRITICAL**: Before saving the final dictionary to `{folder}/result.json`, you MUST convert all pandas or numpy numeric types (like int64, float64) to native Python types using `int()` or `float()`. The `json` library cannot serialize numpy types and will cause a `TypeError`.

You must respond **only** in valid JSON with these properties:
{{
  "code": "string — Python analysis code as plain text",
  "libraries": ["string — names of required libraries"]
}}

Lastly, follow the answer format provided in the questions and save the final answers in the `{folder}/result.json` file.
"""

    model = genai.GenerativeModel(MODEL_NAME)

    # SYSTEM_PROMPT2 needs to be formatted with the folder
    system_prompt2_formatted = SYSTEM_PROMPT2.format(folder=folder)

    response = await model.generate_content_async(
        [system_prompt2_formatted, user_prompt],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)
