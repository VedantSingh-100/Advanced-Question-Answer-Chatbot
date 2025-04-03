# server.py

import os
import evadb
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse
from typing import List, Optional

from vector_store import table_exists  # or define a helper
from retrieval import vector_retrieval, summary_retrieval
from subquestion_generator import generate_subquestions
from aggregator import response_aggregator
from job_seeking import aggregate_job_matches
import time

# We'll have a global cursor for reuse
cursor = None

app = FastAPI(
    title="Palantir Jobs Backend",
    description="A robust backend for Palantir jobs Q&A using EvaDB + LLM",
    version="0.1.0"
)

DOC_NAMES = [f"PALANTIR_JOBS_{i}" for i in range(1, 85)]  # or dynamically discovered
LLM_MODEL = "gpt-3.5-turbo"

import time
import logging

logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
def startup_event():
    global cursor
    db_path = "/home/vhsingh/rag-demystified-main/evadb_data"
    start_time = time.perf_counter()
    
    logging.info("Connecting to EvaDB...")
    connection = evadb.connect(db_path)
    logging.info(f"Done connecting to EvaDB in {time.perf_counter() - start_time:.2f} seconds.")

    start_time = time.perf_counter()
    logging.info("Getting cursor...")
    cursor = connection.cursor()
    logging.info(f"Got cursor in {time.perf_counter() - start_time:.2f} seconds.")

    # Next step, checking table existence
    start_time = time.perf_counter()
    logging.info("Checking if table PALANTIR_JOBS_1 exists...")
    if not table_exists(cursor, "PALANTIR_JOBS_1"):
        logging.warning("Table not found. Did you run offline setup?")
    else:
        logging.info("PALANTIR_JOBS_1 found.")
    logging.info(f"Check took {time.perf_counter() - start_time:.2f} seconds.")


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "OK"}

@app.post("/job_matches")
def get_job_matches(
    user_profile_text: str = Body(..., example="I am a recent CS graduate interested in software engineering."),
    per_table_limit: int = Body(3, example=3),
    global_top_k: int = Body(5, example=5)
):
    """
    Returns job matches based on the user's profile text.
    """
    if cursor is None:
        raise HTTPException(status_code=500, detail="Cursor not initialized. Check server startup logs.")
    
    matches = aggregate_job_matches(
        cursor=cursor,
        doc_names=DOC_NAMES,
        user_profile_text=user_profile_text,
        per_table_limit=per_table_limit,
        global_top_k=global_top_k
    )
    if not matches:
        return "No matches found."
    return matches

@app.post("/ask_question")
def ask_question(
    question: str = Body(...),
    doc_name: Optional[str] = Body(None),
    k: int = Body(3)
):
    if cursor is None:
        return {"error": "Cursor not initialized. Check server startup logs."}

    user_task = """We have a database of job postings from Palantir.
                   We are building an application to answer questions about these jobs.
                   The documents are each representing a single job with fields like job title, location, etc."""
    subquestions_list, cost_gs = generate_subquestions(
        question=question, 
        file_names=DOC_NAMES, 
        user_task=user_task,
        llm_model=LLM_MODEL
    )

    question_cost = cost_gs
    responses = []

    # Iterate over each subquestion bundle using dot notation:
    for item in subquestions_list:
        subq = item.question  # use dot notation
        func = item.function   # use dot notation; if you need the string value, use item.function.value
        doc_list = item.file_names  # a list of enums; later you'll extract their .value

        if doc_name:
            doc_list = [doc_name]

        for doc in doc_list:
            selected_doc = doc.value
            if func == "vector_retrieval" or (hasattr(func, "value") and func.value == "vector_retrieval"):
                start_time = time.time()
                resp, retrieval_cost = vector_retrieval(cursor, LLM_MODEL, subq, selected_doc)
                question_cost += retrieval_cost
                responses.append(resp)
            elif func == "llm_retrieval" or (hasattr(func, "value") and func.value == "llm_retrieval"):
                # If you're storing doc text in memory or somewhere
                resp, sum_cost = summary_retrieval(LLM_MODEL, subq, "SOME_DOC_TEXT")
                question_cost += sum_cost
                responses.append(resp)
            else:
                responses.append("Unknown function call.")

    final_answer, agg_cost = response_aggregator(LLM_MODEL, question, responses)
    elapsed = time.time() - start_time
    print(f"The elapsed time is {elapsed}")
    question_cost += agg_cost

    return final_answer

# Inside server.py (keep the rest of your API endpoints as before)

# In server.py, within your HTML UI returned by GET /
@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <html>
      <head>
        <title>Palantir Jobs Chatbot</title>
        <script>
          // Check for Speech Recognition API support
          let recognition;
          if ('webkitSpeechRecognition' in window) {
              recognition = new webkitSpeechRecognition();
              recognition.continuous = false;
              recognition.interimResults = false;
              recognition.lang = 'en-US';
              recognition.onresult = function(event) {
                  let transcript = event.results[0][0].transcript;
                  // Place transcript into the "question" input field
                  document.getElementById('question').value = transcript;
              };
              recognition.onerror = function(event) {
                  console.error("Speech recognition error:", event.error);
              };
          } else {
              console.log("Speech Recognition API is not supported in this browser.");
          }

          function startVoiceInput() {
              if (recognition) {
                  recognition.start();
              } else {
                  alert("Voice input not supported in your browser.");
              }
          }

          function onQueryTypeChange() {
            const queryType = document.querySelector('input[name="query_type"]:checked').value;
            if (queryType === "matches") {
              document.getElementById("job-matches-fields").style.display = "block";
              document.getElementById("general-question-fields").style.display = "none";
            } else {
              document.getElementById("job-matches-fields").style.display = "none";
              document.getElementById("general-question-fields").style.display = "block";
            }
          }
          
          async function submitForm() {
            const queryType = document.querySelector('input[name="query_type"]:checked').value;
            let resultText = "";
            if (queryType === "matches") {
              // Get user profile info fields
              const skills = document.getElementById("skills").value;
              const experience = document.getElementById("experience").value;
              const locationPref = document.getElementById("location_pref").value;
              const userProfile = "Skills: " + skills + "\\nExperience: " + experience + "\\nLocation Preference: " + locationPref;
              const perTableLimit = document.getElementById("per_table_limit").value;
              const globalTopK = document.getElementById("global_top_k").value;
              
              const response = await fetch("/job_matches", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  user_profile_text: userProfile,
                  per_table_limit: parseInt(perTableLimit),
                  global_top_k: parseInt(globalTopK)
                })
              });
              const data = await response.json();
              resultText = JSON.stringify(data, null, 2);
            } else {
              const question = document.getElementById("question").value;
              const response = await fetch("/ask_question", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question, k: 3 })
              });
              const data = await response.json();
              resultText = JSON.stringify(data, null, 2);
            }
            document.getElementById("result").innerText = resultText;
          }
        </script>
      </head>
      <body onload="onQueryTypeChange()">
        <h1>Palantir Jobs Chatbot</h1>
        <p>Select a query type:</p>
        <label>
          <input type="radio" name="query_type" value="matches" onchange="onQueryTypeChange()" checked>
          Relevant Job Matches
        </label>
        <label>
          <input type="radio" name="query_type" value="general" onchange="onQueryTypeChange()">
          General Job Questions
        </label>
        
        <!-- Job Matches Fields (User Profile Info) -->
        <div id="job-matches-fields" style="display: block; margin-top:20px;">
          <h3>Job Matches Query</h3>
          <label>Skills: <br>
            <input id="skills" type="text" size="50" placeholder="e.g., Python, SQL, Machine Learning">
          </label><br><br>
          <label>Experience: <br>
            <textarea id="experience" rows="3" cols="50" placeholder="Briefly describe your experience"></textarea>
          </label><br><br>
          <label>Location Preference: <br>
            <input id="location_pref" type="text" size="50" placeholder="e.g., New York, Remote">
          </label><br><br>
          <label>Per Table Limit: <input id="per_table_limit" type="number" value="3"></label><br>
          <label>Global Top K: <input id="global_top_k" type="number" value="5"></label><br>
        </div>
        
        <!-- General Questions Fields -->
        <div id="general-question-fields" style="display: none; margin-top:20px;">
          <h3>General Job Question</h3>
          <label>Question: 
            <input id="question" type="text" size="50" placeholder="Enter your question here">
          </label>
          <!-- Voice input button -->
          <button onclick="startVoiceInput()">🎤 Start Voice Input</button>
        </div>
        
        <br>
        <button onclick="submitForm()">Submit Query</button>
        
        <h2>Result:</h2>
        <pre id="result"></pre>
      </body>
    </html>
    """
    return html_content

