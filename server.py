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

DOC_NAMES = "all_jobs"  # or dynamically discovered
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
        user_profile_text=user_profile_text,
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

        for doc in item.file_names:
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
        <style>
          /* Basic reset */
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
          }
          
          .chat-container {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            width: 90%;
            max-width: 600px;
            padding: 20px;
            text-align: center;
          }

          h1 {
            margin-bottom: 10px;
          }

          .query-type-section {
            margin: 20px 0;
          }

          label {
            display: inline-block;
            margin: 5px;
            text-align: left;
          }

          input[type="text"],
          textarea,
          input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
            margin-bottom: 15px;
            border-radius: 4px;
            border: 1px solid #ccc;
          }

          button {
            background: #007bff;
            color: #ffffff;
            border: none;
            padding: 10px 20px;
            margin-top: 10px;
            border-radius: 4px;
            cursor: pointer;
          }

          button:hover {
            background: #0056b3;
          }

          .result-section {
            margin-top: 20px;
            text-align: left;
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            min-height: 100px;
            **overflow-x: auto;       /* ADDED: horizontal scrolling if needed */
            white-space: pre-wrap;   /* ADDED: wrap long lines */
            word-wrap: break-word;   /* ADDED: break long words */
            max-height: 400px;       /* OPTIONAL: limit overall height */
            overflow-y: auto;        /* OPTIONAL: vertical scrolling if needed */**
          }

          #thinking-section {
            margin-top: 20px;
            font-style: italic;
            color: #555;
          }

          .guiding-questions {
            margin: 10px 0;
            text-align: left;
          }

          .guiding-questions h3 {
            margin-bottom: 5px;
          }

          .guiding-questions ul {
            list-style-type: disc;
            padding-left: 20px;
          }

          .guiding-questions li {
            color: #007bff;
            cursor: pointer;
            margin: 5px 0;
          }
          
          .guiding-questions li:hover {
            text-decoration: underline;
          }
        </style>
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

          // Array of "thinking" messages
          const thinkingMessages = [
            "Thinking...",
            "Parsing documents...",
            "Finding the best jobs for you...",
            "Filtering results...",
            "Almost there..."
          ];
          let thinkingIndex = 0;
          let thinkingInterval;

          // Populate the question input when a guiding question is clicked
          function useGuidingQuestion(text) {
            // switch to General Question tab
            document.querySelector('input[name="query_type"][value="general"]').checked = true;
            onQueryTypeChange();
            document.getElementById("question").value = text;
          }
          
          async function submitForm() {
            // Display the "thinking" section
            document.getElementById("thinking-section").style.display = "block";
            thinkingIndex = 0;
            document.getElementById("thinking-text").innerText = thinkingMessages[thinkingIndex];
            
            // Rotate through the thinking messages every 2 seconds
            thinkingInterval = setInterval(() => {
              thinkingIndex = (thinkingIndex + 1) % thinkingMessages.length;
              document.getElementById("thinking-text").innerText = thinkingMessages[thinkingIndex];
            }, 2000);

            const queryType = document.querySelector('input[name="query_type"]:checked').value;
            let resultText = "";
            
            try {
              if (queryType === "matches") {
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
            } catch (e) {
              resultText = "Error fetching data. " + e;
            } finally {
              // Stop the "thinking" messages
              clearInterval(thinkingInterval);
              document.getElementById("thinking-section").style.display = "none";
            }
            
            // Display the result
            document.getElementById("result").innerText = resultText;
          }
        </script>
      </head>
      <body>
        <div class="chat-container">
          <h1>Palantir Jobs Chatbot</h1>
          
          <!-- Guiding Questions -->
          <div class="guiding-questions">
            <h3>Not sure what to ask? Try one of these:</h3>
            <ul>
              <li onclick="useGuidingQuestion('Base Salary for a Data Scientist Position?')">
                Base Salary for a Data Scientist Position?
              </li>
              <li onclick="useGuidingQuestion('What are the software engineering roles available right now?')">
                What are the software engineering roles available?
              </li>
              <li onclick="useGuidingQuestion('Which are the different jobs available in Palo Alto?')">
                Which are the different jobs available in Palo Alto?
              </li>
              <li onclick="useGuidingQuestion('Are there hybrid job opportunities available?')">
                Are there hybrid job opportunities available?
              </li>
            </ul>
          </div>
          
          <div class="query-type-section">
            <p>Select a query type:</p>
            <label>
              <input type="radio" name="query_type" value="matches" onchange="onQueryTypeChange()" checked>
              Relevant Job Matches
            </label>
            <label>
              <input type="radio" name="query_type" value="general" onchange="onQueryTypeChange()">
              General Job Questions
            </label>
          </div>
          
          <!-- Job Matches Fields (User Profile Info) -->
          <div id="job-matches-fields" style="display: block;">
            <h3>Job Matches Query</h3>
            <label>Skills:
              <input id="skills" type="text" placeholder="e.g., Python, SQL, Machine Learning">
            </label>
            <label>Experience:
              <textarea id="experience" rows="3" placeholder="Briefly describe your experience"></textarea>
            </label>
            <label>Location Preference:
              <input id="location_pref" type="text" placeholder="e.g., New York, Remote">
            </label>
            <label>Per Table Limit:
              <input id="per_table_limit" type="number" value="3">
            </label>
            <label>Global Top K:
              <input id="global_top_k" type="number" value="5">
            </label>
          </div>
          
          <!-- General Questions Fields -->
          <div id="general-question-fields" style="display: none;">
            <h3>General Job Question</h3>
            <label>Question:
              <input id="question" type="text" placeholder="Enter your question here">
            </label>
            <!-- Voice input button -->
            <button onclick="startVoiceInput()">🎤 Start Voice Input</button>
          </div>
          
          <button onclick="submitForm()">Submit Query</button>
          
          <!-- "Thinking" Section -->
          <div id="thinking-section" style="display:none;">
            <p id="thinking-text"></p>
          </div>
          
          <!-- Result Section -->
          <div class="result-section">
            <h2>Result:</h2>
            <pre id="result"></pre>
          </div>
        </div>
      </body>
    </html>
    """
    return html_content


