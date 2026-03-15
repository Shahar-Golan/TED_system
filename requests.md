# Requirements

## 1) Build the agent in an optimized way

Your implementation should be efficient:

- Avoid unnecessary LLM calls.
- Minimize prompt/context size (only what's needed).
- Stay within the project budget.

---

## 2) API Endpoints (Required)

Your system must expose the following HTTP endpoints (names must match exactly):

### A) GET /api/team_info

Returns student details.

- **Purpose**: retrieve student names and emails.

**Response format (JSON):**
```json
{
  # I filled here the new value
  "group_batch_order_number": "3_12", // from presentation list
  "team_name": "שחר+תומר+אייל",
  "students": [
    
    { "name": "שחר גולן", "email": "shahar.golan@campus.technion.ac.il" },
    { "name": "תומר פרץ", "email": "tomer.perez@campus.technion.ac.il" },
    { "name": "אייל קוטליק", "email": "eyal.kotlik@campus.technion.ac.il" }
  ]
}
```

### B) GET /api/agent_info

Returns agent meta + how to use it.

**Must include:**
- description
- purpose
- prompt templates (suggested way to work with the agent)
- prompt examples and full responses

**Response format (JSON):**
```json
{
  "description": "…",
  "purpose": "…", // what this agent purpose
  "prompt_template": {
    "template": "…"
  },
  "prompt_examples": [
    {
      "prompt": "What does Hilary Clinton saying about nuclear weapon?",
      "full_response": "Full response your agent returns…",
      "steps": [full list of steps, see below]
    }
  ]
}
# another example: What does Jow Biden opinion about immigration
```

### C) GET /api/model_architecture

Returns the architecture diagram as an image (PNG).

- **Purpose**: retrieve a PNG image of the model architecture.
- The architecture must be clear.
- All sub-modules / sub-agents names must be consistent across:
  - the architecture diagram
  - your steps logging (see /api/execute)
  - any descriptions you provide

**Response:**
- Content-Type: image/png
- Body: the PNG file

### D) POST /api/execute

This is the main entry point.

- User sends an input prompt.
- Your API returns the agent response + the full traced steps.

**Input format (JSON):**
```json
{
  "prompt": "User request here"
}
```

**Response format (JSON)** — must match exactly these top-level fields:
```json
{
  "status": "ok",
  "error": null,
  "response": "…",
  "steps": []
}
```

**If error:**
```json
{
  "status": "error",
  "error": "Human-readable error description",
  "response": null,
  "steps": []
}
```

**Steps:**

`steps` is an array describing every LLM call the agent did in order.

You must include:
- module/submodule name (must correlate to architecture diagram)
- prompt
- response

**Required step object schema:**
```json
{
  "module": "…", // the module name according to your architecture
  "prompt": {},
  "response": {}
}
```

---

# we have done this part.
## 3) Frontend/GUI (Required) 

You must provide a minimal web UI to operate your agent.

### GUI Requirements

- A text input (textarea) for entering a prompt/task.
- A "Run Agent" button that calls POST /api/execute.
- Display the final agent response (response).
- Display the full steps trace (steps), including:
  - module
  - prompt
  - response

### Optional (Only if supported by your agent)

- Support back-and-forth interaction (follow-up prompts).
- Display conversation history in the UI.

The UI should be simple and focused on interacting with the agent and inspecting its execution.
