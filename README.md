# 🧠 Self-Healing DevTools Agent

This repository contains the code and demo for submission to the **Composio x DevClub IITD x Chirality Labs – Agents in Production Hackathon**.

It demonstrates how an AI agent can **automatically detect, fix, and test code errors** using **Composio Tool Router**, **Telemetry**, and **Cloud CLI**, creating a reliable and production-ready workflow.

---

## 🚩 Problem Statement

AI agents in production often fail when:

- Tools break or API keys expire.
- Code errors occur during runtime.
- Developers spend hours manually fixing these issues.

The challenge is to build an **autonomous self-healing agent** that can:

- Detect and understand runtime errors automatically.
- Suggest or apply fixes on its own.
- Collect and visualize telemetry data from multiple tools.
- Use Composio Tool Router to connect and execute external apps.
- Evaluate its performance using reproducible scorecards.

---

## 💡 Solution Overview

The **Self-Healing DevTools Agent** combines multiple features into one complete system:

### 1. 🧩 Execute & Self-Heal
- Paste any Python function and provide JSON arguments.
- The agent runs the code, detects errors, and applies fixes automatically.

### 2. 📈 Telemetry
- Logs every event and ToolRouter call.
- Displays success/failure rates, latency, and context.
- Helps visualize how agents make decisions.

### 3. ⚙️ ToolRouter Dashboard
- Shows all Composio ToolRouter API calls.
- Includes filters for app, action, and status.
- Useful for debugging and tracing tool usage.

### 4. 🧮 Evaluator
- Runs **preflight checks** and **seed-based evaluations**.
- Generates scorecards to test consistency and reliability.
- Measures how well the agent performs over time.

### 5. ☁️ Cloud CLI
- Manage cloud storage (S3/GCS/Azure) using ToolRouter.
- Supports `list`, `sync`, `presign`, and `version`.
- Includes safety checks for write actions.

### 6. 🧠 Autogen Template Loader
- Loads YAML templates to automatically create new agents.
- Example: agent that opens GitHub PRs and triggers CI pipelines.
- Enables dynamic creation of new workflows.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python 3.12 |
| UI Framework | Streamlit |
| AI Model | Llama 3.1 - 8B Instant (Groq API) |
| Tool Integration | Composio Tool Router |
| Evaluation | Custom Evaluator with Seed Control |
| Storage | AWS S3 (via ToolRouter) |
| CI/CD | CircleCI / GitHub Actions |
| Telemetry | JSONL Trace Export |

---

## 🔑 How to Get API Keys

### 1. Groq API Key
1. Go to [Groq Cloud](https://cloud.groq.com/).
2. Sign up / log in → API Keys → copy key.
3. Save in `.env` as `GROQ_API_KEY`.

### 2. Composio Tool Router
1. Go to [Composio Developer Portal](https://app.composio.dev/).
2. Sign up / log in → API Keys → copy key.
3. Save as `COMPOSIO_API_KEY` in `.env`.

### 3. GitHub Token
1. [GitHub Developer Settings](https://github.com/settings/tokens) → Generate new token.
2. Select scopes: `repo`, `workflow`, `write:packages`.
3. Save as `GITHUB_TOKEN` in `.env`.

### 4. CircleCI Token
1. [CircleCI Dashboard](https://app.circleci.com/) → User Settings → Personal API Tokens → Create Token.
2. Save as `CIRCLECI_TOKEN` in `.env`.

### 5. Jira Token (optional)
1. [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens) → Create token.
2. Save in `.env`:
   ```
   JIRA_TOKEN=your_token
   JIRA_BASE_URL=https://yourdomain.atlassian.net
   JIRA_PROJECT_KEY=DEV
   JIRA_EMAIL=you@company.com
   ```

### 6. Slack Bot Token
1. [Slack API Apps](https://api.slack.com/apps) → create app → add Bot Token Scopes: `chat:write`.
2. Install to workspace → copy **Bot User OAuth Token**.
3. Save as `SLACK_BOT_TOKEN` in `.env`.

### 7. AWS S3 Keys
1. [AWS IAM](https://console.aws.amazon.com/iam/) → Create user with **Programmatic access** + **S3 full access**.
2. Copy `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
3. Save in `.env` along with `AWS_DEFAULT_REGION` and `STORAGE_BUCKET`.

---

## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/somanathdiksangi/DevTools-Agent.git
cd DevTools-Agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🖥️ How to Use

### ▶️ Run Demo
- Go to the **Run Demo** tab.
- Paste any Python function, e.g.:

```python
def divide_two_numbers(a, b):
    return a / b
```
- Enter arguments like `[10, 0]`.
- Click **Run self-heal** to see the agent fix errors automatically.

### 📊 Telemetry
- View logs of all agent events and ToolRouter calls.

### 🧭 ToolRouter Dashboard
- See history of all ToolRouter actions.
- Filter by app or action name.

### 🧩 Evaluator
- Run `preflight` or seed-based evaluations (7, 13, 21).
- View reliability and accuracy of the agent.

### ☁️ Cloud CLI
- Choose an action (`list`, `sync`, `presign`, `version`).
- Provide parameters in JSON.
- Execute cloud actions safely through ToolRouter.

### 🧠 Autogen Template Loader
- Paste YAML templates to create new agents dynamically.
- Example YAML:

```yaml
agents:
  - name: fixer
    role: writes patches
    tools: [git_pr, ci_trigger]
tools:
  - name: git_pr
    app: github
    action: open_pr
  - name: ci_trigger
    app: circleci
    action: trigger_pipeline
plan: run fixer -> open PR -> trigger CI
```

---

## 📹 Demo Video

1. Show **Execute & Self-Heal** in action.
2. Display **Telemetry and ToolRouter Dashboard**.
3. Demonstrate **Evaluator and Cloud CLI**.
4. Use **Autogen Template Loader** to create a workflow.

**Demo Link:** https://drive.google.com/drive/folders/1piYEavlqCLHzximOcteUOqqLuYWgDL7J?usp=sharing

---

## 🧾 Friction Log

| Challenge                 | Description                 | Solution                                  |
| ------------------------- | --------------------------- | ----------------------------------------- |
| Authentication            | API keys missing or expired | Used `.env` and mock ToolRouter mode      |
| ToolRouter setup          | Limited examples available  | Referred to `docs.composio.dev`           |
| Evaluator reproducibility | Scores changed per run      | Fixed seeds via `.env` variable           |
| YAML loader parsing       | Some templates invalid      | Added input validation and error handling |

---

## 👥 Team

**Project Lead:** Somanath Diksasngi


---

## 🌱 Why It Matters

This project shows how **AI agents can repair themselves**, understand failures, and recover automatically.  
It reduces debugging time, improves system uptime, and makes autonomous workflows more reliable.

---

