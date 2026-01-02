# Curio Chat Assistant

A full-stack example application demonstrating the **Curio Agent SDK** with a React frontend and Flask backend. This chat assistant can perform math calculations, unit conversions, and engage in natural conversation while maintaining history.

## Features

- **Chat Interface**: Clean, modern React UI for conversations
- **Math Tools**: Basic arithmetic, advanced math (sqrt, pow, trig), percentages
- **Unit Conversions**: Length, weight, temperature, volume
- **Conversation History**: SQLite persistence with conversation management
- **Tiered Models**: Uses Groq models with tier-based routing
- **Agent Transparency**: Shows which tools were used for each response

## Architecture

```
chat_assistant/
├── backend/
│   ├── app.py          # Flask API server
│   ├── agent.py        # ChatAssistantAgent using Curio SDK
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Chat.jsx
│   │   │   └── Sidebar.jsx
│   │   └── styles/
│   │       └── App.css
│   ├── package.json
│   └── vite.config.js
├── .env.example
└── README.md
```

## Prerequisites

- **Python 3.9+**
- **Node.js 18+** and npm
- **Groq API Key** - Get one at [console.groq.com/keys](https://console.groq.com/keys)

## Quick Start

### 1. Clone and Navigate

```bash
cd curio_agent_sdk/examples/chat_assistant
```

### 2. Set Up Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=gsk_your_key_here
```

### 3. Install Backend Dependencies

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Also install the Curio SDK (from parent directory)
pip install -e ../../../
```

### 4. Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # If using virtualenv
python app.py
```

The API will start at `http://localhost:5004`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

The UI will be available at `http://localhost:3000`

## Usage Examples

Try these prompts to test the agent:

### Math Operations
- "What's 15% of 250?"
- "Calculate (45 * 23) + (100 / 4)"
- "What's the square root of 144?"
- "What's 2 to the power of 10?"

### Unit Conversions
- "Convert 100 kilometers to miles"
- "What's 98.6 Fahrenheit in Celsius?"
- "Convert 5 gallons to liters"

### Percentage Calculations
- "30 is what percent of 150?"
- "What's the percentage change from 80 to 120?"

### General Chat
- "What can you help me with?"
- "Explain how compound interest works"

## Configuration

### Model Tiers

The `.env` file configures model tiers for different agent phases:

| Tier | Use Case | Default Model |
|------|----------|---------------|
| tier1 | Fast responses, synthesis | llama-3.1-8b-instant |
| tier2 | Planning, tool execution | llama-3.3-70b-versatile |

### Agent Configuration

Edit `backend/agent.py` to customize:

```python
class ChatAssistantAgent(BaseAgent):
    def __init__(self, agent_id, config, conversation_history):
        super().__init__(
            agent_id=agent_id,
            config=config,
            plan_tier="tier2",      # Model tier for planning
            critique_tier="tier1",  # Model tier for critique
            synthesis_tier="tier1", # Model tier for final response
            action_tier="tier1",    # Model tier for tool execution
        )
        self.max_iterations = 3     # Max planning loops
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations` | Create new conversation |
| GET | `/api/conversations/:id` | Get conversation with messages |
| DELETE | `/api/conversations/:id` | Delete conversation |
| POST | `/api/chat` | Send message and get response |

### Chat Request

```json
POST /api/chat
{
  "conversation_id": "optional-existing-id",
  "message": "What's 15% of 200?"
}
```

### Chat Response

```json
{
  "conversation_id": "abc123",
  "response": "15% of 200 is 30. I calculated this using the percentage tool...",
  "metadata": {
    "run_id": "run-xyz",
    "iterations": 1,
    "status": "done",
    "tools_used": [
      {"action": "percentage", "result_status": "ok"}
    ]
  }
}
```

## Adding Custom Tools

To add new tools to the agent, edit `backend/agent.py`:

```python
def initialize_tools(self) -> None:
    # Existing tools...
    self.register_tool("my_tool", self.my_tool_method)

def my_tool_method(self, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    name: my_tool
    description: Description of what this tool does
    parameters:
        param1: Description of parameter 1
        param2: Description of parameter 2
    required_parameters:
        - param1
    """
    # Implementation
    result = do_something(args.get("param1"))
    return {"status": "ok", "result": result}
```

## Troubleshooting

### "No API key found"
Make sure your `.env` file exists and contains a valid `GROQ_API_KEY`.

### "Module not found: curio_agent_sdk"
Install the SDK: `pip install -e ../../../` from the backend directory.

### Frontend can't connect to backend
- Ensure backend is running on port 5004
- Check that the Vite proxy is configured in `vite.config.js`

### Slow responses
- Try using smaller models in tier1
- Reduce `max_iterations` in the agent

## Development

### Backend Development
```bash
# Run with auto-reload
FLASK_DEBUG=true python app.py
```

### Frontend Development
```bash
# Vite dev server with hot reload
npm run dev
```

### Build for Production
```bash
# Build frontend
cd frontend
npm run build

# Serve with Flask (configure to serve static files)
```

## License

This example is part of the Curio Agent SDK. See the main repository for license information.
