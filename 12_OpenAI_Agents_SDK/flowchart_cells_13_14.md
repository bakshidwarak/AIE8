# Flowchart for Cells 13 and 14

## Visual Flowchart Diagram

```mermaid
flowchart TD
    %% Cell 13: Question about Structured Responses
    A[Question #1:<br/>Why structured response templates?] --> B[Answer:<br/>✅ Easy Parsing<br/>✅ Reliable outcomes<br/>✅ Predictable error handling]
    
    %% Cell 14: Search Agent Creation
    C[Task 2: Create Search Agent] --> D[SEARCH_PROMPT:<br/>Research assistant instructions<br/>2-3 paragraphs, <300 words]
    
    D --> E[WebSearchTool:<br/>• OpenAI Responses API<br/>• Web search capability<br/>• Hosted tool]
    
    E --> F[Search Agent Creation:<br/>Agent with WebSearchTool<br/>tool_choice='required']
    
    F --> G[Key Features:<br/>• Required tool usage<br/>• Concise summaries<br/>• Report synthesis ready]
    
    %% Integration Flow
    H[User Query] --> I[Planner Agent<br/>Cell 12]
    I --> J[Search Agent<br/>Cell 14]
    J --> K[Writer Agent<br/>Cell 26]
    
    %% Connections showing relationship
    B -.-> F
    B -.-> I
    B -.-> K
    
    %% Styling
    classDef questionBox fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef answerBox fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef agentBox fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef toolBox fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef flowBox fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class A questionBox
    class B answerBox
    class C,D,F,G agentBox
    class E toolBox
    class H,I,J,K flowBox
```

## Cell 13: Question about Structured Responses

**Question #1:** "Why is it important to provide a structured response template? (Why are structured outputs helpful/preferred in Agentic workflows?)"

**Answer:**
- ✅ Easy Parsing (avoids random parsing)
- ✅ Better reliable outcomes
- ✅ Ease for error handling and predictable output template

## Cell 14: Search Agent Creation Flow

**Task 2: Create Search Agent**

1. **SEARCH_PROMPT**: Instructions for research assistant to create concise summaries
2. **WebSearchTool**: OpenAI's hosted tool for web searches
3. **Search Agent Creation**: Agent configured with required tool usage
4. **Key Features**: Ensures reliable tool usage and structured output

## Integration Flow: How Cells 13 & 14 Connect

The flowchart shows how Cell 13's structured output principles are applied throughout the agent workflow:

1. **Cell 13** establishes WHY structured outputs are important
2. **Cell 14** implements HOW to use structured outputs with tools
3. **Integration**: All agents (Planner, Search, Writer) benefit from structured outputs
4. **Flow**: User Query → Planner → Search Agent → Writer Agent

## Key Relationships

- **Cell 13** explains the theoretical benefits of structured outputs
- **Cell 14** demonstrates practical implementation with `tool_choice="required"`
- **Integration**: The Search Agent uses structured outputs to ensure reliable web search and summary generation
- **Workflow**: All agents in the chain benefit from structured, predictable outputs
