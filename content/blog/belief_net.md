---
title: Agentic Memory as a Belief-Revision System
tldr: For agents to make autonomous decisions over a scientific / complex task (e.g. autoresearch), we must store the AI system's *beliefs* over facts. The key question is how to induce or update beliefs as new evidence arrives.
author: Jiatong Yu
date: Sept 2026
description: Agentic memory as a belief-revision system
---

> This blog is motivated by my experience building AI automations for a hedge fund. It is an incredibly exciting field, and AI agents as-of today can succeed in most isolated tasks already. The gap is to build a system that can reuse reasoning trajectories for incremental updates.

## Overview

Harnesses like Claude Code or Codex are centered around *sessions*. In their use cases, sessions carry very different tasks (e.g. "fix this bug", "write me a frontend", "pull from database and compile a report"). Repeated components (e.g. code review, data health check) can be shipped as skills or MCP tools to increase efficiency. Knowledge about the working environment (e.g. the repo, the Snowflake configs) can be stored as long-term memory so that all sessions can learn from it. This setup is great for *individual developers* that invoke AI agents through individual sessions. In other words, Claude Code and Codex are *general-user-centered*.

Memory management becomes more tricky when we build more specialized harnesses, e.g. AI assistants built for a large enterprise. The harness needs to organize a large amount of domain knowledge and retrieve relevant information into agentic context. This requirement necessiates either knowledge graphs or carefully managed "wiki" file systems. These applications are still centered around sessions and users, but more specialized to domain knowledge.

At the end of the spectrum is full AI automation over complex, scientific tasks, such as autoresearch (e.g. [Mirendil](https://mirendil.com/) and [Loop Discovery](https://www.discoveryloop.com/)) and AI hedge funds. In these applications, harnesses should no longer center around sessions. Instead, the system should be built around ingesting new evidence and making decisions; sessions are only spawned to serve those goals. Several properties of this task:

1. **New evidence arrives sponteneously and incrementally**. Information (papers, news, call notes, etc.) arrives randomly, and the system must react to it immediately. The information is almost always *incremental* to what the system already knows, and we need to *update* instead of rebuild the system.
2. **Decision making relies on beliefs**. It is not enough to only hold facts in memory. In autoresearch, if the long term memory only contains empirical facts (run configs, logs, etc.), it's very hard for AI agents to find what's the next best frontier to explore. Instead, we want to store prior beliefs (such as "*optimizer X would work well in setting Y*", "*ensembling can help catastrophic loss spikes*") that are continuously induced or modified by new pieces of evidence, and feed them to AI agents when decision making is needed.

These properties combined call for a **[belief-revision system](https://en.wikipedia.org/wiki/Belief_revision)** (BRS) for agentic memory management, where analysis, conclusions, and reasoning traces are treated as first-class citizen. The reason is simple: as long as context is finite, we can not spam all facts into one session. This necessiates a system to organize the agents' beliefs over facts, so that incremental evidence doesn't trigger a full rebuild from scratch. As new evidence arrives, the system needs to track which beliefs need to be revised / removed and restore internal consistency after revisions. 

### Prior Work

Belief-revision as an academic field can be traced back to the 1980s. The AGM framework models a belief state as a logically closed set and characterizes three change operations: *expansion, contraction, and revision*. Expansion refers to adding a new belief to the system. Contraction is when you want to remvoe a prior belief without replacing it with a new one. To maintain internal consistency, one must locates all other beliefs that depend on this contracted belief and remove them accordingly. In symbolic logic, revision can be viewed as contraction followed by expansion, although this is not useful in practice. 

Applying BRS in agentic memory is gaining some initial attention. [T³](https://arxiv.org/pdf/2510.12264) tracks belief deviation and truncates belief-trapped training trajectories, yielding gains of up to 30 points while cutting token cost by up to 34%. [BeliefMem](https://arxiv.org/abs/2605.05583) maintains conclusions under partial observability with contradiction-triggered down-weighting and versioned history; the [Belief Engine](https://arxiv.org/abs/2605.15343) treats belief as an evidential state over a proposition and updates it with a log-odds rule controlled by an evidence-uptake and a prior-anchoring parameter.

> Arguably, agentic memory management is not / cannot be a hot research field. I think the reason is that *this is inherently a system design problem that should be developed case-by-case* depending on the concrete task. Attempts to build one-for-all architectures are likely not going to work. This [Reddit post](https://www.reddit.com/r/AI_Agents/comments/1ts3nq2/i_spent_a_year_building_agent_memory_on_knowledge/) argues how solving these tasks should be driven by bottom-up iterations instead of top-down design, which I resonate with a lot

### Our task: Parameter Golf Challenge

In this blog, I will document my attempts and lessons-learned building a belief-revision memory system for *autoresearch*, using OpenAI's [Parameter Golf](https://github.com/openai/parameter-golf) challenge. I chose this task because it carries two desired properties:

- The task has a well-defined evaluation criteria, namely training loss achieved under clear compute budgets and fixed dataset. We can evaluate design choices against the final experiments designed by the AI agents.
- Plenty of external evidence. We will use participants' submission records to simulate new "evidence" arriving to the system to stimulate belief generation and revision. 

**How the system evolves**: I built a BRS harness that iterates between ingesting external submission records and designing its own experiments to run. After ingesting all submissions and getting empirical results from 10 internal experiments, we will prompt an agent to design its final verdict. 

**Baseline**: We will compare the proposed harness to a wiki-based memory design. The baseline converts submission records into markdown files. An agent is prompted to read the wiki, design experiments, add results to the wiki, and repeat for 10 iterations before we ask for the final verdict. 


:::note Update
This project is WIP. To be continued.
:::