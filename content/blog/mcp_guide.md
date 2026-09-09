---
title: Notes on building concurrency-friendly MCP tools
tldr: A system-level guide on how to design MCP servers and tools for very large scale agentic systems.
author: Jiatong Yu
date: September 2026
description: Notes for building concurrency-friendly MCP tools
---
# From Threads to Tools: Concurrency Notes for MCP Servers at Scale

Tooling is a central component in AI agent harnesses, and designing good MCP servers and tools is more delicate than it seems. When you have only a few running agents, function specifications are the only concern. When the workflow is long, you need to consider the return values of tools and whether they will pollute the context window. But as you scale from a few agents to dozens of agents (and each can spawn subagents or make parallel tool calls), a poorly designed tooling will lead to mysterious pipeline crashes, halts, or wasted tokens.

This blog carries my learning notes about tooling designs, along with some of my past lessons-learned.

## Primer

> Extremely basic stuff, feel free to skip.

A **process** is the operating system's unit of isolation. Every process gets its own virtual address space: a private, contiguous-looking range of memory addresses that the OS maps onto physical RAM. Two processes hold same virtual address and be referring to different physical memory. This isolation ensures that a crash in one process cannot corrupt another.

A **thread** is a unit of execution inside a process. All threads in a process share the same address space: the same heap, the same globals, the same loaded libraries. Sharing data between threads is free. But the lack of isolation means two threads can mutate the same structure and get into a data race.

Python **GIL** (Global Interpreter Lock) is a lock that only allows one thread to be in a state of execution at any point in time. Different python versions have slight nuances on when the lock is released. Numerical or scientific Python stack such as NumPy, SciPy, etc. are Python interfaces over C and Fortan kernals and GIL is released at their execution. So threads are useless for CPU work written in Python bytecode, but works well for CPU work that lives in GIL-releasing compiled kernels. This distinction becomes important later.

A **CPU-bound task** is limited by computation: a matrix decomposition, a backtest inner loop. Its natural resource is cores, and its natural failure mode is contention. An **I/O-bound task** is limited by waiting: an HTTP call to a market data API, a database query. While it waits, the CPU is idle; its natural resources are connections and rate limits, and its natural failure modes are exhausted pools and slow upstreams cascading into pile-ups.

Each CPU core has small, fast private caches (L1/L2) holding the working set of whatever it's computing; a cache hit is on the order of 100× faster than a fetch from main RAM, so dense numerical code gets its speed precisely from staying in cache. 

When there are more runnable threads than cores, the OS scheduler does **context switches**: every few milliseconds it suspends a thread and swaps in another. Incoming thread's data evicts the outgoing thread's working set, which must then be rebuilt from RAM when it get scheduled again.

### Shared memory

Because processes cannot see each other's memory, the default way to move data between them is copying: serialize the object to bytes, push the bytes through an OS pipe, deserialize on the other side. For small control messages this is fine. For a 2GB NumPy array it is ruinous.

Shared memory is the OS's solution to this: it maps the same physical pages into more than one process's virtual address space. 
Each process keeps its own page table and its own virtual addresses, but the entries point at identical physical RAM. There is no channel and no transfer; a write by one process is instantly visible to the other because only one copy exists.

One neighboring trick worth knowing: on Linux, `fork()` doesn't copy the parent's memory either. Children share all pages copy-on-write, so "load the dataset in the parent, then fork workers" gives implicit sharing of read-only data. The classic Python gotcha is that merely *reading* a Python object writes to it — reference counting updates a counter in the object header — dirtying the page and triggering the copy anyway. Big NumPy buffers survive this (the header is one page; the payload is never written). A dict of millions of small Python objects will slowly copy itself into every worker.

### Event Loops
Asyncio is Python's library to write concurrent code, and event loop is every asyncio application's core engine. Event loops run asynchronous tasks and callbacks, perform network IO operations, and run subprocesses. It runs on *one thread* and iterates between waiting for tasks and running code that handles a task. 

The event loop thread adopts I/O multiplexing, meaning that it can monitor the many file descriptors (network sockets or pipes) at the same time. Once ran, a task in an event loop is *never* interrupted. The event loop will wait for the the code to yield an `await`, and the task has the loop to itself in between.

The single thread and single task design is driven by the "C10K" problem: serving ten thousand concurrent connections with a thread each meant ten thousand stacks and endless context switching, while one event-loop thread handled them nearly for free. 

## MCP Server Basics

An MCP server is a program that exchanges JSON-RPC 2.0 messages with a client over a transport, and advertises capabilities: tools it can run, resources it serves, prompts it offers.

MCP servers have two types of transports:
- Stido transport: the host application launches the server as a child subprocess and talks to it through pipes. The relationship is strictly one client to one server process.
- Streamable HTTP transport: the server is a long-running web service that independent clients connect to. 

Activating an MCP server is the same for the two transports. The host application reads its config, establishes the transport (spawns the subprocess or opens the connection), and performs an initial round of handshake. Server spawning is not free: if the host application spawns fresh servers per agent or per task, session activation itself becomes a resource drain.

### Event loop


### Session startup

Activation is the same either way, and most "no tools found" or hang-at-startup bugs live in one of these steps. The host reads its config, establishes the transport (spawns the subprocess or opens the connection), and performs a handshake: the client sends an `initialize` request declaring its protocol version and capabilities, the server replies with its own, and the client confirms with an `initialized` notification. Only then does the client call `tools/list` to discover what's available, after which `tools/call` traffic flows.

Spawning is not free. A Python stdio server importing pandas, NumPy, and a BLAS backend can take seconds to start and immediately claims a full thread pool. If the orchestrator spawns fresh servers per subagent or per task, session activation itself becomes a resource event: a burst of process creation, memory allocation, and thread-pool instantiation. This is often the first mystery slowdown people hit when scaling agents.

### Two archetypes of server

A **local compute server** does the work itself. When `tools/call` arrives — "run this backtest" — the CPU burns inside the server process. Its constraints are physical: cores, memory, the GIL, BLAS thread pools. A **remote/API gateway server** does almost no work itself; a tool call translates to an outbound network request and a wait, often hundreds of milliseconds, with the CPU idle. Its constraints are external: rate limits, connection pools, upstream latency. Same protocol, same message shapes, opposite resource profiles — which is why one concurrency design cannot serve both.

### The event loop, and how concurrent requests actually flow

JSON-RPC is message-oriented: every request carries an `id`, and responses are matched by id, not arrival order. The protocol therefore fully supports many in-flight requests over one connection, with out-of-order completion. Whether requests are actually *processed* concurrently is entirely up to the server implementation.

Most MCP SDKs are built on an async event loop: a single thread juggling many tasks by switching between them at `await` points — moments where a task is waiting on I/O. For an I/O-bound gateway server this is a perfect fit: ten agents call `get_quote`, the server fires ten outbound requests, and while all ten are in flight the loop is free; responses return as upstreams answer and get matched by id. Massive concurrency, one thread.

For a CPU-bound tool the loop is a trap. A handler computing a big matrix decomposition never hits an `await`; for those seconds the single loop thread is occupied and the server cannot even *read* the next request off the pipe. Every other agent's call queues behind it. This is head-of-line blocking, and from the outside it looks exactly like "everything mysteriously locks up when I scale parallel agents doing heavy math."

### Sync versus async handlers: where should the work run?

The rule that falls out of all this: **the event loop is a dispatcher, never a computer.** A handler may spend microseconds parsing and validating on the loop, then it must hand the heavy work elsewhere and `await` the result.

"Elsewhere" has two main shapes. The first is a **thread pool** feeding GIL-releasing kernels:

```python
executor = ThreadPoolExecutor(max_workers=4)   # explicit parallelism budget

@mcp.tool()
async def run_backtest(params: BacktestParams) -> dict:
    arr = load_as_numpy(params)                 # Python glue, holds GIL, fast
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(        # loop is free again here
        executor, kernel.backtest, arr          # C/NumPy releases GIL inside
    )
    return format_result(result)
```

Because the kernel drops the GIL, the threads achieve true multi-core parallelism inside a single process — with zero-copy access to shared data, since threads share the address space. The price is blast radius: a segfault in native code kills the whole server and every in-flight request, and the kernel must now be thread-safe against itself.

The second shape is a **process pool**: full crash isolation and per-worker GILs, at the cost of the pickling tax on every input and output. The tax is neutralized by splitting traffic into two lanes — tiny control messages (task descriptors, segment names, shapes, dtypes) through the pool's normal queue, and bulk data through a named shared-memory segment that the coordinator writes once and every worker attaches to:

```python
# Coordinator: create once, write once
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes,
                                 name="prices_2026q3")
np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)[:] = arr

# Worker: attach by name — zero copy
shm = shared_memory.SharedMemory(name="prices_2026q3")
prices = np.ndarray(shape, dtype, buffer=shm.buf)
```

The synchronization story is best solved by making synchronization unnecessary: write-once, then immutable. Publish the segment name only after it is fully written, and treat it as read-only forever after; new data becomes a new segment, never a mutation. Concurrent readers of immutable data need no locks. (If hand-rolling this starts to feel like maintaining a bespoke object store, that's because it is one — Ray's shared-memory object store and Apache Arrow are the industrialized versions.)

The production pattern I've converged on is a hybrid: a small process pool for isolation between agents, where each worker internally runs the multi-threaded kernel design, and everyone attaches to an immutable shared data segment. Total CPU use becomes one explicit product: workers × threads per worker.

### Topology: one shared server, or one per agent?

Because stdio is one-server-per-client, N parallel agents force a choice. Share one connection to one server process, and you get one memory footprint and one BLAS pool — but head-of-line risk and state contention if the server is stateful. Give each agent its own server process, and you get isolation — but N interpreters' worth of memory, N startup costs, and N BLAS pools each defaulting to every core. Neither is wrong; the mature answer is usually one shared, *stateless*, well-pooled compute server with a bounded worker pool inside, rather than N naive ones.

## Failure Modes

### Head-of-line blocking

Covered above, but it earns its own entry because it is the most common and the most disguised. One synchronous call in one async handler — a stray `requests.get`, a heavy NumPy expression inline, even a `time.sleep` — and every concurrent request stalls behind it, heartbeats are missed, and timeouts fire spuriously across the whole fleet. Enforce the dispatcher-only rule by convention, and verify it mechanically: asyncio's debug mode logs any callback occupying the loop longer than `loop.slow_callback_duration`. It is a cheap tripwire that catches a blocking call the day it sneaks in, not the day it takes down a run.

### Oversubscription

The failure mode that motivated these notes. NumPy delegates real linear algebra to a BLAS library (OpenBLAS or MKL), and BLAS assumes it owns the machine: at first use it spawns roughly one thread per core it can see. Thread counts then *multiply* across layers. Eight workers each calling `np.linalg.svd` on a 16-core box is 8 × 16 = 128 compute-hungry runnable threads on 16 cores. Nothing crashes and nothing errors — the cores are 100% busy — but they are busy context-switching and rebuilding evicted caches, and throughput collapses, sometimes to worse than serial.

The fix is one budget equation, enforced deliberately:

```
workers × BLAS_threads_per_worker ≤ physical cores
```

Cap the hidden layer with `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` — set *before* NumPy is imported, i.e. in the Dockerfile or the worker launch environment — or surgically at runtime with `threadpoolctl`. Then spend the budget consciously: many workers × 1 BLAS thread for many independent modest tasks (parallel subagents), or few workers × many threads when individual operations are enormous and latency matters. The arithmetic applies to *runnable* CPU-bound threads; an I/O-bound pool whose threads mostly sleep can healthily exceed core count.

### Docker lies to your libraries

Two container gotchas that both masquerade as application bugs. First, BLAS sizes its pool by asking how many cores the machine has — and inside a container it typically sees the *host's* cores, not the cgroup CPU quota. A container limited to 4 CPUs on a 64-core host spawns 64 BLAS threads per worker: oversubscription squared, plus cgroup throttling on top. In containers, capping BLAS threads is not an optimization; it is mandatory. Second, Docker defaults `/dev/shm` to 64MB. A shared-memory design that works perfectly on the dev machine dies in the container with cryptic `SIGBUS` or "no space left on device" errors the first time a real dataset is written. Raise it explicitly (`--shm-size=8g` or the compose/k8s equivalent).

### Shared memory leaks

A segment is a named OS object, not a Python object, and no pool manages it for you. Every attaching process must `close()` its mapping; exactly one owner must `unlink()` to destroy the segment. Unlink twice and something crashes; unlink never and the segment outlives all your processes — it is a file in `/dev/shm`, pinning gigabytes of RAM after a crashed run until reboot or manual cleanup. Treat creation and destruction like file handles: context managers or `finally` blocks, always.

### Zombies, orphans, and cancellation that goes nowhere

stdio servers die when their pipe closes — usually. A crashed host, or a server that spawned grandchildren, leaves zombie processes holding memory and file descriptors, and with N agents cycling servers the leaks compound fast. Relatedly: MCP supports cancellation notifications, but a CPU-bound kernel that never yields cannot notice one arriving, and a hung upstream API means a `tools/call` that never returns. Every layer — agent, client, server, worker, outbound call — needs its own timeout; relying on a single layer's timeout is how you get orphaned work burning CPU long after the caller gave up. Supervising subprocesses and reaping them is unglamorous and essential.

## Closing: one budget, three knobs

If I compress everything above into a single mental model, it's this. An MCP server for heavy computation is a three-layer threading system: an event loop that must only dispatch, a worker pool whose size is a deliberate spend of the core budget, and a hidden BLAS layer that will spend the budget again behind your back unless capped. Data moves on two lanes: tiny control messages through queues, bulk arrays through immutable shared-memory segments that are created, closed, and unlinked with the same discipline as file handles. And containers change the defaults under you — core counts and `/dev/shm` most of all — so every one of these settings must be explicit in the image, not inherited from whatever machine the code happened to be written on.

None of these failures announce themselves with an error message. The OS is doing exactly what you asked; the trick is knowing what you asked for.