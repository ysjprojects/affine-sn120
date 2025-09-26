# Affine Framework Documentation

This document provides a comprehensive overview of all files in the `affine/` directory, detailing their purpose, functions, call signatures, and overarching architecture.

## Table of Contents

1. [Overview](#overview)
2. [Core Module (`__init__.py`)](#core-module-__init__py)
3. [Environment System (`envs/`)](#environment-system-envs)
4. [Database Module (`database.py`)](#database-module-databasepy)
5. [Networking & Communication](#networking--communication)
6. [Miners & Execution](#miners--execution)
7. [Utilities](#utilities)
8. [Logging System](#logging-system)

---

## Overview

The Affine framework is a distributed machine learning evaluation system built on the Bittensor network. It implements multiple computational environments (challenges) that test AI models across different domains including coding, mathematics, satisfiability problems, and virtual machine programming. The system uses a miner-validator architecture where miners deploy AI models and validators evaluate their performance.

---

## Core Module (`__init__.py`)

### Purpose
The main entry point that defines core data models, CLI interface, and global configuration for the Affine framework.

### Key Classes and Functions

#### Configuration Functions
- **`get_conf(key: str, default=None) -> Any`**
  - Retrieves environment variables or raises ValueError if not found
  - Used throughout the system for configuration management

- **`get_subtensor() -> bt.async_subtensor`**
  - Returns global singleton connection to Bittensor network
  - Handles connection initialization and fallback endpoints

#### Core Data Models

##### `BaseEnv(BaseModel, ABC)`
Abstract base class for all competition environments.
- **Properties:**
  - `name -> str`: Returns class name
- **Abstract Methods:**
  - `generate() -> Challenge`: Must generate new challenges
  - `evaluate(challenge: Challenge, response: Response) -> Evaluation`: Must evaluate responses

##### `Challenge(BaseModel)`
Represents a challenge/task given to miners.
- **Fields:**
  - `env: BaseEnv`: The environment that created this challenge
  - `prompt: str`: The challenge prompt for the miner
  - `extra: Dict[str, Any]`: Additional challenge data
  - `challenge_id: str`: Auto-generated unique identifier
- **Methods:**
  - `evaluate(resp: Response) -> Evaluation`: Delegates to environment's evaluate method

##### `Response(BaseModel)`
Represents a miner's response to a challenge.
- **Fields:**
  - `response: Optional[str]`: The actual response text
  - `latency_seconds: float`: Time taken to generate response
  - `attempts: int`: Number of retry attempts
  - `model: str`: Model identifier used
  - `error: Optional[str]`: Error message if failed
  - `success: bool`: Whether response was successful

##### `Evaluation(BaseModel)`
Represents the scored evaluation of a response.
- **Fields:**
  - `env: BaseEnv`: Environment that performed evaluation
  - `score: float`: Numeric score (typically 0.0-1.0)
  - `extra: Dict[str, Any]`: Additional evaluation metadata

##### `Miner(BaseModel)`
Represents a miner in the network.
- **Fields:**
  - `uid: int`: Unique identifier on the network
  - `hotkey: str`: Bittensor hotkey address
  - `model: Optional[str]`: Model name/identifier
  - `revision: Optional[str]`: Model revision/version
  - `block: Optional[int]`: Block number of registration
  - `chute: Optional[Dict[str, Any]]`: Chute deployment information
  - `slug: Optional[str]`: API endpoint slug

##### `Result(BaseModel)`
Complete result combining challenge, response, and evaluation.
- **Fields:**
  - `miner: Miner`: The miner that provided the response
  - `challenge: Challenge`: The original challenge
  - `response: Response`: The miner's response
  - `evaluation: Evaluation`: The scored evaluation
  - `version: str`: Framework version
  - `signature: str`: Cryptographic signature
  - `hotkey: str`: Signer's hotkey
- **Methods:**
  - `sign(wallet) -> None`: Signs the result with wallet
  - `verify() -> bool`: Verifies the signature

#### CLI Interface
- **`cli(verbose: int)`**: Main Click CLI group with verbosity control
- **Command: `upload-dataset`**: Uploads HuggingFace datasets to PostgreSQL
- **Command: `stats`**: Shows sample counts per environment and miner

#### Watchdog System
- **`watchdog(timeout: int = 300)`**: Process monitoring that exits if no heartbeat

---

## Environment System (`envs/`)

### `envs/__init__.py`
**Purpose:** Environment registry and dynamic loading system.

#### Functions
- **`_register_from_module(mod) -> None`**: Registers environment classes from a module
- **`get_env(name: str) -> Type[BaseEnv]`**: Retrieves environment class by name
- **`register_env(name: str, cls) -> None`**: Manually registers an environment

#### Global Registry
- **`ENVS: Dict[str, Type[BaseEnv]]`**: Central mapping of environment names to classes

### `envs/sat.py` - Boolean Satisfiability
**Purpose:** Tests logical reasoning through SAT problem solving.

#### Class: `SAT(BaseEnv)`
- **Constructor:** `__init__(n=15, k=10, m=None)`
  - `n`: Number of variables
  - `k`: Variables per clause  
  - `m`: Number of clauses
- **`generate() -> Challenge`**: Creates random satisfiable k-SAT formulas
- **`evaluate(challenge: Challenge, response: Response) -> Evaluation`**: Validates variable assignments

### `envs/abd.py` - Program Input Generation (Abduction)
**Purpose:** Tests ability to generate inputs that produce specific program outputs.

#### Class: `ABD(BaseEnv)`
- **`generate() -> Challenge`**: Uses LLM to generate new input for Python programs
- **`evaluate(challenge: Challenge, response: Response) -> Evaluation`**: Executes program with generated input
- **Helper Methods:**
  - `extract_input_from_response(response: str) -> str`: Parses `<INPUT>` tags
  - `_validate_input_for_program(program: str, inp: str) -> bool`: Validates input format
  - `compare_outputs(expected: str, actual: str) -> bool`: Normalizes and compares outputs

### `envs/ded.py` - Code Generation (Deduction) 
**Purpose:** Tests programming ability through code generation challenges.

#### Class: `DED(BaseEnv)`
- **`generate() -> Challenge`**: Provides programming problems with test cases
- **`evaluate(challenge: Challenge, response: Response) -> Evaluation`**: Runs code against test cases
- **Helper Functions:**
  - `_to_str(x) -> str`: Converts JSON data to stdin format
  - `_normalize(text: str) -> str`: Normalizes output for comparison

### `envs/elr.py` - Project Euler Problems
**Purpose:** Tests mathematical problem-solving skills.

#### Class: `ELR(BaseEnv)`
- **`generate() -> Challenge`**: Provides Project Euler mathematical problems
- **`evaluate(challenge: Challenge, response: Response) -> Evaluation`**: Checks numerical answers
- **Helper Methods:**
  - `extract_answer_from_response(response: str) -> str`: Parses `<Answer>` tags

### `envs/hvm.py` - Hole-filled Virtual Machine
**Purpose:** Tests program synthesis through stack-based VM programming.

#### Class: `HVM(BaseEnv)`
- **Constructor:** `__init__(seed: Optional[int] = None)`
- **`generate() -> Challenge`**: Creates VM programs with unknown constants (holes)
- **`evaluate(challenge: Challenge, response: Response) -> Evaluation`**: Validates hole assignments
- **Core Methods:**
  - `_make_program(hard: bool) -> Dict[str, Any]`: Generates VM program specifications
  - `_forge_io(prog: Dict, n_cases: int) -> Tuple[List[List[int]], List[str]]`: Creates test cases
  - `_run_vm_local(prog: Dict, holes: Dict, inputs: List[int]) -> Tuple[bool, str]`: Local VM execution
  - `_run_vm_sandbox(prog: Dict, holes: Dict, inputs: List[int]) -> Tuple[bool, str]`: Sandboxed execution
  - `_parse_holes(text: str) -> Optional[Dict[str, int]]`: Parses `<HOLES>` block

---

## Database Module (`database.py`)

### Purpose
PostgreSQL database interface for storing results, datasets, and providing query capabilities.

### Configuration
- Supports both read-only and write access modes
- Automatic database creation and schema migration
- Connection pooling with configurable limits

### Schema

#### `affine_results` Table
Stores evaluation results from miners.

**Key Columns:**
- `env_name`, `uid`, `hotkey`, `model`, `revision`: Identification
- `prompt`, `response`, `score`: Challenge data
- `challenge_id`: Unique challenge identifier for deduplication
- `success`, `latency_seconds`, `attempts`, `error`: Performance metrics
- `r2_key`, `r2_last_modified`: Data provenance
- `extra`: JSON metadata storage

#### `dataset_rows` Table
Stores raw dataset rows for environments.

### Core Functions

#### Engine Management
- **`_get_engine() -> AsyncEngine`**: Returns initialized database engine
- **`_sm() -> async_sessionmaker`**: Returns session maker

#### Query Interface
- **`count(**filters: Any) -> int`**: Counts rows with flexible filtering
- **`select_rows(*, limit: int = 1000, order: str = "r2_last_modified", ascending: bool = False, **filters) -> List[Dict[str, Any]]`**: Generic row selection
- **`aggregate_success_by_env(*, env_name: str, pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]`**: Aggregates performance metrics
- **`get_env_counts(*, pairs: List[Tuple[str, str]]) -> Dict[str, Dict[Tuple[str, str], int]]`**: Counts by environment

#### Dataset Management  
- **`select_dataset_rows(*, dataset_name: str, config: str = "default", split: str = "train", limit: int = 1000, offset: int = 0, include_index: bool = False) -> List[Dict[str, Any]]`**: Fetches dataset rows

#### Data Persistence
- **`sink(*, wallet: Any, results: List[Result], block: Optional[int] = None) -> None`**: Persists results to database

---

## Networking & Communication

### `chutes.py` - Model Query System
**Purpose:** HTTP client for querying deployed AI models and managing miner information.

#### Key Functions

##### Model Querying
- **`query(prompt: str, model: str = "unsloth/gemma-3-12b-it", slug: str = "llm", timeout: int = 150, retries: int = 0, backoff: int = 1) -> Response`**
  - Sends prompts to deployed models via HTTP API
  - Handles retries, timeouts, and error cases
  - Returns structured Response objects

##### Model Management
- **`check_model_gated(model_id: str, revision: Optional[str] = None) -> Optional[bool]`**
  - Checks if HuggingFace model is gated/private
  - Implements caching to reduce API calls

##### Execution Pipeline
- **`run(challenges, miners, timeout: int = 240, retries: int = 0, backoff: int = 1) -> List[Result]`**
  - Orchestrates challenge execution across miners
  - Handles concurrent processing and progress logging

##### Chute API Integration
- **`get_chute(chutes_id: str) -> Dict[str, Any]`**: Retrieves chute deployment info
- **`get_chute_code(identifier: str) -> Optional[str]`**: Gets chute source code
- **`get_latest_chute_id(model_name: str, api_key: Optional[str] = None) -> Optional[str]`**: Finds latest chute for model

##### Miner Discovery
- **`get_miners(uids: Optional[Union[int, List[int]]] = None, netuid: int = NETUID, meta: object = None) -> Dict[int, Miner]`**
  - Discovers active miners on Bittensor network
  - Filters by model availability and gating status
  - Handles deduplication by model/revision pairs

### `signer.py` - Cryptographic Services
**Purpose:** Handles cryptographic signing and blockchain weight setting.

#### Functions

##### Result Signing
- **`sign_results(wallet: Any, results: List[Result]) -> Tuple[str, List[Result]]`**
  - Signs results via external signer service or local wallet
  - Returns hotkey address and signed results

##### Weight Management
- **`_set_weights_with_confirmation(wallet, netuid: int, uids: List[int], weights: List[float], wait_for_inclusion: bool = False, retries: int = 10, delay_s: float = 2.0, log_prefix: str = "") -> bool`**
  - Sets validator weights with confirmation checking
  - Implements retry logic for network failures

- **`retry_set_weights(wallet, uids: List[int], weights: List[float], retry: int = 10)`**
  - High-level weight setting with signer service integration
  - Fallback to local signing if signer unavailable

#### CLI Commands
- **`signer`**: Starts HTTP signer service with endpoints:
  - `/healthz`: Health check
  - `/sign`: Signs arbitrary payloads  
  - `/set_weights`: Sets validator weights

---

## Miners & Execution

### `miners.py` - Model Deployment
**Purpose:** CLI commands for deploying and managing AI models as miners.

#### CLI Commands

##### `pull` - Download Models
- **Signature:** `pull(uid: int, model_path: str, hf_token: str)`
- **Purpose:** Downloads miner's model from HuggingFace
- **Process:**
  1. Looks up miner by UID on-chain
  2. Downloads model snapshot to local directory
  3. Handles authentication and resume capabilities

##### `push` - Deploy Models  
- **Signature:** `push(model_path: str, existing_repo: str, revision: str, coldkey: str, hotkey: str, chutes_api_key: str)`
- **Purpose:** Complete model deployment pipeline
- **Process:**
  1. Creates/secures HuggingFace repository
  2. Uploads model files (unless using existing repo)
  3. Deploys to Chutes hosting platform
  4. Commits model metadata to Bittensor chain
  5. Performs warmup to mark model as "hot"

### `runner.py` - Continuous Evaluation
**Purpose:** Continuously runs challenges across miners for data collection.

#### Core Function
- **CLI Command: `runner`**
- **Architecture:**
  - Maintains pool of active miners
  - Implements weighted selection based on sample counts
  - Uses backoff for failed miners
  - Concurrent execution with semaphore limits
  - Periodic refresh of miner/count data

#### Selection Algorithm
- Selects environment with fewest total samples
- Within environment, weights miners by: `(mean_count + ε) / (miner_count + backoff + ε)`
- Implements exponential backoff for failing miners

### `validator.py` - Consensus and Scoring
**Purpose:** Validator logic for computing miner weights and setting consensus.

#### Main Function
- **`get_weights(tail: int = TAIL, scale: float = 1) -> Tuple[List[int], List[float]]`**

#### Scoring Algorithm
1. **Data Collection**: Aggregates recent results per miner per environment
2. **Eligibility**: Requires minimum sample counts across all environments  
3. **ε-Pareto Dominance**: 
   - Statistical significance testing for "not worse" comparisons
   - Strict improvement thresholds for "better" claims
   - Tie-breaking by earliest deployment block
4. **Combinatorial Scoring**:
   - For each subset of environments, finds ε-Pareto winner
   - Awards points: K₁ = scale, Kₛ = C(N, s-1) × Kₛ₋₁
   - Weights increase exponentially with subset size
5. **Weight Normalization**: Converts scores to probability distribution

#### Statistical Functions
- **`thr_not_worse(a_i: float, n_i: int, a_j: float, n_j: int) -> float`**: Computes tolerance for "not worse"
- **`thr_better(a_i: float, n_i: int, a_j: float, n_j: int, nw: float) -> float`**: Computes margin for "better"
- **`dominates_on(a: str, b: str, subset) -> bool`**: Tests ε-dominance relationship

#### CLI Commands
- **`validate`**: Runs continuous validation loop
- **`weights`**: One-time weight computation

---

## Utilities

### `utils/dataset.py` - Data Management
**Purpose:** Buffered dataset loading from PostgreSQL.

#### Class: `R2BufferedDataset`
- **Constructor:** `__init__(dataset_name: str, total_size: int = 0, buffer_size: int = 100, max_batch: int = 10, seed: Optional[int] = None, config: str = "default", split: str = "train")`
- **`get() -> Any`**: Retrieves next dataset item with async buffering
- **`_fill_buffer() -> None`**: Asynchronously fills internal buffer
- **`_read_next_rows(desired: int) -> List[Any]`**: Reads from PostgreSQL with wraparound

### `utils/executor.py` - Sandboxed Code Execution  
**Purpose:** Secure Python code execution for programming challenges.

#### Class: `ProgramExecutor`
- **Constructor:** `__init__(timeout: int = 30, cpu_time: int = 10, mem_bytes: int = 512MB, max_output: int = 1MB)`

#### Core Methods
- **`execute(raw_code: str, stdin: str | bytes = "") -> Tuple[str, str]`**
  - Executes Python code with input
  - Returns (stdout, stderr) tuple
  - Implements two-stage execution for solve() functions

#### Security Features
- **Resource Limits**: CPU time, memory, output size limits
- **Process Isolation**: Separate process groups, cleanup on timeout
- **Code Sanitization**: Fence stripping, automatic main detection
- **Incremental I/O**: Non-blocking reads to prevent hangs

#### Helper Methods
- **`_strip_fences(text: str) -> str`**: Extracts code from markdown fences
- **`_posix_rlimits() -> None`**: Applies POSIX resource limits
- **`_run_once(script: str, stdin_data: str) -> Tuple[str, str]`**: Low-level execution

---

## Logging System

### `logging.py`
**Purpose:** Centralized logging configuration and metrics collection.

#### Configuration
- **`setup_logging(verbosity: int)`**: Configures log levels and Prometheus metrics
- **Log Levels**: CRITICAL+1 (silent), INFO, DEBUG, TRACE (custom level)
- **Metrics Server**: Starts Prometheus HTTP server on configurable port

#### Prometheus Metrics
- **`QCOUNT`**: Query count by model
- **`SCORE`**: Performance scores by UID and environment  
- **`RANK`**: Ranking metrics by UID and environment
- **`WEIGHT`**: Validator weights by UID
- **`LASTSET`**: Last weight-setting timestamp
- **`NRESULTS`**: Total results count
- **`MAXENV`**: Maximum score per environment
- **`CACHE`**: Cache hit metrics

#### Constants
- **`NETUID = 120`**: Bittensor subnet identifier
- **`TRACE = 5`**: Custom trace logging level

#### Utility Functions
- **`singleton(key: str, factory) -> callable`**: Creates singleton factory functions
- **`info()`, `debug()`, `trace()`**: Quick logging setup functions

---

## Architecture Summary

The Affine framework implements a comprehensive distributed AI evaluation system with the following key characteristics:

1. **Modular Environments**: Pluggable challenge types (SAT, coding, math, VM programming)
2. **Secure Execution**: Sandboxed code execution with resource limits
3. **Statistical Validation**: ε-Pareto dominance with significance testing  
4. **Scalable Infrastructure**: Async I/O, connection pooling, concurrent processing
5. **Blockchain Integration**: Bittensor network for consensus and incentives
6. **Production Ready**: Comprehensive logging, metrics, error handling, and monitoring

The system supports both miner operations (model deployment and serving) and validator operations (challenge generation, evaluation, and consensus), creating a complete ecosystem for decentralized AI evaluation and improvement.