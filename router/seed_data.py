"""
Seed training dataset for the v0 ML router.

Labels:
    "small" — cheap, deterministic, one-shot (small LLM handles fine)
    "big"   — multi-step reasoning, synthesis, proofs, code design,
              anything where small LLM quality is likely to degrade

~120 examples, roughly balanced. Hand-crafted to cover:
    greetings, factual lookups, trivia, unit conversions,
    light code (small) ────────── vs. ──────────
    proofs, architecture, system design, analysis, debugging,
    multi-constraint optimization, research-style questions (big)

Add more rows as you collect real data via ClassificationLogger and
retrain with router/train.py.
"""

SEED_DATASET: list[tuple[str, str]] = [
    # ── small: greetings / small talk ─────────────────────────────────────
    ("Hi! How are you?", "small"),
    ("Hello there", "small"),
    ("Hey, what's up?", "small"),
    ("Good morning!", "small"),
    ("Thanks!", "small"),
    ("Thank you so much", "small"),
    ("Good night", "small"),
    ("Bye", "small"),

    # ── small: factual lookups ────────────────────────────────────────────
    ("What is the capital of Japan?", "small"),
    ("Who wrote Hamlet?", "small"),
    ("When did World War 2 end?", "small"),
    ("What is the speed of light?", "small"),
    ("Where is the Eiffel Tower?", "small"),
    ("Define photosynthesis", "small"),
    ("What is the population of France?", "small"),
    ("Who is the president of the United States?", "small"),
    ("What year did the Titanic sink?", "small"),
    ("What is the boiling point of water?", "small"),
    ("Capital of Australia?", "small"),
    ("What language do they speak in Brazil?", "small"),
    ("How many continents are there?", "small"),
    ("What is the largest ocean?", "small"),
    ("What is H2O?", "small"),

    # ── small: simple conversions / calculations ─────────────────────────
    ("Convert 5 miles to kilometers", "small"),
    ("How many ounces in a pound?", "small"),
    ("What is 15% of 200?", "small"),
    ("What is 2 + 2?", "small"),
    ("Convert 100 celsius to fahrenheit", "small"),
    ("How many minutes in a day?", "small"),

    # ── small: trivia / quick facts ──────────────────────────────────────
    ("What color is a giraffe's tongue?", "small"),
    ("Tallest mountain in the world?", "small"),
    ("Who invented the telephone?", "small"),
    ("What is the chemical symbol for gold?", "small"),
    ("How fast does sound travel?", "small"),

    # ── small: simple definitions ────────────────────────────────────────
    ("What is a noun?", "small"),
    ("Define empathy", "small"),
    ("What does DNS stand for?", "small"),
    ("What is an API?", "small"),
    ("What is HTTP?", "small"),

    # ── small: light formatting / summarization (very short input) ───────
    ("List 3 fruits", "small"),
    ("Name 5 planets", "small"),
    ("Give me a synonym for happy", "small"),

    # ── small: yes/no style clarifications ───────────────────────────────
    ("Is Python interpreted or compiled?", "small"),
    ("Is the sun a star?", "small"),
    ("Are whales mammals?", "small"),
    ("Is SQL case sensitive?", "small"),

    # ── small: single-concept code questions ─────────────────────────────
    ("What does `pip install` do?", "small"),
    ("What is the syntax for a Python list?", "small"),
    ("How do I print in Python?", "small"),
    ("What does `git status` show?", "small"),

    # ── small: personal / preference style ───────────────────────────────
    ("What should I eat for breakfast?", "small"),
    ("Recommend a fun hobby", "small"),
    ("Give me a motivational quote", "small"),

    # ── small: short creative prompts ────────────────────────────────────
    ("Tell me a joke", "small"),
    ("Say something nice", "small"),
    ("Make a pun about cats", "small"),

    # ── small: short time/date ───────────────────────────────────────────
    ("What day is it today?", "small"),
    ("What time zone is New York in?", "small"),
    ("How many days until Christmas?", "small"),

    # ── small: list-ish answers ──────────────────────────────────────────
    ("List the primary colors", "small"),
    ("What are the noble gases?", "small"),
    ("Name the Great Lakes", "small"),

    # ── small: brief opinion ─────────────────────────────────────────────
    ("What is your favorite color?", "small"),

    # ── big: deep reasoning / analysis ───────────────────────────────────
    ("Explain how transformer self-attention works and why it scales better than RNNs.", "big"),
    ("Why does gradient descent work, and when does it fail?", "big"),
    ("Compare the CAP theorem trade-offs in Cassandra vs MongoDB vs Spanner.", "big"),
    ("Analyze the economic causes of the 2008 financial crisis.", "big"),
    ("Explain the differences between eventual, causal, and strong consistency in distributed systems.", "big"),
    ("How does backpropagation through time differ from standard backprop, and why does it cause vanishing gradients?", "big"),
    ("Compare and contrast REST, GraphQL, and gRPC. When should I use each?", "big"),

    # ── big: proofs / math derivations ───────────────────────────────────
    ("Prove that the square root of 2 is irrational.", "big"),
    ("Derive the closed-form solution for linear regression.", "big"),
    ("Prove Bayes' theorem from the definition of conditional probability.", "big"),
    ("Show that the sum of two continuous functions is continuous.", "big"),
    ("Prove that there are infinitely many prime numbers.", "big"),
    ("Derive the backpropagation equations for a 2-layer neural network.", "big"),

    # ── big: system design ───────────────────────────────────────────────
    ("Design a URL shortener that handles 100M requests per day.", "big"),
    ("How would you architect a real-time chat application with 10M concurrent users?", "big"),
    ("Design a recommendation system for a music streaming service.", "big"),
    ("Design a distributed rate limiter across multiple data centers.", "big"),
    ("How would you build a search engine from scratch? Walk through the architecture.", "big"),

    # ── big: multi-step code / debugging ─────────────────────────────────
    ("My Flask app crashes under load with 'too many open files'. Walk me through diagnosing and fixing it.", "big"),
    ("Refactor this function for readability and explain your choices: [pretend code here with many branches]", "big"),
    ("Write a Python function that detects cycles in a directed graph and explain the algorithm.", "big"),
    ("Implement and explain the A* pathfinding algorithm with an admissible heuristic.", "big"),
    ("Design a thread-safe LRU cache in Python and discuss the trade-offs.", "big"),

    # ── big: compound / multi-constraint questions ──────────────────────
    ("I need to reduce my AWS bill by 40% while keeping 99.9% uptime. What's the plan?", "big"),
    ("How should I design a machine learning pipeline that retrains daily on 1TB of new data?", "big"),
    ("Plan a migration from a monolith to microservices for a 10-person team over 6 months.", "big"),

    # ── big: research-style / synthesis ──────────────────────────────────
    ("Summarize the key differences between diffusion models and GANs for image generation, including training stability.", "big"),
    ("What are the main open problems in alignment research for large language models?", "big"),
    ("Compare the pros and cons of on-policy vs off-policy reinforcement learning algorithms.", "big"),
    ("Walk me through the evolution of CNN architectures from LeNet to ConvNeXt.", "big"),

    # ── big: ambiguous / requires clarification-style reasoning ──────────
    ("Help me think through whether to use Postgres or DynamoDB for a multi-tenant SaaS.", "big"),
    ("Should I use a monorepo or polyrepo setup for a team of 20 engineers? Discuss.", "big"),

    # ── big: deep domain-specific ────────────────────────────────────────
    ("Explain the role of attention masks in causal vs bidirectional language models.", "big"),
    ("Why do we use layer normalization in transformers instead of batch normalization?", "big"),
    ("Explain how Raft consensus achieves leader election and log replication.", "big"),
    ("How does TLS 1.3 improve handshake latency over TLS 1.2? Walk through the steps.", "big"),

    # ── big: long multi-part prompts ─────────────────────────────────────
    ("I'm building a SaaS for small dental clinics. They need appointment scheduling, patient records, insurance claims, and SMS reminders. Outline the domain model, tech stack, and the riskiest parts of this build.", "big"),
    ("Given a dataset of 50M customer transactions, I want to detect fraud. Discuss feature engineering, model choice, handling class imbalance, evaluation, and how to deploy and monitor.", "big"),

    # ── big: tricky edge-cases where small LLM typically hallucinates ────
    ("What are the subtle differences between `__str__`, `__repr__`, and `__format__` in Python, and when is each called?", "big"),
    ("Explain how JavaScript's event loop handles microtasks vs macrotasks with an example.", "big"),
    ("Why does `1.0 + 2.0 == 3.0` in Python but `0.1 + 0.2 != 0.3`? Explain IEEE 754.", "big"),

    # ── big: creative with constraints ───────────────────────────────────
    ("Write a short story in exactly 200 words about an AI that learns regret, and explain your structural choices.", "big"),

    # ── big: stepwise problem solving ────────────────────────────────────
    ("A train leaves Boston at 60 mph heading west. Another train leaves Chicago at 80 mph heading east. Chicago is 1000 miles from Boston. When and where do they meet? Show all steps.", "big"),
    ("I have 8 balls, one is slightly heavier. Using a balance scale twice, how do I find it? Explain the reasoning.", "big"),

    # ── big: comparisons requiring synthesis ─────────────────────────────
    ("Compare Rust, Go, and Zig for writing a new database engine.", "big"),
    ("What's the difference between Kafka, RabbitMQ, and AWS SQS? When should I pick each?", "big"),
]


def load_seed() -> list[tuple[str, str]]:
    """Return the seed dataset as a list of (query, label) tuples."""
    return list(SEED_DATASET)
