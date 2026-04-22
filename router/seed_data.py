"""
Seed training dataset for the v0 ML router.

Labels:
    "small" — cheap, deterministic, one-shot (small LLM handles fine)
    "big"   — multi-step reasoning, synthesis, proofs, code design,
              anything where small LLM quality is likely to degrade

~280 examples, roughly balanced. Hand-crafted to cover:
    greetings, factual lookups, trivia, unit conversions,
    light code (small) ────────── vs. ──────────
    proofs, architecture, system design, analysis, debugging,
    multi-constraint optimization, research-style questions (big)
    plus: ambiguous/opinion, creative, career/learning, research synthesis
    (added in v1 expansion to address eval weak spots)

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

    # ══════════════════════════════════════════════════════════════════════════
    # V1 EXPANSION — 160 new examples added to address eval_500 weak spots:
    #   ambiguous/opinion (big), creative (small), research synthesis (big),
    #   career/learning (big), factual-that-looks-big (small)
    # ══════════════════════════════════════════════════════════════════════════

    # ── AMBIGUOUS / OPINION "big" — 40 new examples ──────────────────────────
    # These triggered HAS_FACTUAL_KW on "what is" but are actually
    # philosophical, comparative, or opinion-requiring questions.
    ("What is consciousness?", "big"),
    ("What is the best programming language?", "big"),
    ("Is AI dangerous?", "big"),
    ("What causes inflation?", "big"),
    ("Why is the sky blue?", "big"),
    ("How do vaccines work?", "big"),
    ("How do neural networks learn?", "big"),
    ("How does sleep affect memory?", "big"),
    ("What makes a good leader?", "big"),
    ("Is remote work better than in-office?", "big"),
    ("How do I start investing?", "big"),
    ("What is the meaning of life?", "big"),
    ("What is a good salary in San Francisco?", "big"),
    ("Should I use React or Vue?", "big"),
    ("Is Python better than Java?", "big"),
    ("What are the pros and cons of microservices?", "big"),
    ("How does the stock market work?", "big"),
    ("What is the best database for a startup?", "big"),
    ("Why should I use Docker?", "big"),
    ("Is Kubernetes worth learning?", "big"),
    ("What is the impact of social media on mental health?", "big"),
    ("How does caffeine affect the brain?", "big"),
    ("What is quantum computing and why does it matter?", "big"),
    ("How does climate change work?", "big"),
    ("What is the best way to learn programming?", "big"),
    ("Is college worth it for software engineers?", "big"),
    ("How should I negotiate my salary?", "big"),
    ("What's the difference between a startup and a big company for a new grad?", "big"),
    ("How do I know if I'm ready for senior engineer?", "big"),
    ("What is the best way to prepare for a technical interview?", "big"),
    ("What is the difference between supervised and unsupervised learning?", "big"),
    ("What is the impact of automation on jobs?", "big"),
    ("What are the implications of quantum computing for cryptography?", "big"),
    ("How does blockchain technology actually work?", "big"),
    ("What is the best architecture for a microservices system?", "big"),
    ("How does the immune system fight viruses?", "big"),
    ("What is the best way to structure a machine learning project?", "big"),
    ("Why do some startups fail and others succeed?", "big"),
    ("What is the effect of interest rates on the economy?", "big"),
    ("How do I choose between a PhD and industry after undergrad?", "big"),

    # ── CREATIVE "small" — 25 new examples ───────────────────────────────────
    # Short, one-shot creative requests that a small model handles perfectly.
    ("Tell me a fun fact", "small"),
    ("Give me a motivational quote about coding", "small"),
    ("Write a haiku about debugging", "small"),
    ("Make up a fun name for a startup", "small"),
    ("Give me a clever pun about Python", "small"),
    ("Write a limerick about git commits", "small"),
    ("Tell me a programming joke", "small"),
    ("Give me a rhyme about algorithms", "small"),
    ("Write a one-liner joke about databases", "small"),
    ("Make up a funny team name for hackathon", "small"),
    ("Give me a riddle about software", "small"),
    ("Write a haiku about machine learning", "small"),
    ("Tell me a knock-knock joke about servers", "small"),
    ("Give me a fun tagline for a developer tool", "small"),
    ("Make up a pun about recursion", "small"),
    ("Tell me a joke about JavaScript", "small"),
    ("Give me a short poem about open source", "small"),
    ("Write a two-line poem about bugs", "small"),
    ("Give me a punny name for a DevOps tool", "small"),
    ("Tell me a fun coding riddle", "small"),
    ("Give me a motivational quote about persistence", "small"),
    ("Write a haiku about cloud computing", "small"),
    ("Make up a silly variable name for happiness", "small"),
    ("Give me a fun fact about programming history", "small"),
    ("Tell me a short story about a compiler error", "small"),

    # ── RESEARCH SYNTHESIS "big" — 20 new examples ───────────────────────────
    ("What does the research say about pair programming effectiveness?", "big"),
    ("What do studies show about the benefits of code review?", "big"),
    ("What does the literature say about the Dunning-Kruger effect?", "big"),
    ("What does research show about sleep deprivation on cognitive performance?", "big"),
    ("What are the findings on spaced repetition versus massed practice?", "big"),
    ("Summarize the evidence on remote work productivity from recent studies.", "big"),
    ("What does the research say about the effectiveness of agile methods?", "big"),
    ("What do studies show about the impact of diversity on team performance?", "big"),
    ("What is the current research on transformer model scaling laws?", "big"),
    ("What does the literature say about technical debt accumulation?", "big"),
    ("What does research show about the long-term effects of caffeine on cognition?", "big"),
    ("What are the findings from research on code readability and maintainability?", "big"),
    ("What does the evidence say about test-driven development effectiveness?", "big"),
    ("What do meta-analyses show about the effectiveness of code comments?", "big"),
    ("What does the research say about developer productivity measurement?", "big"),
    ("What does the literature on burnout say about software engineers?", "big"),
    ("What are the research findings on deep learning versus classical ML for tabular data?", "big"),
    ("What studies have been done on the impact of open-plan offices on productivity?", "big"),
    ("What does the evidence show about microservices versus monoliths in practice?", "big"),
    ("What does recent research say about AI safety and alignment?", "big"),

    # ── CAREER / LEARNING "big" — 20 new examples ────────────────────────────
    ("What skills should I learn as a junior developer?", "big"),
    ("How do I transition from web dev to ML engineering?", "big"),
    ("What should I focus on in my first 90 days as a software engineer?", "big"),
    ("How do I build a portfolio that gets me a data science job?", "big"),
    ("What are the best resources for learning system design?", "big"),
    ("How do I get promoted faster as a software engineer?", "big"),
    ("What is the best way to learn a new programming language quickly?", "big"),
    ("How do I prepare for behavioral interviews at top tech companies?", "big"),
    ("What certifications are worth getting for cloud engineering?", "big"),
    ("How do I move from a backend role to a full-stack position?", "big"),
    ("What are the most important algorithms to know for coding interviews?", "big"),
    ("How do I build leadership skills as an individual contributor?", "big"),
    ("What is the best way to network in the tech industry?", "big"),
    ("How do I evaluate whether a company has a good engineering culture?", "big"),
    ("What should I consider when choosing between two job offers?", "big"),
    ("How do I break into AI research without a PhD?", "big"),
    ("What are the best open-source projects to contribute to for career growth?", "big"),
    ("How do I stay motivated while learning machine learning independently?", "big"),
    ("What should I know about equity compensation before joining a startup?", "big"),
    ("How do I make the most of a software engineering internship?", "big"),

    # ── FACTUAL "small" easily confused with "big" — 15 new examples ─────────
    # These look abstract but have a single short correct answer.
    ("What is the capital of South Korea?", "small"),
    ("What is the boiling point of water in Fahrenheit?", "small"),
    ("Who invented the internet?", "small"),
    ("What year was Python created?", "small"),
    ("Who founded Apple?", "small"),
    ("What does CPU stand for?", "small"),
    ("What language is spoken in Argentina?", "small"),
    ("What is the atomic weight of oxygen?", "small"),
    ("Who wrote the Communist Manifesto?", "small"),
    ("What country is Zurich in?", "small"),
    ("What does RAM stand for?", "small"),
    ("Who invented C++?", "small"),
    ("What is the chemical symbol for iron?", "small"),
    ("What is the population of Tokyo?", "small"),
    ("What does SQL stand for?", "small"),

    # ── EXTRA AMBIGUOUS OPINION "big" — 10 more ──────────────────────────────
    ("What is the best text editor for Python development?", "big"),
    ("What is the most important skill for a software engineer?", "big"),
    ("Is TypeScript worth learning if you already know JavaScript?", "big"),
    ("What is the impact of open source on the software industry?", "big"),
    ("How do you build a strong engineering culture?", "big"),
    ("What are the trade-offs of using a managed Kubernetes service?", "big"),
    ("Is it better to specialize or be a generalist as an engineer?", "big"),
    ("What is the best way to handle disagreements in a technical team?", "big"),
    ("How do I decide when to refactor versus rewrite code?", "big"),
    ("What are the implications of choosing a relational over a NoSQL database?", "big"),
]


def load_seed() -> list[tuple[str, str]]:
    """Return the seed dataset as a list of (query, label) tuples."""
    return list(SEED_DATASET)
