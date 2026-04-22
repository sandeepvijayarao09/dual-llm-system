"""
eval_500.py — Offline evaluation of the ML Router against 500 labeled test cases.

Usage:
    python -m router.eval_500

Requirements:
    - router/models/router_v0.joblib must exist  (run: python -m router.train)
    - scikit-learn, joblib installed

No LLM calls are made.  Only the MLRouter (TF-IDF + Logistic Regression) is
evaluated.  Cascade rate is computed as a simulation: queries where
ML confidence < ML_ROUTER_CONFIDENCE_THRESHOLD (0.65) *would* fall back to
the LLM classifier in production.
"""

from __future__ import annotations

import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 500 labeled test cases
# Format: (query, ground_truth_label, category)
# ground_truth: "small" | "big"
# category: human-readable bucket used in per-category breakdown
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = [
    # =========================================================
    # GREETINGS  (20 cases — all "small")
    # =========================================================
    ("Hi!", "small", "greetings"),
    ("Hello!", "small", "greetings"),
    ("Hey there", "small", "greetings"),
    ("Good morning!", "small", "greetings"),
    ("Good afternoon", "small", "greetings"),
    ("Good evening!", "small", "greetings"),
    ("Good night", "small", "greetings"),
    ("Thanks!", "small", "greetings"),
    ("Thank you so much", "small", "greetings"),
    ("Bye!", "small", "greetings"),
    ("See you later", "small", "greetings"),
    ("Take care!", "small", "greetings"),
    ("How are you?", "small", "greetings"),
    ("What's up?", "small", "greetings"),
    ("Howdy!", "small", "greetings"),
    ("Sup?", "small", "greetings"),
    ("Cheers!", "small", "greetings"),
    ("Have a great day", "small", "greetings"),
    ("Hope you're doing well", "small", "greetings"),
    ("Pleasure to meet you", "small", "greetings"),

    # =========================================================
    # FACTUAL LOOKUPS  (60 cases — all "small")
    # =========================================================
    ("What is the capital of France?", "small", "factual"),
    ("Who wrote Romeo and Juliet?", "small", "factual"),
    ("When did World War 1 start?", "small", "factual"),
    ("What is the capital of Japan?", "small", "factual"),
    ("Who invented the telephone?", "small", "factual"),
    ("What year did the Titanic sink?", "small", "factual"),
    ("What is the speed of light?", "small", "factual"),
    ("Where is the Eiffel Tower located?", "small", "factual"),
    ("What is the population of China?", "small", "factual"),
    ("Who is the author of Harry Potter?", "small", "factual"),
    ("What is the largest planet in our solar system?", "small", "factual"),
    ("What is the boiling point of water?", "small", "factual"),
    ("How many bones are in the human body?", "small", "factual"),
    ("What country is the Amazon rainforest in?", "small", "factual"),
    ("Who painted the Mona Lisa?", "small", "factual"),
    ("What is the chemical formula for water?", "small", "factual"),
    ("What currency does Japan use?", "small", "factual"),
    ("How tall is Mount Everest?", "small", "factual"),
    ("What language is spoken in Portugal?", "small", "factual"),
    ("Who was the first person on the moon?", "small", "factual"),
    ("What is the largest ocean on Earth?", "small", "factual"),
    ("What is the capital of Canada?", "small", "factual"),
    ("Who wrote 1984?", "small", "factual"),
    ("What is the atomic number of carbon?", "small", "factual"),
    ("What year was the iPhone first released?", "small", "factual"),
    ("What is the national bird of the United States?", "small", "factual"),
    ("What planet is closest to the sun?", "small", "factual"),
    ("What is the smallest country in the world?", "small", "factual"),
    ("Who discovered penicillin?", "small", "factual"),
    ("What is the capital of Brazil?", "small", "factual"),
    ("What is the longest river in Africa?", "small", "factual"),
    ("How many teeth does an adult human have?", "small", "factual"),
    ("What element does the symbol Fe represent?", "small", "factual"),
    ("What year did the Berlin Wall fall?", "small", "factual"),
    ("How many chambers does the human heart have?", "small", "factual"),
    ("Who was the first president of the United States?", "small", "factual"),
    ("What is the hardest natural substance on Earth?", "small", "factual"),
    ("What does DNA stand for?", "small", "factual"),
    ("Who invented the light bulb?", "small", "factual"),
    ("What is the tallest mammal?", "small", "factual"),
    ("What is the capital of Australia?", "small", "factual"),
    ("How many days are in a leap year?", "small", "factual"),
    ("What is the chemical symbol for sodium?", "small", "factual"),
    ("What is the deepest ocean trench on Earth?", "small", "factual"),
    ("Who wrote Pride and Prejudice?", "small", "factual"),
    ("What is the longest bone in the human body?", "small", "factual"),
    ("What year was the United Nations founded?", "small", "factual"),
    ("What continent is Egypt in?", "small", "factual"),
    ("What is the fastest land animal?", "small", "factual"),
    ("Who developed the theory of relativity?", "small", "factual"),
    ("What is the capital of India?", "small", "factual"),
    ("How many planets are in our solar system?", "small", "factual"),
    ("What is the freezing point of water in Celsius?", "small", "factual"),
    ("Who is known as the father of computers?", "small", "factual"),
    ("What is the largest desert in the world?", "small", "factual"),
    ("What does HTTP stand for?", "small", "factual"),
    ("What is the capital of Germany?", "small", "factual"),
    ("Who composed Beethoven's 5th Symphony?", "small", "factual"),
    ("What is the currency of the United Kingdom?", "small", "factual"),
    ("What year was Google founded?", "small", "factual"),

    # =========================================================
    # MATH — simple arithmetic & conversions  (30 cases — all "small")
    # =========================================================
    ("What is 2 + 2?", "small", "math_simple"),
    ("What is 15% of 200?", "small", "math_simple"),
    ("Convert 5 miles to kilometers", "small", "math_simple"),
    ("How many ounces in a pound?", "small", "math_simple"),
    ("What is 100 celsius in fahrenheit?", "small", "math_simple"),
    ("How many minutes in a day?", "small", "math_simple"),
    ("What is the square root of 144?", "small", "math_simple"),
    ("How many seconds are in an hour?", "small", "math_simple"),
    ("What is 7 times 8?", "small", "math_simple"),
    ("Convert 10 kilograms to pounds", "small", "math_simple"),
    ("What is 25 divided by 5?", "small", "math_simple"),
    ("How many feet are in a mile?", "small", "math_simple"),
    ("What is 30% of 90?", "small", "math_simple"),
    ("Convert 50 USD to EUR at 0.92 exchange rate", "small", "math_simple"),
    ("What is 2 to the power of 10?", "small", "math_simple"),
    ("How many centimeters in an inch?", "small", "math_simple"),
    ("What is 1000 divided by 8?", "small", "math_simple"),
    ("What is the area of a rectangle 5 by 3?", "small", "math_simple"),
    ("How many weeks are in a year?", "small", "math_simple"),
    ("Convert 72 degrees Fahrenheit to Celsius", "small", "math_simple"),
    ("What is half of 250?", "small", "math_simple"),
    ("How many grams in a kilogram?", "small", "math_simple"),
    ("What is 9 squared?", "small", "math_simple"),
    ("How many liters in a gallon?", "small", "math_simple"),
    ("What is 45 + 78?", "small", "math_simple"),
    ("Convert 3 hours to minutes", "small", "math_simple"),
    ("What is 360 divided by 12?", "small", "math_simple"),
    ("How many bytes in a kilobyte?", "small", "math_simple"),
    ("What is 17 times 6?", "small", "math_simple"),
    ("What is 1/4 as a decimal?", "small", "math_simple"),

    # =========================================================
    # YES / NO  (20 cases — all "small")
    # =========================================================
    ("Is Python interpreted?", "small", "yes_no"),
    ("Is the sun a star?", "small", "yes_no"),
    ("Are whales mammals?", "small", "yes_no"),
    ("Is SQL case-sensitive?", "small", "yes_no"),
    ("Is water a compound?", "small", "yes_no"),
    ("Is JavaScript single-threaded?", "small", "yes_no"),
    ("Is the Great Wall of China visible from space?", "small", "yes_no"),
    ("Are diamonds made of carbon?", "small", "yes_no"),
    ("Is Antarctica a continent?", "small", "yes_no"),
    ("Is Bitcoin decentralized?", "small", "yes_no"),
    ("Is light faster than sound?", "small", "yes_no"),
    ("Is Python object-oriented?", "small", "yes_no"),
    ("Is the earth round?", "small", "yes_no"),
    ("Does HTML stand for HyperText Markup Language?", "small", "yes_no"),
    ("Is Mars smaller than Earth?", "small", "yes_no"),
    ("Can humans survive without oxygen?", "small", "yes_no"),
    ("Is Go a statically typed language?", "small", "yes_no"),
    ("Is the Pacific Ocean bigger than the Atlantic?", "small", "yes_no"),
    ("Is TypeScript a superset of JavaScript?", "small", "yes_no"),
    ("Is caffeine a stimulant?", "small", "yes_no"),

    # =========================================================
    # DEFINITIONS  (30 cases — all "small")
    # =========================================================
    ("What is a noun?", "small", "definitions"),
    ("Define empathy", "small", "definitions"),
    ("What does DNS stand for?", "small", "definitions"),
    ("What is an API?", "small", "definitions"),
    ("What is HTTP?", "small", "definitions"),
    ("Define photosynthesis", "small", "definitions"),
    ("What is entropy?", "small", "definitions"),
    ("What is machine learning?", "small", "definitions"),
    ("Define recursion", "small", "definitions"),
    ("What is a variable in programming?", "small", "definitions"),
    ("What is a data structure?", "small", "definitions"),
    ("Define latency", "small", "definitions"),
    ("What is a compiler?", "small", "definitions"),
    ("What does REST stand for?", "small", "definitions"),
    ("What is a primary key in a database?", "small", "definitions"),
    ("Define bandwidth", "small", "definitions"),
    ("What is an algorithm?", "small", "definitions"),
    ("What is a function in programming?", "small", "definitions"),
    ("Define abstraction in computer science", "small", "definitions"),
    ("What is a hash map?", "small", "definitions"),
    ("What does IDE stand for?", "small", "definitions"),
    ("What is a neural network?", "small", "definitions"),
    ("Define polymorphism", "small", "definitions"),
    ("What is TCP/IP?", "small", "definitions"),
    ("What is blockchain?", "small", "definitions"),
    ("Define open source", "small", "definitions"),
    ("What is a race condition?", "small", "definitions"),
    ("What is Agile methodology?", "small", "definitions"),
    ("Define gradient descent", "small", "definitions"),
    ("What is a container in software?", "small", "definitions"),

    # =========================================================
    # TRIVIA  (20 cases — all "small")
    # =========================================================
    ("What color is a giraffe's tongue?", "small", "trivia"),
    ("How many strings does a standard guitar have?", "small", "trivia"),
    ("What is the most spoken language in the world?", "small", "trivia"),
    ("Which planet has the most moons?", "small", "trivia"),
    ("What is the symbol for the Olympic Games?", "small", "trivia"),
    ("How many colors are in a rainbow?", "small", "trivia"),
    ("What sport is played at Wimbledon?", "small", "trivia"),
    ("Who is Mickey Mouse's creator?", "small", "trivia"),
    ("What is the most abundant gas in Earth's atmosphere?", "small", "trivia"),
    ("What country invented pizza?", "small", "trivia"),
    ("How many players are on a soccer team?", "small", "trivia"),
    ("What is the largest continent?", "small", "trivia"),
    ("What is the capital of New Zealand?", "small", "trivia"),
    ("Which insect has the most legs?", "small", "trivia"),
    ("What is the most watched sport in the world?", "small", "trivia"),
    ("How long is a marathon in miles?", "small", "trivia"),
    ("What country has the most natural lakes?", "small", "trivia"),
    ("What is the rarest blood type?", "small", "trivia"),
    ("How many keys does a standard piano have?", "small", "trivia"),
    ("What language has the most native speakers?", "small", "trivia"),

    # =========================================================
    # CONVERSIONS  (20 cases — all "small")
    # =========================================================
    ("Convert 100 meters to feet", "small", "conversions"),
    ("How many tablespoons in a cup?", "small", "conversions"),
    ("What is 1 megabyte in bytes?", "small", "conversions"),
    ("Convert 90 km/h to mph", "small", "conversions"),
    ("How many milliliters in a cup?", "small", "conversions"),
    ("Convert 5 feet 10 inches to centimeters", "small", "conversions"),
    ("What is 1 light-year in kilometers?", "small", "conversions"),
    ("How many teaspoons in a tablespoon?", "small", "conversions"),
    ("Convert 3.5 hours to seconds", "small", "conversions"),
    ("How many pints in a gallon?", "small", "conversions"),
    ("What is 500 grams in ounces?", "small", "conversions"),
    ("Convert 1 acre to square meters", "small", "conversions"),
    ("How many megabytes in a gigabyte?", "small", "conversions"),
    ("What is 45 pounds in kilograms?", "small", "conversions"),
    ("Convert 20 minutes to seconds", "small", "conversions"),
    ("How many cups in a liter?", "small", "conversions"),
    ("What is 1 nautical mile in kilometers?", "small", "conversions"),
    ("Convert 250 milliliters to fluid ounces", "small", "conversions"),
    ("How many milligrams in a gram?", "small", "conversions"),
    ("What is 1 horsepower in watts?", "small", "conversions"),

    # =========================================================
    # SHORT CREATIVE  (20 cases — all "small")
    # =========================================================
    ("Tell me a joke", "small", "short_creative"),
    ("Give me a pun about cats", "small", "short_creative"),
    ("Say something nice", "small", "short_creative"),
    ("Give me a motivational quote", "small", "short_creative"),
    ("Tell me a fun fact", "small", "short_creative"),
    ("Give me a word of encouragement", "small", "short_creative"),
    ("Tell me a riddle", "small", "short_creative"),
    ("Write a haiku about spring", "small", "short_creative"),
    ("Give me a rhyme about the ocean", "small", "short_creative"),
    ("Write a one-sentence story about a robot", "small", "short_creative"),
    ("Tell me a knock-knock joke", "small", "short_creative"),
    ("Give me an icebreaker question", "small", "short_creative"),
    ("Write a fun tagline for a coffee shop", "small", "short_creative"),
    ("Give me a compliment I can share with a friend", "small", "short_creative"),
    ("Tell me a clever limerick", "small", "short_creative"),
    ("Give me a random interesting word", "small", "short_creative"),
    ("Write a two-line poem about Monday", "small", "short_creative"),
    ("Give me a toast for a birthday party", "small", "short_creative"),
    ("Make up a silly team name", "small", "short_creative"),
    ("Give me a fun prompt for a journal entry", "small", "short_creative"),

    # =========================================================
    # PROOFS  (20 cases — all "big")
    # =========================================================
    ("Prove that the square root of 2 is irrational.", "big", "proofs"),
    ("Prove that there are infinitely many prime numbers.", "big", "proofs"),
    ("Prove Bayes' theorem from first principles.", "big", "proofs"),
    ("Derive the quadratic formula from ax^2 + bx + c = 0.", "big", "proofs"),
    ("Show that the sum of two continuous functions is continuous.", "big", "proofs"),
    ("Prove that every bounded sequence has a convergent subsequence (Bolzano-Weierstrass).", "big", "proofs"),
    ("Show that e is irrational.", "big", "proofs"),
    ("Prove the Cauchy-Schwarz inequality.", "big", "proofs"),
    ("Derive the formula for the sum of a geometric series.", "big", "proofs"),
    ("Prove that the halting problem is undecidable using a diagonalization argument.", "big", "proofs"),
    ("Show that P(A and B) = P(A) * P(B) only when A and B are independent. Prove it.", "big", "proofs"),
    ("Derive the closed-form solution for linear regression using the normal equations.", "big", "proofs"),
    ("Prove that a graph is bipartite if and only if it contains no odd-length cycle.", "big", "proofs"),
    ("Show that log(n!) = Theta(n log n) using Stirling's approximation.", "big", "proofs"),
    ("Prove the triangle inequality for Euclidean distance.", "big", "proofs"),
    ("Derive the backpropagation equations for a two-layer neural network.", "big", "proofs"),
    ("Prove that the eigenvalues of a symmetric matrix are always real.", "big", "proofs"),
    ("Show that a continuous function on a closed interval attains its maximum (extreme value theorem).", "big", "proofs"),
    ("Prove that NP-complete problems cannot be solved in polynomial time unless P=NP.", "big", "proofs"),
    ("Derive the gradient of the cross-entropy loss with respect to the softmax logits.", "big", "proofs"),

    # =========================================================
    # SYSTEM DESIGN  (30 cases — all "big")
    # =========================================================
    ("Design a URL shortener that handles 100 million requests per day.", "big", "system_design"),
    ("How would you architect a real-time chat app for 10 million concurrent users?", "big", "system_design"),
    ("Design a ride-sharing backend like Uber. Cover matching, pricing, and fault tolerance.", "big", "system_design"),
    ("Design a distributed rate limiter across multiple data centers.", "big", "system_design"),
    ("How would you build a search engine from scratch? Walk through the full architecture.", "big", "system_design"),
    ("Design a recommendation system for a music streaming service.", "big", "system_design"),
    ("How do you design a globally distributed key-value store with strong consistency?", "big", "system_design"),
    ("Design a payment processing system that handles 1M transactions per day with ACID guarantees.", "big", "system_design"),
    ("How would you architect a video streaming platform like YouTube?", "big", "system_design"),
    ("Design a notification service that sends 100M push notifications per day across iOS, Android, and email.", "big", "system_design"),
    ("How would you build an event-driven microservices system for an e-commerce platform?", "big", "system_design"),
    ("Design a multi-tenant SaaS platform where data isolation, billing, and feature flagging are core requirements.", "big", "system_design"),
    ("How would you design a CI/CD pipeline for a team of 200 engineers?", "big", "system_design"),
    ("Design a distributed job scheduler that guarantees at-least-once execution.", "big", "system_design"),
    ("How would you architect a healthcare data platform that must comply with HIPAA?", "big", "system_design"),
    ("Design a caching layer for a social media feed that minimizes stale data.", "big", "system_design"),
    ("How would you build a feature store for a machine learning platform?", "big", "system_design"),
    ("Design a fraud detection system for a fintech company processing real-time transactions.", "big", "system_design"),
    ("How would you architect the data pipeline for a real-time analytics dashboard?", "big", "system_design"),
    ("Design a service mesh for microservices. What are the trade-offs of using Istio vs Linkerd?", "big", "system_design"),
    ("How would you design a multi-region active-active database setup?", "big", "system_design"),
    ("Design an online multiplayer game server that handles state synchronization.", "big", "system_design"),
    ("How would you build a content delivery network (CDN) from scratch?", "big", "system_design"),
    ("Design a document collaboration system like Google Docs with real-time conflict resolution.", "big", "system_design"),
    ("How would you architect a log aggregation and monitoring platform for 10,000 microservices?", "big", "system_design"),
    ("Design a machine learning model serving system with A/B testing and canary deployments.", "big", "system_design"),
    ("How would you build a graph database from the ground up and handle traversal at scale?", "big", "system_design"),
    ("Design a hotel booking platform. Cover inventory management, pricing, and double-booking prevention.", "big", "system_design"),
    ("How would you architect a blockchain-based supply chain tracking system?", "big", "system_design"),
    ("Design a scalable IoT platform that ingests telemetry from 50 million devices.", "big", "system_design"),

    # =========================================================
    # MULTI-STEP CODE  (30 cases — all "big")
    # =========================================================
    ("Write a Python function that detects cycles in a directed graph and explain the algorithm.", "big", "multi_step_code"),
    ("Implement and explain the A* pathfinding algorithm with an admissible heuristic in Python.", "big", "multi_step_code"),
    ("Design a thread-safe LRU cache in Python and discuss the trade-offs.", "big", "multi_step_code"),
    ("My Flask app crashes under load with 'too many open files'. Walk me through diagnosing and fixing it.", "big", "multi_step_code"),
    ("Write a concurrent web scraper in Python using asyncio and explain how you handle rate limiting.", "big", "multi_step_code"),
    ("Implement a trie data structure in Python with insert, search, and prefix matching operations.", "big", "multi_step_code"),
    ("Write a merge sort implementation and then analyze its time and space complexity.", "big", "multi_step_code"),
    ("Implement Dijkstra's shortest path algorithm and explain why it fails for negative weights.", "big", "multi_step_code"),
    ("Write a Python context manager that automatically retries failed HTTP calls with exponential backoff.", "big", "multi_step_code"),
    ("Implement a pub-sub messaging system in Python without using any external libraries.", "big", "multi_step_code"),
    ("Write a recursive descent parser for simple arithmetic expressions. Explain the grammar.", "big", "multi_step_code"),
    ("Implement a bloom filter in Python. Explain the false positive rate formula.", "big", "multi_step_code"),
    ("Write a consistent hashing implementation for a distributed cache and explain the ring topology.", "big", "multi_step_code"),
    ("Implement an in-memory SQL-like query engine that supports SELECT, WHERE, and JOIN.", "big", "multi_step_code"),
    ("Write a rate limiter using the token bucket algorithm and a sliding window log. Compare them.", "big", "multi_step_code"),
    ("Implement a distributed lock using Redis. Discuss the Redlock algorithm and its critiques.", "big", "multi_step_code"),
    ("Write a Python decorator that enforces type checking at runtime. Handle generics.", "big", "multi_step_code"),
    ("Implement a minimax algorithm with alpha-beta pruning for a tic-tac-toe AI.", "big", "multi_step_code"),
    ("Write a simple LSTM cell forward pass in NumPy. Explain each gate mathematically.", "big", "multi_step_code"),
    ("Implement k-means clustering from scratch and explain the convergence guarantee.", "big", "multi_step_code"),
    ("Write a Python script that tail-calls a file in real-time, parse structured logs, and emit alerts.", "big", "multi_step_code"),
    ("Implement an event loop in pure Python using selectors. Explain how it differs from threads.", "big", "multi_step_code"),
    ("Write a generic binary search tree with insert, delete, and in-order traversal. Handle edge cases.", "big", "multi_step_code"),
    ("Implement a simple interpreter for a subset of Python (variables, arithmetic, if/else). Explain the pipeline.", "big", "multi_step_code"),
    ("Write a multi-producer multi-consumer queue in Python and explain backpressure.", "big", "multi_step_code"),
    ("Implement a word2vec skip-gram training loop in PyTorch and explain negative sampling.", "big", "multi_step_code"),
    ("Write a circuit breaker pattern in Python. Explain the state machine (closed, open, half-open).", "big", "multi_step_code"),
    ("Implement a leader election algorithm using ZooKeeper-style ephemeral nodes in Python.", "big", "multi_step_code"),
    ("Write a database connection pool from scratch in Python. Handle connection timeouts and stale connections.", "big", "multi_step_code"),
    ("Implement a JSON schema validator. Explain the recursive structure and how you handle $ref.", "big", "multi_step_code"),

    # =========================================================
    # ANALYSIS  (30 cases — all "big")
    # =========================================================
    ("Analyze the economic causes of the 2008 global financial crisis.", "big", "analysis"),
    ("Explain how transformer self-attention works and why it scales better than RNNs.", "big", "analysis"),
    ("Why does gradient descent work, and under what conditions does it fail to converge?", "big", "analysis"),
    ("Compare the CAP theorem trade-offs in Cassandra, MongoDB, and Spanner.", "big", "analysis"),
    ("Analyze the pros and cons of microservices vs monoliths for a startup at various growth stages.", "big", "analysis"),
    ("Explain why 0.1 + 0.2 != 0.3 in Python. Walk through IEEE 754 floating-point representation.", "big", "analysis"),
    ("Analyze the societal impact of large language models on knowledge work over the next decade.", "big", "analysis"),
    ("Explain the differences between eventual, causal, and strong consistency in distributed systems.", "big", "analysis"),
    ("What are the root causes of the replication crisis in psychology, and how is the field responding?", "big", "analysis"),
    ("Analyze how TLS 1.3 improves on TLS 1.2 in terms of handshake latency and security.", "big", "analysis"),
    ("Explain how backpropagation through time differs from standard backprop and why it causes vanishing gradients.", "big", "analysis"),
    ("Analyze the trade-offs between batch normalization, layer normalization, and group normalization.", "big", "analysis"),
    ("Why did the Roman Empire fall? Analyze the military, economic, and political factors.", "big", "analysis"),
    ("Explain how Raft consensus achieves leader election and log replication. What are its failure modes?", "big", "analysis"),
    ("Analyze the architectural differences between GPT and BERT and explain when to use each.", "big", "analysis"),
    ("Explain how modern JavaScript engines optimize hot code paths (JIT compilation, hidden classes, etc.).", "big", "analysis"),
    ("Analyze the trade-offs of on-device ML inference versus cloud inference for mobile applications.", "big", "analysis"),
    ("Explain how Linux kernel scheduling works with CFS (Completely Fair Scheduler). What are its weaknesses?", "big", "analysis"),
    ("Analyze the effectiveness of affirmative action policies in higher education admissions.", "big", "analysis"),
    ("Explain how memory allocators like jemalloc and tcmalloc improve on glibc malloc.", "big", "analysis"),
    ("Analyze the trade-offs of type systems: structural vs nominal typing, and duck typing.", "big", "analysis"),
    ("Explain how garbage collection works in Go vs JVM vs CPython. Compare stop-the-world pause times.", "big", "analysis"),
    ("Analyze the strategic risks of vendor lock-in when using AWS-managed services.", "big", "analysis"),
    ("Explain how column-oriented storage (Parquet, Redshift) improves analytical query performance.", "big", "analysis"),
    ("Analyze the ethical implications of surveillance capitalism and ad-targeting algorithms.", "big", "analysis"),
    ("Explain how HTTPS works end-to-end: DNS resolution, TCP, TLS handshake, and HTTP/2.", "big", "analysis"),
    ("Analyze the causes and effects of hyperinflation in Weimar Germany.", "big", "analysis"),
    ("Explain how attention mechanisms in vision transformers (ViT) differ from convolutional approaches.", "big", "analysis"),
    ("Analyze the game-theoretic dynamics of nuclear deterrence between rival states.", "big", "analysis"),
    ("Explain how Kubernetes schedules pods, handles preemption, and balances resource quotas.", "big", "analysis"),

    # =========================================================
    # RESEARCH SYNTHESIS  (20 cases — all "big")
    # =========================================================
    ("Summarize the key differences between diffusion models and GANs for image generation, including training stability.", "big", "research_synthesis"),
    ("What are the main open problems in AI alignment research for large language models?", "big", "research_synthesis"),
    ("Compare on-policy vs off-policy reinforcement learning algorithms and their sample efficiency.", "big", "research_synthesis"),
    ("Walk me through the evolution of CNN architectures from LeNet to ConvNeXt.", "big", "research_synthesis"),
    ("Synthesize the current state of quantum computing and its realistic near-term applications.", "big", "research_synthesis"),
    ("What does the literature say about the effectiveness of spaced repetition for long-term memory?", "big", "research_synthesis"),
    ("Summarize the key findings from recent research on transformer scaling laws.", "big", "research_synthesis"),
    ("What are the main approaches to continual learning and how do they address catastrophic forgetting?", "big", "research_synthesis"),
    ("Synthesize recent advances in retrieval-augmented generation (RAG) and their impact on hallucination.", "big", "research_synthesis"),
    ("What does the research say about the effectiveness of code review processes in catching bugs?", "big", "research_synthesis"),
    ("Summarize the history and current state of reinforcement learning from human feedback (RLHF).", "big", "research_synthesis"),
    ("What are the leading theories for why large language models appear to reason?", "big", "research_synthesis"),
    ("Synthesize the evidence for and against universal basic income from economic research.", "big", "research_synthesis"),
    ("What does the research say about the most effective interventions for reducing recidivism?", "big", "research_synthesis"),
    ("Summarize the key architectural innovations in protein structure prediction (AlphaFold onwards).", "big", "research_synthesis"),
    ("What are the main competing hypotheses for the origin of consciousness in neuroscience?", "big", "research_synthesis"),
    ("Synthesize the current evidence on whether remote work increases or decreases productivity.", "big", "research_synthesis"),
    ("What does recent ML safety research say about emergent capabilities in large models?", "big", "research_synthesis"),
    ("Summarize the trade-offs between federated learning and centralized training for privacy-sensitive data.", "big", "research_synthesis"),
    ("What are the state-of-the-art techniques for few-shot learning and when do they fail?", "big", "research_synthesis"),

    # =========================================================
    # COMPOUND QUESTIONS  (30 cases — all "big")
    # =========================================================
    ("Compare REST, GraphQL, and gRPC. When should I use each, and what are the operational trade-offs?", "big", "compound_questions"),
    ("Should I use Postgres or DynamoDB for a multi-tenant SaaS? Walk me through the decision.", "big", "compound_questions"),
    ("What are the differences between Kafka, RabbitMQ, and AWS SQS? When should I pick each?", "big", "compound_questions"),
    ("How does Python handle memory management, and what are the implications for multi-threaded code?", "big", "compound_questions"),
    ("Compare Rust, Go, and C++ for writing a high-performance database engine. What matters most?", "big", "compound_questions"),
    ("What are the subtle differences between `__str__`, `__repr__`, and `__format__` in Python, and when is each called?", "big", "compound_questions"),
    ("How do JavaScript promises, async/await, and the event loop interact? Give an example that reveals a common pitfall.", "big", "compound_questions"),
    ("Compare supervised, unsupervised, and self-supervised learning. When is each appropriate, and what are the label cost trade-offs?", "big", "compound_questions"),
    ("What is the difference between L1 and L2 regularization? How do they affect model sparsity and weight distributions?", "big", "compound_questions"),
    ("Compare monorepo vs polyrepo strategies for a 50-engineer team. Address versioning, CI speed, and ownership.", "big", "compound_questions"),
    ("What are the differences between synchronous and asynchronous programming models, and when does each hurt performance?", "big", "compound_questions"),
    ("Compare Docker and Kubernetes: what does each solve, and at what scale does Kubernetes become necessary?", "big", "compound_questions"),
    ("How does SQL query planning work? Explain joins, indexes, and the role of statistics in query optimization.", "big", "compound_questions"),
    ("What are the trade-offs between embedding-based search and keyword search (BM25)? When should I use each?", "big", "compound_questions"),
    ("Compare Redis, Memcached, and Hazelcast. What are the right workloads for each?", "big", "compound_questions"),
    ("How do BERT and GPT differ architecturally and in their training objectives? What does this mean for downstream tasks?", "big", "compound_questions"),
    ("What is the difference between precision and recall? When should I optimize for one over the other?", "big", "compound_questions"),
    ("Compare TCP and UDP. In which scenarios does UDP outperform TCP despite being unreliable?", "big", "compound_questions"),
    ("How do optimistic and pessimistic concurrency control differ, and what are the implications for database contention?", "big", "compound_questions"),
    ("Compare CNNs and Vision Transformers for image classification. What are the data efficiency trade-offs?", "big", "compound_questions"),
    ("What are the differences between horizontal and vertical scaling, and how do you decide which to apply first?", "big", "compound_questions"),
    ("Compare fine-tuning, LoRA, and prompt engineering for adapting large language models. When is each best?", "big", "compound_questions"),
    ("How does React's reconciliation algorithm work, and how does it differ from Vue's virtual DOM approach?", "big", "compound_questions"),
    ("What are the differences between OLTP and OLAP databases? Can a single system serve both workloads well?", "big", "compound_questions"),
    ("Compare mutex, semaphore, and monitor for synchronization. When does each lead to deadlock?", "big", "compound_questions"),
    ("How do convolutional and recurrent architectures handle sequential data differently? What has replaced them?", "big", "compound_questions"),
    ("Compare LSM trees and B-trees for storage engines. How does each optimize for reads vs writes?", "big", "compound_questions"),
    ("What are the differences between strong typing and dynamic typing at runtime? How does each affect bug discovery?", "big", "compound_questions"),
    ("Compare the ACID properties in relational databases to the BASE properties in NoSQL systems.", "big", "compound_questions"),
    ("How does gradient clipping differ from gradient normalization? When does each help training stability?", "big", "compound_questions"),

    # =========================================================
    # AMBIGUOUS / BORDERLINE  (30 cases — mixed)
    # =========================================================
    # These are intentionally tricky — some look simple but are complex, or vice versa.
    ("What is the meaning of life?", "big", "ambiguous"),
    ("Why is the sky blue?", "big", "ambiguous"),
    ("How does WiFi work?", "big", "ambiguous"),
    ("What makes a good leader?", "big", "ambiguous"),
    ("Is AI dangerous?", "big", "ambiguous"),
    ("What is consciousness?", "big", "ambiguous"),
    ("How do vaccines work?", "big", "ambiguous"),
    ("What causes inflation?", "big", "ambiguous"),
    ("How do I get better at coding?", "big", "ambiguous"),
    ("What is love?", "small", "ambiguous"),
    ("What time is it?", "small", "ambiguous"),
    ("What's 2 * 3?", "small", "ambiguous"),
    ("Name a color", "small", "ambiguous"),
    ("What's a good movie to watch?", "small", "ambiguous"),
    ("How are you?", "small", "ambiguous"),
    ("Give me an example of a loop in Python", "small", "ambiguous"),
    ("What is 1 + 1?", "small", "ambiguous"),
    ("What's the weather like?", "small", "ambiguous"),
    ("Who is Einstein?", "small", "ambiguous"),
    ("What's a synonym for big?", "small", "ambiguous"),
    ("How do neural networks learn?", "big", "ambiguous"),
    ("What is the best programming language?", "big", "ambiguous"),
    ("Why should I use Docker?", "big", "ambiguous"),
    ("How does the stock market work?", "big", "ambiguous"),
    ("What is a good salary in San Francisco?", "big", "ambiguous"),
    ("How do I start investing?", "big", "ambiguous"),
    ("What should I learn first: Python or JavaScript?", "big", "ambiguous"),
    ("Is remote work better than in-office?", "big", "ambiguous"),
    ("What is the best diet?", "big", "ambiguous"),
    ("How does sleep affect memory?", "big", "ambiguous"),

    # =========================================================
    # REAL-WORLD PLANNING  (20 cases — all "big")
    # =========================================================
    ("I need to reduce my AWS bill by 40% while maintaining 99.9% uptime. What's the strategy?", "big", "real_world_planning"),
    ("Plan a migration from a monolith to microservices for a 10-person team over 6 months.", "big", "real_world_planning"),
    ("I want to launch a SaaS product in 3 months. What's the MVP scope and tech stack for a solo founder?", "big", "real_world_planning"),
    ("How should I set up a data infrastructure for a startup going from 0 to 1M users?", "big", "real_world_planning"),
    ("I'm building a dental clinic management app. Outline the domain model, stack, and riskiest parts.", "big", "real_world_planning"),
    ("Plan a zero-downtime database migration from MySQL to PostgreSQL for a live production system.", "big", "real_world_planning"),
    ("How should I structure my engineering team as we grow from 5 to 50 engineers?", "big", "real_world_planning"),
    ("I want to productionize a machine learning model that retrains daily on 1TB of data. Outline the pipeline.", "big", "real_world_planning"),
    ("How do I build a disaster recovery plan for a critical financial services backend?", "big", "real_world_planning"),
    ("Plan a content strategy for a technical blog targeting ML engineers. Include topics, cadence, and distribution.", "big", "real_world_planning"),
    ("I need to onboard 20 junior engineers in 3 months. Design the onboarding program.", "big", "real_world_planning"),
    ("How do I migrate a legacy PHP app to a modern Python/FastAPI stack without breaking production?", "big", "real_world_planning"),
    ("Design a rollout plan for a new pricing model that won't churn existing customers.", "big", "real_world_planning"),
    ("I'm preparing for a Series A fundraise. What technical due diligence should I expect?", "big", "real_world_planning"),
    ("How should I structure an API versioning strategy for a public API with thousands of clients?", "big", "real_world_planning"),
    ("I want to build an open-source project that gains traction. What's the strategy for the first 6 months?", "big", "real_world_planning"),
    ("Plan a cost-effective multi-region deployment for a global SaaS product.", "big", "real_world_planning"),
    ("How do I evaluate and hire a VP of Engineering for a 30-person startup?", "big", "real_world_planning"),
    ("I want to enter the enterprise market. What changes do I need to make to my product and GTM?", "big", "real_world_planning"),
    ("How should I set up observability (logging, metrics, tracing) for a new microservices platform from day one?", "big", "real_world_planning"),

    # =========================================================
    # CAREER / LEARNING ADVICE  (20 cases — all "big")
    # =========================================================
    ("How should I structure my learning path to become a machine learning engineer in 12 months?", "big", "career_learning"),
    ("What skills should I prioritize as a backend engineer wanting to move into ML infrastructure?", "big", "career_learning"),
    ("How do I negotiate a higher salary at a big tech company?", "big", "career_learning"),
    ("I'm a self-taught programmer. How do I land my first full-time software engineering job?", "big", "career_learning"),
    ("What are the most important concepts a CS graduate should master before their first job?", "big", "career_learning"),
    ("How do I transition from software engineering to engineering management?", "big", "career_learning"),
    ("What should I include in my ML engineer portfolio to impress top companies?", "big", "career_learning"),
    ("How do I get good at system design interviews in 8 weeks?", "big", "career_learning"),
    ("I want to publish my first machine learning research paper. What's the process from idea to submission?", "big", "career_learning"),
    ("How do I decide between an MBA and a master's in CS for a technical product management career?", "big", "career_learning"),
    ("What's the best way to learn deep learning if I already know Python but am new to ML?", "big", "career_learning"),
    ("How do I build credibility as an engineer at a new company in the first 90 days?", "big", "career_learning"),
    ("I'm getting a PhD in ML. Should I do an internship at a company or focus on research? What are the trade-offs?", "big", "career_learning"),
    ("How should I approach open-source contributions to advance my career?", "big", "career_learning"),
    ("What are the key differences between working at a startup versus a FAANG company for an early-career engineer?", "big", "career_learning"),
    ("How do I build a strong professional network as a software engineer?", "big", "career_learning"),
    ("What's the best way to stay current with rapidly evolving ML research without burning out?", "big", "career_learning"),
    ("I want to move from academia to industry ML. What should I know about the differences?", "big", "career_learning"),
    ("How do I identify and close skill gaps as a mid-level engineer aiming for senior promotion?", "big", "career_learning"),
    ("What mentorship strategies work best for accelerating growth as a junior engineer?", "big", "career_learning"),

    # =========================================================
    # MULTI-CONSTRAINT  (20 cases — all "big")
    # =========================================================
    ("Build a REST API in Python that must: handle 10k RPS, authenticate via JWT, rate-limit per user, and log every request to an async queue.", "big", "multi_constraint"),
    ("Design a caching strategy that minimizes cost, maximizes hit rate, keeps data fresh within 60 seconds, and gracefully handles cache failures.", "big", "multi_constraint"),
    ("I need a database schema for a social network that supports millions of users, efficient feed generation, privacy controls, and soft deletion.", "big", "multi_constraint"),
    ("Write a training loop in PyTorch that: uses mixed precision, supports gradient accumulation, checkpoints every epoch, and resumes from checkpoints.", "big", "multi_constraint"),
    ("Design a CI/CD pipeline that: runs tests in parallel, deploys to staging automatically, requires manual approval for production, and rolls back on failure.", "big", "multi_constraint"),
    ("I need a search feature that: returns results in under 100ms, supports typo tolerance, ranks by relevance + recency, and handles 1M documents.", "big", "multi_constraint"),
    ("Build a multi-tenant authentication system that supports SSO, MFA, per-tenant password policies, and audit logging.", "big", "multi_constraint"),
    ("Design a data pipeline that: ingests from 5 sources, deduplicates, validates schema, handles late-arriving data, and writes to a data warehouse.", "big", "multi_constraint"),
    ("Create a microservice that must: be stateless, horizontally scalable, idempotent, and retry-safe for all external API calls.", "big", "multi_constraint"),
    ("Design an ML inference system that: serves models in under 50ms p99, supports hot-swapping models without downtime, and tracks prediction drift.", "big", "multi_constraint"),
    ("Build a reporting system that: generates PDFs, allows custom date ranges, handles large datasets without memory issues, and emails reports asynchronously.", "big", "multi_constraint"),
    ("Design a storage system for user uploads that: encrypts data at rest, limits upload size, generates thumbnails, and CDN-serves content globally.", "big", "multi_constraint"),
    ("Write a distributed task queue that: guarantees at-least-once delivery, supports priorities, handles worker crashes, and exposes metrics.", "big", "multi_constraint"),
    ("Design a logging system that: captures structured logs, supports full-text search, retains data for 90 days, and alerts on error patterns.", "big", "multi_constraint"),
    ("Build an A/B testing framework that: assigns users consistently, splits traffic by percentage, tracks conversion, and is statistically sound.", "big", "multi_constraint"),
    ("Design a geospatial service that: finds nearby points within a radius, handles millions of locations, updates in real time, and serves in under 20ms.", "big", "multi_constraint"),
    ("Create a recommendation engine that: personalizes per user, avoids filter bubbles, explains recommendations, and respects user opt-outs.", "big", "multi_constraint"),
    ("Design a webhook delivery system that: retries on failure, prevents replay attacks, verifies signatures, and tracks delivery status.", "big", "multi_constraint"),
    ("Build a data anonymization pipeline that: removes PII, maintains statistical utility, is auditable, and handles structured and unstructured data.", "big", "multi_constraint"),
    ("Design a multi-cloud failover strategy that: achieves RPO < 1 min, RTO < 5 min, uses active-active topology, and minimizes egress costs.", "big", "multi_constraint"),

    # =========================================================
    # DOMAIN-SPECIFIC DEEP  (30 cases — all "big")
    # =========================================================
    ("Explain the role of attention masks in causal vs bidirectional language models and why each is appropriate for its task.", "big", "domain_specific_deep"),
    ("Why do we use layer normalization in transformers instead of batch normalization? What are the mathematical reasons?", "big", "domain_specific_deep"),
    ("Explain how Raft consensus achieves leader election and log replication. What happens during a network partition?", "big", "domain_specific_deep"),
    ("How does LoRA reduce the number of trainable parameters in a large language model? Derive the rank decomposition.", "big", "domain_specific_deep"),
    ("Explain the MVCC (multi-version concurrency control) mechanism in PostgreSQL and how it handles read-write conflicts.", "big", "domain_specific_deep"),
    ("How does the Linux virtual memory system work? Explain page tables, TLBs, and demand paging.", "big", "domain_specific_deep"),
    ("Explain the mathematics behind principal component analysis (PCA) and its relationship to SVD.", "big", "domain_specific_deep"),
    ("How does QUIC improve on TCP+TLS? Walk through the connection establishment and 0-RTT feature.", "big", "domain_specific_deep"),
    ("Explain the insider threat model for Byzantine fault tolerance and why Raft does not handle it.", "big", "domain_specific_deep"),
    ("How does the Adam optimizer work? Derive the update rule and explain the bias correction terms.", "big", "domain_specific_deep"),
    ("Explain speculative execution in modern CPUs and how Spectre-class vulnerabilities exploit it.", "big", "domain_specific_deep"),
    ("How does the Transformer's positional encoding using sinusoidal functions allow generalization to unseen lengths?", "big", "domain_specific_deep"),
    ("Explain write-ahead logging (WAL) in databases. How does it enable crash recovery and replication?", "big", "domain_specific_deep"),
    ("How does RLHF (Reinforcement Learning from Human Feedback) work mathematically? Explain the reward model and PPO update.", "big", "domain_specific_deep"),
    ("Explain how convolutional neural networks achieve translation invariance and what is lost when you remove pooling layers.", "big", "domain_specific_deep"),
    ("How does Kafka achieve exactly-once semantics with idempotent producers and transactional APIs?", "big", "domain_specific_deep"),
    ("Explain the Bellman equation in reinforcement learning and derive the Q-learning update rule.", "big", "domain_specific_deep"),
    ("How does LLVM's intermediate representation enable cross-language optimization? Explain SSA form.", "big", "domain_specific_deep"),
    ("Explain how kernel methods (SVM with RBF kernel) implicitly map to infinite-dimensional feature spaces.", "big", "domain_specific_deep"),
    ("How does the Python GIL affect multi-threaded CPU-bound workloads, and what are the idiomatic workarounds?", "big", "domain_specific_deep"),
    ("Explain variational autoencoders: derive the ELBO, the reparameterization trick, and why KL divergence matters.", "big", "domain_specific_deep"),
    ("How does ZFS achieve copy-on-write, checksumming, and self-healing? What are the performance trade-offs?", "big", "domain_specific_deep"),
    ("Explain the mathematics of contrastive learning (SimCLR) and why the InfoNCE loss works.", "big", "domain_specific_deep"),
    ("How does eBPF work in the Linux kernel? Explain the verifier, JIT compilation, and safe execution model.", "big", "domain_specific_deep"),
    ("Explain mixture of experts (MoE) models: how does routing work, and what are the training instability challenges?", "big", "domain_specific_deep"),
    ("How does the Paxos consensus algorithm guarantee safety? Walk through the two-phase protocol.", "big", "domain_specific_deep"),
    ("Explain how Rust's borrow checker prevents data races at compile time. Give an example of a rejected program and why.", "big", "domain_specific_deep"),
    ("How does flash attention reduce memory complexity from O(n^2) to O(n) during self-attention computation?", "big", "domain_specific_deep"),
    ("Explain the Chord distributed hash table: how does it achieve O(log n) lookup with finger tables?", "big", "domain_specific_deep"),
    ("How does the PageRank algorithm work mathematically? Explain the damping factor and the power iteration method.", "big", "domain_specific_deep"),
]

# Verify count
assert len(TEST_CASES) == 500, f"Expected 500 test cases, got {len(TEST_CASES)}"


# ---------------------------------------------------------------------------
# Evaluation dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    ground_truth: str
    category: str
    ml_decision: str
    ml_confidence: float
    correct: bool
    would_cascade: bool


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

CASCADE_THRESHOLD = 0.65  # matches config.ML_ROUTER_CONFIDENCE_THRESHOLD


def run_evaluation(model_path: str) -> list[QueryResult]:
    """Load the ML router and evaluate all 500 test cases."""
    from router.ml_router import MLRouter

    print(f"\nLoading ML Router from: {model_path}")
    router = MLRouter.load(model_path)
    print(f"Model loaded successfully.\n")

    results: list[QueryResult] = []
    for query, ground_truth, category in TEST_CASES:
        pred = router.predict(query)
        correct = pred.decision == ground_truth
        would_cascade = pred.confidence < CASCADE_THRESHOLD

        results.append(QueryResult(
            query=query,
            ground_truth=ground_truth,
            category=category,
            ml_decision=pred.decision,
            ml_confidence=pred.confidence,
            correct=correct,
            would_cascade=would_cascade,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _header(title: str, width: int = 72) -> str:
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}"


def _subheader(title: str, width: int = 72) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def print_report(results: list[QueryResult]) -> None:  # noqa: C901
    total = len(results)
    correct_all = sum(1 for r in results if r.correct)
    overall_accuracy = correct_all / total

    # Confusion matrix counts (positive class = "big")
    tp = sum(1 for r in results if r.ground_truth == "big" and r.ml_decision == "big")
    fp = sum(1 for r in results if r.ground_truth == "small" and r.ml_decision == "big")
    tn = sum(1 for r in results if r.ground_truth == "small" and r.ml_decision == "small")
    fn = sum(1 for r in results if r.ground_truth == "big" and r.ml_decision == "small")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Confidence distribution
    confidences = [r.ml_confidence for r in results]
    conf_mean   = statistics.mean(confidences)
    conf_median = statistics.median(confidences)
    conf_min    = min(confidences)
    conf_max    = max(confidences)
    above_thresh = sum(1 for c in confidences if c >= CASCADE_THRESHOLD)

    # Cascade stats
    cascade_count = sum(1 for r in results if r.would_cascade)
    cascade_rate  = cascade_count / total

    # Per-category breakdown
    categories: dict[str, list[QueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    # ── Print report ─────────────────────────────────────────────────────────

    print(_header("ML ROUTER EVALUATION REPORT — 500 LABELED TEST CASES"))

    print(f"\n  Total test cases : {total}")
    print(f"  Cascade threshold: {CASCADE_THRESHOLD}")

    # ── 1. Overall accuracy ──────────────────────────────────────────────────
    print(_subheader("1. OVERALL ML ROUTER ACCURACY"))
    print(f"  Correct predictions : {correct_all} / {total}")
    print(f"  Accuracy            : {overall_accuracy:.1%}")

    # ── 2. Confusion matrix ──────────────────────────────────────────────────
    print(_subheader("2. CONFUSION MATRIX  (positive class = 'big')"))
    print(f"  {'':30s} {'Predicted BIG':>14} {'Predicted SMALL':>15}")
    print(f"  {'Actual BIG':30s} {'TP =':>8} {tp:>5}   {'FN =':>8} {fn:>5}")
    print(f"  {'Actual SMALL':30s} {'FP =':>8} {fp:>5}   {'TN =':>8} {tn:>5}")
    print()
    print(f"  Precision (big class) : {precision:.3f}")
    print(f"  Recall    (big class) : {recall:.3f}")
    print(f"  F1 score  (big class) : {f1:.3f}")

    # ── 3. Confidence distribution ───────────────────────────────────────────
    print(_subheader("3. CONFIDENCE DISTRIBUTION"))
    print(f"  Mean   : {conf_mean:.3f}")
    print(f"  Median : {conf_median:.3f}")
    print(f"  Min    : {conf_min:.3f}")
    print(f"  Max    : {conf_max:.3f}")
    print(f"  Above cascade threshold ({CASCADE_THRESHOLD}): {above_thresh} / {total}  "
          f"({above_thresh/total:.1%})")

    # Histogram (10 buckets)
    print()
    bucket_width = 0.10
    print("  Confidence histogram:")
    for i in range(10):
        lo = i * bucket_width
        hi = lo + bucket_width
        count = sum(1 for c in confidences if lo <= c < hi)
        bar = "#" * count
        # Last bucket is inclusive
        if i == 9:
            count = sum(1 for c in confidences if lo <= c <= hi)
        print(f"    [{lo:.1f} – {hi:.1f}) : {count:>4}  {bar[:60]}")

    # ── 4. Cascade rate ───────────────────────────────────────────────────────
    print(_subheader("4. CASCADE RATE  (ML confidence < threshold → LLM fallback)"))
    print(f"  Would cascade       : {cascade_count} / {total}  ({cascade_rate:.1%})")
    print(f"  Handled by ML alone : {total - cascade_count} / {total}  "
          f"({(total - cascade_count)/total:.1%})")

    # ── 5. Per-category accuracy & cascade rate ───────────────────────────────
    print(_subheader("5. PER-CATEGORY BREAKDOWN"))
    cat_names = sorted(categories.keys())
    col1 = max(len(n) for n in cat_names) + 2
    header_line = (
        f"  {'Category':<{col1}} {'N':>4}  {'Acc':>7}  {'Correct':>7}  "
        f"{'CascadeRate':>11}  {'Cascades':>8}"
    )
    print(header_line)
    print("  " + "-" * (len(header_line) - 2))

    for cat in cat_names:
        cat_results = categories[cat]
        n = len(cat_results)
        n_correct = sum(1 for r in cat_results if r.correct)
        n_cascade = sum(1 for r in cat_results if r.would_cascade)
        acc = n_correct / n
        casc_rate = n_cascade / n
        print(
            f"  {cat:<{col1}} {n:>4}  {acc:>7.1%}  {n_correct:>7}  "
            f"{casc_rate:>11.1%}  {n_cascade:>8}"
        )

    # ── 6. Top 20 wrong predictions ───────────────────────────────────────────
    print(_subheader("6. TOP 20 WRONG PREDICTIONS  (highest ML confidence)"))
    wrong = [r for r in results if not r.correct]
    wrong_sorted = sorted(wrong, key=lambda r: r.ml_confidence, reverse=True)[:20]

    if not wrong_sorted:
        print("  No wrong predictions! Perfect accuracy.")
    else:
        for i, r in enumerate(wrong_sorted, 1):
            q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
            print(
                f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
                f"gt={r.ground_truth} pred={r.ml_decision}\n"
                f"      \"{q_display}\""
            )

    # ── 7. Top 20 low-confidence predictions ─────────────────────────────────
    print(_subheader("7. TOP 20 MOST UNCERTAIN PREDICTIONS  (lowest confidence)"))
    uncertain = sorted(results, key=lambda r: r.ml_confidence)[:20]

    for i, r in enumerate(uncertain, 1):
        status = "CORRECT" if r.correct else "WRONG"
        q_display = r.query[:70] + ("..." if len(r.query) > 70 else "")
        print(
            f"  {i:>2}. [{r.category}] conf={r.ml_confidence:.3f} "
            f"gt={r.ground_truth} pred={r.ml_decision}  [{status}]\n"
            f"      \"{q_display}\""
        )

    # ── 8. Summary recommendation ─────────────────────────────────────────────
    print(_subheader("8. SUMMARY & RECOMMENDATION"))

    issues: list[str] = []
    positives: list[str] = []

    if overall_accuracy >= 0.90:
        positives.append(f"Strong overall accuracy ({overall_accuracy:.1%}).")
    elif overall_accuracy >= 0.80:
        positives.append(f"Acceptable overall accuracy ({overall_accuracy:.1%}); room for improvement.")
    else:
        issues.append(f"Low overall accuracy ({overall_accuracy:.1%}) — consider retraining with more data.")

    if cascade_rate <= 0.15:
        positives.append(f"Low cascade rate ({cascade_rate:.1%}) — ML router handles most queries confidently.")
    elif cascade_rate <= 0.30:
        issues.append(
            f"Moderate cascade rate ({cascade_rate:.1%}) — roughly {cascade_count} queries would "
            f"fall back to the LLM classifier."
        )
    else:
        issues.append(
            f"High cascade rate ({cascade_rate:.1%}) — {cascade_count} queries require LLM fallback. "
            "The ML router lacks confidence on many inputs; collect more training data."
        )

    if precision < 0.85:
        issues.append(
            f"Precision for 'big' class is {precision:.1%} — too many small queries are being "
            "misrouted to the large LLM (expensive false positives)."
        )
    if recall < 0.85:
        issues.append(
            f"Recall for 'big' class is {recall:.1%} — too many complex queries are being "
            "incorrectly handled by the small LLM (quality-degrading false negatives)."
        )

    # Identify worst categories
    cat_accs = {
        cat: sum(1 for r in cat_results if r.correct) / len(cat_results)
        for cat, cat_results in categories.items()
    }
    worst_cats = sorted(cat_accs, key=cat_accs.get)[:3]  # type: ignore[arg-type]
    if cat_accs[worst_cats[0]] < 0.80:
        bad_str = ", ".join(f"{c} ({cat_accs[c]:.0%})" for c in worst_cats if cat_accs[c] < 0.80)
        if bad_str:
            issues.append(f"Weakest categories: {bad_str}. Add more examples for these to seed_data.py.")

    print()
    if positives:
        print("  Strengths:")
        for p in positives:
            print(f"    + {p}")
    if issues:
        print("\n  Issues / Action items:")
        for item in issues:
            print(f"    ! {item}")

    print("\n  Recommended next steps:")
    if overall_accuracy < 0.90:
        print("    1. Add 50-100 more labeled examples for weak categories to router/seed_data.py")
        print("    2. Retrain with: python -m router.train")
    if cascade_rate > 0.20:
        print("    3. Consider tuning ML_ROUTER_CONFIDENCE_THRESHOLD (currently 0.65).")
        print("       Lowering it reduces LLM calls but risks more ML errors.")
    if f1 < 0.85:
        print("    4. Review feature engineering in router/features.py for ambiguous categories.")
    print("    5. Collect real production queries via ClassificationLogger and retrain periodically.")

    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve model path relative to project root (two levels up from router/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "router", "models", "router_v0.joblib")

    # Allow override via CLI arg
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Train the model first with:  python -m router.train")
        sys.exit(1)

    results = run_evaluation(model_path)
    print_report(results)


if __name__ == "__main__":
    main()
