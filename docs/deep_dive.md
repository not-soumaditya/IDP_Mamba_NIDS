# Deep Dive: DDoS, Mamba Memory, Transformer vs Mamba, Temporal Patterns

---

## 1. DDoS Attacks — What They Are and Why They're Hard to Catch

### What DDoS Actually Means

**DDoS = Distributed Denial of Service**

Break it down word by word:
- **Denial of Service** — make a server unable to serve its legitimate users
- **Distributed** — the attack comes from many sources simultaneously, not just one

The core idea is brutally simple: **flood a server with so much traffic that it can't handle real requests anymore.** It's like 10,000 people trying to squeeze through a single door at the same time — nobody gets through.

### How a DDoS Attack Actually Works — Step by Step

```
BEFORE THE ATTACK (Botnet Building Phase):
═══════════════════════════════════════════

  Attacker's Computer
        │
        ├──► Infects IoT Camera in Brazil      ──► Now a "zombie"
        ├──► Infects Smart Fridge in Germany    ──► Now a "zombie"  
        ├──► Infects Old Router in India        ──► Now a "zombie"
        ├──► Infects Laptop in Nigeria          ──► Now a "zombie"
        └──► ... (thousands more)
        
  All these infected devices = "BOTNET"
  The owners don't even know their devices are compromised.


THE ATTACK:
═══════════

  Attacker sends ONE command: "Flood target 203.0.113.50 NOW"
        │
        ▼
  ┌─────────────────────────────────────────────────┐
  │  Thousands of zombies simultaneously send       │
  │  massive amounts of traffic to the target       │
  └─────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────┐
  │   Target Server   │  ← Can handle 10,000 requests/sec
  │   (e.g., hospital │  ← Receiving 5,000,000 requests/sec
  │    website)        │  ← OVERWHELMED → CRASHES
  └──────────────────┘
        │
        ▼
  Real patients trying to access the portal → "503 Service Unavailable"
```

### The Three Types of DDoS

#### Type 1: Volumetric Attacks (the "firehose")
**Goal**: Saturate the target's bandwidth.

The attacker sends so much raw data that the internet pipe to the server literally fills up. It's like trying to drink from a fire hydrant.

- **UDP Flood**: Send millions of UDP packets to random ports. The server wastes time checking each one and responding with "port unreachable" messages.
- **DNS Amplification**: Send a tiny request to a DNS server, spoofing the return address as the victim's IP. The DNS server sends a response 50-70× larger to the victim. The attacker sends 1 GB, the victim receives 50-70 GB.

**Real example**: In February 2020, AWS reported mitigating a **2.3 Tbps** (terabits per second) DDoS attack — that's about 287 gigabytes of data hitting the server every single second for hours.

#### Type 2: Protocol Attacks (the "handshake abuser")
**Goal**: Exhaust the server's connection-handling capacity.

- **SYN Flood**: Exploits the TCP three-way handshake:
  ```
  Normal connection:
    Client  ──SYN──►  Server     "Hey, want to connect?"
    Client  ◄─SYN/ACK─  Server   "Sure, here's my handshake"
    Client  ──ACK──►  Server     "Great, we're connected!"
  
  SYN Flood attack:
    Zombie₁  ──SYN──►  Server     Server allocates memory, waits for ACK...
    Zombie₂  ──SYN──►  Server     Server allocates more memory, waits...
    Zombie₃  ──SYN──►  Server     More memory...
    ... × 1,000,000
    (Nobody ever sends the final ACK)
    
    Server's connection table: FULL
    Real users trying to connect: REJECTED
  ```

**Real example**: The 2016 **Dyn DNS attack** (Mirai botnet) used ~100,000 compromised IoT devices to SYN-flood Dyn's DNS servers, taking down Twitter, Netflix, Reddit, Spotify, and GitHub simultaneously for hours.

#### Type 3: Application Layer Attacks (the "smart flood")
**Goal**: Overwhelm the application logic, not just the network pipe.

- **HTTP Flood**: Send thousands of legitimate-looking HTTP requests (GET /search?q=random_string). Each request forces the server to query a database, render a page, and send it back. The requests look completely normal individually — you can't block them with a simple rule.
- **Slowloris**: Open many connections to the server but send data veeeery slooowly, keeping each connection alive. The server waits patiently for each request to complete, until all its connection slots are occupied.

**Real example**: In 2018, **GitHub** was hit by a 1.35 Tbps memcached amplification attack — the largest DDoS ever recorded at the time. GitHub's service was intermittently unavailable for about 10 minutes before Akamai's scrubbing service kicked in.

### Why DDoS Is Hard to Stop

| Challenge | Why It's Hard |
|-----------|---------------|
| **Distributed sources** | Traffic comes from thousands of IPs. You can't just block one IP. |
| **Looks legitimate** | Each individual packet is often a valid TCP/UDP/HTTP request. No single packet screams "attack." |
| **Volume** | Even if you detect it, you may not have enough bandwidth to absorb the flood while filtering. |
| **Evolving** | Attackers constantly change patterns — slow ramp-ups, rotating source IPs, mixing attack traffic with real traffic. |

### How Mamba Helps Detect DDoS

A DDoS attack has a **temporal signature** — it doesn't appear as a single suspicious packet, it appears as a *pattern over many packets*:

```
Time →  
Normal:  ▁▁▂▁▁▂▁▁▁▂▁▁▂▁▁▁▁▂▁▁    (low, steady traffic)
DDoS:   ▁▁▂▃▄▅▆▇█████████████████  (sudden ramp-up, sustained flood)
```

Mamba reads the sequence of traffic flows one by one, updating its hidden state. As it sees the ramp-up pattern (increasing packet rate, many different source IPs, repetitive packet sizes), the hidden state encodes "this looks like a flood is building" — and it flags the attack *during the ramp-up*, before the server goes down.

A classical ML model looking at individual flows would see each one as "just another packet" and miss the pattern entirely.

---

## 2. Mamba's O(1) Memory — How Is That Even Possible?

### The Core Idea: Compression, Not Storage

To understand this, let's compare two strategies for "remembering" a sequence:

#### Strategy A: Store Everything (What Transformers Do)

```
Packet 1 arrives → Store it     → Memory: [P₁]
Packet 2 arrives → Store it     → Memory: [P₁, P₂]
Packet 3 arrives → Store it     → Memory: [P₁, P₂, P₃]
...
Packet L arrives → Store it     → Memory: [P₁, P₂, P₃, ..., Pₗ]

Memory usage: O(L) — grows with every packet
```

This is what Transformers do with their **KV cache** (Key-Value cache). To generate an output at step $L$, the Transformer needs to attend to ALL previous keys and values. So it stores every single past token's representation. Memory grows linearly with sequence length.

#### Strategy B: Compress Into a Fixed Summary (What Mamba Does)

```
Start with a state vector h = [0, 0, 0, 0]    (size: always 4 numbers)

Packet 1 arrives → Update h     → h = [0.3, 0.1, 0.7, 0.2]
Packet 2 arrives → Update h     → h = [0.5, 0.4, 0.3, 0.8]
Packet 3 arrives → Update h     → h = [0.2, 0.6, 0.9, 0.1]
...
Packet L arrives → Update h     → h = [0.7, 0.3, 0.5, 0.4]

Memory usage: O(1) — ALWAYS just 4 numbers, whether L=10 or L=10,000,000
```

### What IS the State Vector?

The state vector $h$ is a **fixed-size array of numbers** that acts as a **compressed summary** of everything the model has seen so far.

Think of it like a person's mental state while watching a security camera feed:

```
Your brain doesn't store every frame of video you've ever watched.
Instead, you maintain a MENTAL STATE:
  - "Traffic seems normal right now"           → encoded as some neurons firing
  - "There was a suspicious spike 30 seconds ago" → encoded in your short-term memory
  - "This IP has appeared 47 times in the last minute" → encoded as a feeling of "something's off"

Your brain is ~1.4 kg regardless of whether you've watched 
1 minute of footage or 10 hours. THAT's O(1) memory.
```

In Mamba, the state vector might be, say, **64 numbers** (if `d_state=16` and `d_inner=4`, you get a state matrix of shape 4×16 = 64 values). These 64 numbers encode the model's "understanding" of all the traffic it has seen.

### The Math — How the State Gets Updated

At every time step $k$, the state update equation is:

$$h_k = \bar{\mathbf{A}}_k \cdot h_{k-1} + \bar{\mathbf{B}}_k \cdot x_k$$

Let's trace through a concrete numeric example with a **tiny** state (size 4) processing 5 packets:

```
State size: 4 numbers
Input: each packet is represented by 1 feature (simplified)

───── Packet 1 (x₁ = 0.5, a normal packet) ─────
  Ā₁ = [0.9, 0.9, 0.9, 0.9]    (keep 90% of previous state)
  B̄₁ = [0.2, 0.1, 0.3, 0.1]    (how much this input affects state)
  
  h₁ = Ā₁ · h₀ + B̄₁ · x₁
     = [0.9,0.9,0.9,0.9] · [0,0,0,0] + [0.2,0.1,0.3,0.1] · 0.5
     = [0, 0, 0, 0] + [0.10, 0.05, 0.15, 0.05]
     = [0.10, 0.05, 0.15, 0.05]
     
  Memory used: just these 4 numbers ✓

───── Packet 2 (x₂ = 0.6, another normal packet) ─────
  Ā₂ = [0.9, 0.9, 0.9, 0.9]    (still keeping 90%)
  B̄₂ = [0.2, 0.1, 0.3, 0.1]
  
  h₂ = [0.9,0.9,0.9,0.9] · [0.10, 0.05, 0.15, 0.05] 
       + [0.2,0.1,0.3,0.1] · 0.6
     = [0.09, 0.045, 0.135, 0.045] + [0.12, 0.06, 0.18, 0.06]
     = [0.21, 0.105, 0.315, 0.105]
     
  h₁ is GONE. We don't store it. Memory: still just 4 numbers ✓

───── Packet 3 (x₃ = 8.5, a SUSPICIOUS spike!) ─────
  Because Mamba is SELECTIVE, the parameters change:
  Ā₃ = [0.3, 0.3, 0.3, 0.3]    (keep only 30%! — "forget the old normal stuff")
  B̄₃ = [0.9, 0.8, 0.95, 0.7]   (let this input strongly affect state!)
  
  h₃ = [0.3,0.3,0.3,0.3] · [0.21, 0.105, 0.315, 0.105]
       + [0.9,0.8,0.95,0.7] · 8.5
     = [0.063, 0.032, 0.095, 0.032] + [7.65, 6.8, 8.075, 5.95]
     = [7.71, 6.83, 8.17, 5.98]    ← state now DOMINATED by the spike
     
  Memory: still just 4 numbers ✓

───── Packets 4, 5 continue... ─────
  Same process. Always 4 numbers. Always O(1).
```

> [!IMPORTANT]
> **Notice what happened at Packet 3**: The selective mechanism (the input-dependent $\bar{\mathbf{A}}$ and $\bar{\mathbf{B}}$) caused the model to **aggressively forget** the old normal state and **strongly absorb** the anomalous input. This is the "selective" in "Selective State Spaces." A non-selective SSM (S4) would have used the same $\bar{\mathbf{A}} = 0.9$ regardless, and the spike would have been diluted by the old state.

### The Tradeoff: What's the Catch?

O(1) memory sounds like magic. There IS a tradeoff: **lossy compression**.

```
Transformer (stores everything):
  "Packet 1 was X, Packet 2 was Y, Packet 3 was Z, ..."
  → Can go back and look at ANY specific past packet
  → Perfect recall, but expensive
  
Mamba (compresses to fixed state):
  "The general trend has been normal traffic with one big spike recently"
  → Cannot recover the exact value of Packet 1
  → Approximate recall, but cheap
```

**Why this is okay for NIDS**: You don't NEED to remember the exact bytes of packet #47 from 10 minutes ago. You need to remember the *pattern*: "traffic has been ramping up," "this IP keeps appearing," "packet sizes are suspiciously uniform." The state vector captures these **statistical patterns** without storing raw data. That's actually how human security analysts think too.

### Visual Summary

```
TRANSFORMER MEMORY MODEL:
┌──────────────────────────────────────────────┐
│ K₁ V₁ │ K₂ V₂ │ K₃ V₃ │ ... │ Kₗ Vₗ      │  ← Grows and grows
│ 128d   │ 128d   │ 128d   │     │ 128d        │
└──────────────────────────────────────────────┘
Total memory: L × 256 floats  (at L=10,000 → 2,560,000 floats)

MAMBA MEMORY MODEL:
┌─────────────────────┐
│  h = [64 floats]    │  ← Always this size. That's it.
└─────────────────────┘
Total memory: 64 floats  (at L=10,000 → still 64 floats)
```

---

## 3. Transformer vs Mamba — Detailed Comparison

### How a Transformer Processes a Sequence

The Transformer's core mechanism is **self-attention**: every token looks at every other token to decide what's important.

```
Input sequence: [P₁, P₂, P₃, P₄, P₅]  (5 packets)

Self-Attention computes:
  "How relevant is P₁ to P₁?" → score₁₁
  "How relevant is P₁ to P₂?" → score₁₂
  "How relevant is P₁ to P₃?" → score₁₃
  ... (every pair)
  "How relevant is P₅ to P₅?" → score₅₅
  
Total scores computed: 5 × 5 = 25

For 1,000 packets: 1,000 × 1,000 = 1,000,000 scores
For 10,000 packets: 10,000 × 10,000 = 100,000,000 scores
```

This is the **attention matrix** — an $L \times L$ grid where $L$ is the sequence length:

```
         P₁    P₂    P₃    P₄    P₅
    P₁ [ 0.8   0.1   0.05  0.03  0.02 ]
    P₂ [ 0.2   0.5   0.15  0.1   0.05 ]
    P₃ [ 0.1   0.3   0.4   0.15  0.05 ]    ← L×L matrix
    P₄ [ 0.05  0.1   0.2   0.5   0.15 ]
    P₅ [ 0.02  0.08  0.1   0.3   0.5  ]
    
Memory: O(L²) to store this matrix
Compute: O(L²) to calculate all scores
```

**Strength**: Every packet can directly attend to every other packet, no matter how far apart. Packet 1,000 can look directly at packet 1.

**Weakness**: The $L^2$ cost. At 10,000 packets, you need 100 million attention scores computed AND stored. This is why Transformers need massive GPUs.

### How Mamba Processes a Sequence

Mamba never compares packets to each other. Instead, it **reads packets one by one** and maintains a running state:

```
Input sequence: [P₁, P₂, P₃, P₄, P₅]  (5 packets)

Step 1: Read P₁, update state h
  h₁ = Ā₁ · h₀ + B̄₁ · P₁        (1 state update)

Step 2: Read P₂, update state h  
  h₂ = Ā₂ · h₁ + B̄₂ · P₂        (1 state update)

Step 3: Read P₃, update state h
  h₃ = Ā₃ · h₂ + B̄₃ · P₃        (1 state update)

...

Total operations: 5 (one per packet)
For 1,000 packets: 1,000 operations
For 10,000 packets: 10,000 operations
```

**Strength**: Linear cost. 10× more packets = 10× more work (not 100×).

**Weakness**: Information from early packets must "survive" through all the state updates to be remembered. It can't jump directly to an early packet like attention can. But the selective mechanism ($\bar{\mathbf{A}}$ adapting per-input) mitigates this significantly.

### Head-to-Head Comparison

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| **Core mechanism** | Self-attention: every token looks at every other token | Selective state space: each token updates a compressed running state |
| **How it "remembers"** | Stores explicit Key-Value pairs for ALL past tokens | Compresses history into a fixed-size state vector |
| **Training computation** | $O(L^2 \cdot d)$ — quadratic in sequence length | $O(L \cdot d \cdot N)$ — linear in sequence length |
| **Inference (per new token)** | $O(L \cdot d)$ — must attend to all cached tokens | $O(d \cdot N)$ — just one state update, independent of history length |
| **Training memory** | $O(L^2)$ for attention matrix + $O(L \cdot d)$ for KV | $O(L \cdot d \cdot N)$ — no attention matrix needed |
| **Inference memory** | $O(L \cdot d)$ — KV cache grows with each token | $O(d \cdot N)$ — constant, just the state |
| **Long-range dependency** | ✅ Perfect — direct attention from any token to any token | ✅ Good — information persists via state updates, selective mechanism prevents forgetting important info |
| **Content-based filtering** | ✅ Attention scores are input-dependent | ✅ $\mathbf{B}$, $\mathbf{C}$, $\Delta$ are input-dependent |
| **Training parallelism** | ✅ All attention scores computed in parallel | ✅ Parallel scan algorithm (or convolution in non-selective case) |
| **Positional encoding** | Needs explicit position embeddings (sinusoidal, RoPE, etc.) | Built-in — the sequential state update naturally encodes position |

### Concrete Example: Processing 1,000 Packets on a 16GB GPU

```
Model dimension d = 256
Attention heads = 8
Sequence length L = 1,000

─── TRANSFORMER ───
Attention matrix: L × L × heads = 1,000 × 1,000 × 8 = 8,000,000 floats
                  = 8M × 4 bytes = 32 MB (just for attention, one layer)
KV Cache: 2 × L × d = 2 × 1,000 × 256 = 512,000 floats
          = 512K × 4 bytes = 2 MB per layer
With 12 layers: 32×12 + 2×12 = 408 MB

At L = 10,000: attention alone = 3,200 MB = 3.2 GB (one layer!)
At L = 100,000: attention = 320 GB → IMPOSSIBLE on any single GPU

─── MAMBA ───
State: d_inner × d_state = 512 × 16 = 8,192 floats = 32 KB per layer
With 12 layers: 32 × 12 = 384 KB

At L = 10,000: still 384 KB
At L = 100,000: still 384 KB
At L = 1,000,000: still 384 KB ← doesn't change!
```

### When Would You Still Pick a Transformer?

Transformers aren't obsolete. They're better when:

1. **You need precise recall of specific past tokens** — e.g., "What was the exact query in packet #47?" (retrieval tasks, question answering)
2. **Sequence length is short** — at $L < 512$, the quadratic cost is manageable and attention gives more expressive power
3. **You have unlimited compute** — if hardware isn't a constraint, attention's direct token-to-token comparison is more powerful

For NIDS, you don't need precise recall of old packets, your sequences can be very long, and you need real-time performance on modest hardware → **Mamba wins.**

---

## 4. Temporal Patterns — What They Are and Why They Matter for NIDS

### Definition

A **temporal pattern** is a pattern that **only exists in the ordering and timing of events**, not in any single event on its own.

```
Individual event: "A person entered the bank"         → Normal
Individual event: "A person entered the bank"         → Normal
Individual event: "A person entered the bank"         → Normal

Temporal pattern: "47 people entered the bank in 30 seconds" → SUSPICIOUS
```

The suspicious part isn't any single person entering — it's the **rate, sequence, and timing** of many events together.

### 5 Real Temporal Patterns in Network Attacks

#### Pattern 1: DDoS Ramp-Up

```
Time:        t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7
Packets/sec: 100    100    150    300    800    2000   8000   50000
                                   ↑
                          This is where a temporal model
                          starts detecting the exponential ramp

Individual view at t=3: "300 packets this second" — is that a lot? Hard to say.
Temporal view at t=3: "Packets went 100→100→150→300 — exponential growth pattern!" → ALERT
```

A model without temporal awareness sees "300 packets" and doesn't know if that's normal for this server. A model WITH temporal awareness sees the **trend** and raises the alarm early.

#### Pattern 2: Port Scan Sweep

An attacker probes sequential ports to find open services:

```
Time →
  Connection to port 21  (FTP)     → refused
  Connection to port 22  (SSH)     → open! (noted by attacker)
  Connection to port 23  (Telnet)  → refused
  Connection to port 25  (SMTP)    → refused
  Connection to port 80  (HTTP)    → open! (noted by attacker)
  Connection to port 443 (HTTPS)   → open! (noted by attacker)
  ...

Each individual connection attempt is perfectly legal.
The TEMPORAL PATTERN (sequential port numbers from one source IP) 
reveals it's a reconnaissance scan.
```

#### Pattern 3: Command-and-Control (C2) Beaconing

Malware phones home at regular intervals:

```
Time:    00:00   00:05   00:10   00:15   00:20   00:25   00:30
Event:   ping    ping    ping    ping    ping    ping    ping
         to C2   to C2   to C2   to C2   to C2   to C2   to C2

Each individual ping: "A DNS lookup to some domain" — totally normal
Temporal pattern: "Exactly every 5 minutes, same size, same destination" 
                  → C2 BEACONING DETECTED
```

Human analysts call this "periodicity detection." The malware's regular heartbeat is invisible in individual packet analysis but screams in temporal analysis.

#### Pattern 4: Slowloris Attack

```
Time →
  Connection 1: Send "GET / HTTP/1.1\r\n"  then wait 9 seconds...
                Send "X-a: b\r\n"           then wait 9 seconds...
                Send "X-c: d\r\n"           then wait 9 seconds...
  Connection 2: (same pattern)
  Connection 3: (same pattern)
  ... × 1000 connections

Individual view: "A slow HTTP request" — annoying but not alarming
Temporal view: "1,000 connections all doing the same slow-drip pattern 
simultaneously" → SLOWLORIS ATTACK
```

#### Pattern 5: Brute Force Login

```
Time →
  POST /login {user: admin, pass: password123}     → 401 Unauthorized
  POST /login {user: admin, pass: admin}           → 401 Unauthorized
  POST /login {user: admin, pass: letmein}         → 401 Unauthorized
  POST /login {user: admin, pass: qwerty}          → 401 Unauthorized
  ... (thousands of attempts)

Each individual attempt: "A failed login" — happens all the time
Temporal pattern: "500 failed logins from the same IP in 60 seconds, 
all targeting 'admin'" → BRUTE FORCE DETECTED
```

### Why Non-Temporal Models Miss These

```
┌─────────────────────────────────────────────────────────────┐
│              What a NON-TEMPORAL model sees:                │
│                                                             │
│  Row 1: [src_ip=1.2.3.4, dst_port=22, bytes=64, flag=SYN] │
│  Row 2: [src_ip=1.2.3.4, dst_port=23, bytes=64, flag=SYN] │
│  Row 3: [src_ip=1.2.3.4, dst_port=25, bytes=64, flag=SYN] │
│                                                             │
│  Each row is classified INDEPENDENTLY.                      │
│  The model sees 3 separate, normal-looking SYN packets.     │
│  Verdict: Normal, Normal, Normal ❌                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              What MAMBA sees:                               │
│                                                             │
│  Sequence: [port22_SYN] → [port23_SYN] → [port25_SYN]     │
│                                                             │
│  After port22_SYN: h = "someone probed SSH"                │
│  After port23_SYN: h = "same IP, sequential port, scan?"   │
│  After port25_SYN: h = "definitely a port scan pattern"    │
│  Verdict: Port Scan ATTACK ✅                               │
└─────────────────────────────────────────────────────────────┘
```

### How Mamba's Hidden State Captures Temporal Patterns

The hidden state $h$ isn't just storing the last packet — it's building up a **temporal representation**:

```
Think of h as having learned to track things like:
  h[0:4]   → "running average of packet rate"
  h[4:8]   → "variance in packet sizes recently"  
  h[8:12]  → "how many distinct destination ports in recent history"
  h[12:16] → "periodicity signal — is traffic arriving in regular intervals?"

(The model learns WHAT to track during training — these aren't hardcoded)

For a DDoS:
  h[0:4] starts spiking (packet rate increasing)
  h[4:8] drops (all packets become similar size)
  → Classifier reads h and outputs: "DDoS, confidence 94%"

For a port scan:
  h[8:12] starts spiking (many different ports being contacted)
  h[0:4] stays low (packet rate isn't high)
  → Classifier reads h and outputs: "Reconnaissance, confidence 91%"
```

The selective mechanism ($\mathbf{B}$, $\mathbf{C}$, $\Delta$ being input-dependent) is what allows the model to learn WHICH temporal features to track. A normal packet doesn't change $h$ much (low $\Delta$). A suspicious packet dramatically shifts $h$ (high $\Delta$), ensuring the model remembers it.

**This is why Mamba is a natural fit for NIDS: network attacks ARE temporal patterns, and Mamba is built to encode temporal patterns into a compact, fixed-size state.**
