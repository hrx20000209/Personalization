import time
import random
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer


###############################################################################
# 1. Example raw data: you should replace this with your own records
###############################################################################

# Each record: one intervention + 6 MC scores
# Here I put your 6 records as examples. For real 10-shot, you need >= 10 items.
records = [
    {
        "user": "Meliodas",
        "intervention": "Physical Health 💪  - Great job on super low screen time—it's easing eye strain and posture woes! 🚀 "
                        "Build on those gentle campus walks with a 2-min stretch or extra laps to energize your body. 🏃‍♂️",
        "mc": {"MC1": 1, "MC2": 4, "MC3": 4, "MC4": 3, "MC5": 4, "MC6": 5},
    },
    {
        "user": "Meliodas",
        "intervention": "Physical Health 💪  - Great job on super low screen time—it's easing eye strain and posture woes! 🚀 "
                        "Build on those gentle campus walks with a 2-min stretch or extra laps to energize your body. 🏃‍♂️✨  "
                        "Mental Health 🧠  - Loving your calm, focused vibe with minimal distractions—pure mental clarity! 💫 "
                        "Keep light WeChat check-ins warming connections; pause for a deep breath to sustain that chill campus rhythm. 😊🌟",
        "mc": {"MC1": 1, "MC2": 4, "MC3": 4, "MC4": 3, "MC5": 4, "MC6": 5},
    },
    {
        "user": "yjliu",
        "intervention": "Physical Health 💪  - Lots of sitting and screen time might cause stiffness—try a 2-minute stretch or "
                        "quick campus walk to energize your body! 🚶‍♀️  - This boosts circulation and counters low movement gently. 😊  "
                        "Mental Health 🧠  - Your social chats and study focus are shining, keeping you connected and productive—awesome! 👏  "
                        "Balance screen bursts with a 1-minute deep-breath break for refreshed clarity. 🌟",
        "mc": {"MC1": 1, "MC2": 5, "MC3": 5, "MC4": 4, "MC5": 5, "MC6": 5},
    },
    {
        "user": "yjliu",
        "intervention": "Physical Health 💪  - Your balanced bursts of walking, stairs, and running are spot-on for energy without "
                        "burnout—amazing job! 🚀  - Pair low screen time with a 1-minute neck stretch to ease any tension. 🧘‍♀️  "
                        "Mental Health 🧠  - That calm focus and recharge mode at uni feels restorative—keep shining! 🌙  "
                        "Take 3 deep breaths or note one win from today to amplify the positivity. 😊",
        "mc": {"MC1": 1, "MC2": 4, "MC3": 5, "MC4": 4, "MC5": 4, "MC6": 4},
    },
    {
        "user": "hrx",
        "intervention": "Physical Health 💪  - Fantastic bursts of running and stair climbing – you're building strength and heart "
                        "health beautifully! 🌟 Follow up with a 1-minute stretch to ease any tension and boost posture.  "
                        "Mental Health 🧠  - Light screen time and real-world campus connections are fueling your vibrant energy "
                        "and low stress! 😊 Jot down one thing you're grateful for to lock in this glow.",
        "mc": {"MC1": 1, "MC2": 4, "MC3": 5, "MC4": 4, "MC5": 4, "MC6": 5},
    },
    {
        "user": "Meliodas",
        "intervention": "Physical Health 💪  - Your walking, running bursts, and stairs are fantastic for energy—keep that momentum "
                        "going! 🏃‍♂️  - Counter screen time with a quick posture stretch: shoulders back, neck rolls for 30 seconds. 😌  "
                        "Mental Health 🧠  - Loving your focused productivity and light WeChat chats—perfect balance for staying "
                        "connected without overload! 👏  - Pause for a 1-minute deep breath to sustain that energized vibe. 🌟",
        "mc": {"MC1": 1, "MC2": 4, "MC3": 4, "MC4": 4, "MC5": 4, "MC6": 3},
    }
]


###############################################################################
# 2. Reward function: combine MC1–MC6 into a scalar
###############################################################################

def combine_mc_scores(mc: Dict[str, float]) -> float:
    """
    Convert MC1..MC6 to a single scalar reward in [0,1] roughly.
    Here I assume all MC are on a 1-5 scale. You can change per-key max if needed.
    """
    # you can adjust these weights; this is just a reasonable default
    weights = {
        "MC1": 0.35,   # relevance
        "MC2": 0.25,   # usefulness
        "MC3": 0.15,   # information
        "MC4": -0.10,  # length (negative, too long is bad)
        "MC5": 0.075,
        "MC6": 0.075,
    }
    min_val = 1.0
    max_val = 5.0
    span = max_val - min_val

    score = 0.0
    for key, w in weights.items():
        raw = float(mc[key])
        norm = (raw - min_val) / span  # map 1..5 → 0..1
        score += w * norm
    # optional: rescale back to 0..1 for stability
    # here just shift and clip
    score = max(0.0, min(1.0, score + 0.1))
    return score


###############################################################################
# 3. LoRe-style single-user reward model
###############################################################################

class LoReSingleUser(nn.Module):
    """
    Lightweight LoRe-style user reward model.

    r = (V @ softmax(w)) · x

    where:
        V: [d, r] basis vectors (can be fixed or learnable)
        w: [r] user preference vector (learnable)
        x: [B, d] embedding of intervention text
    """

    def __init__(self, feature_dim: int, rank: int = 8, learn_V: bool = False, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.device = device

        if learn_V:
            self.V = nn.Parameter(torch.randn(feature_dim, rank) / feature_dim ** 0.5)
        else:
            V = torch.randn(feature_dim, rank) / feature_dim ** 0.5
            self.V = nn.Parameter(V, requires_grad=False)

        self.w = nn.Parameter(torch.zeros(rank))

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d]
        return: [B], reward prediction
        """
        x = x.to(self.device)
        Vw = self.V @ F.softmax(self.w, dim=0)   # [d]
        return (x * Vw).sum(dim=-1)


def train_single_user(model: LoReSingleUser,
                      X: torch.Tensor,
                      rewards: torch.Tensor,
                      lr: float = 0.01,
                      steps: int = 300) -> None:
    """
    Simple MSE training on few-shot data.
    X: [N, d]
    rewards: [N]
    """
    X = X.to(model.device)
    rewards = rewards.to(model.device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()
    for i in range(steps):
        pred = model(X)
        loss = F.mse_loss(pred, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f"[Train] step {i+1:4d}  loss={loss.item():.4f}")


###############################################################################
# 4. Main pipeline: 10-shot training + latency measurement
###############################################################################

def main():
    # you can change model name to something else
    encoder_name = "paraphrase-MiniLM-L6-v2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 4.1 load encoder and measure one-shot embedding latency
    print(f"Loading sentence encoder: {encoder_name}")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(encoder_name, device=str(device))
    t1 = time.perf_counter()
    print(f"Encoder loaded in {(t1 - t0):.3f}s")

    feature_dim = encoder.get_sentence_embedding_dimension()
    print("Embedding dimension:", feature_dim)

    # 4.2 build dataset: text → embedding, MC → reward
    texts: List[str] = []
    rewards: List[float] = []

    for rec in records:
        text = rec["intervention"]
        r = combine_mc_scores(rec["mc"])
        texts.append(text)
        rewards.append(r)

    # 4.3 compute embeddings and measure average latency per sample
    print("Computing embeddings and measuring latency...")
    all_embeddings = []
    times = []

    for text in texts:
        t_start = time.perf_counter()
        emb = encoder.encode(text, convert_to_tensor=True)
        t_end = time.perf_counter()
        all_embeddings.append(emb)
        times.append(t_end - t_start)

    X_all = torch.stack(all_embeddings)   # [N, d]
    y_all = torch.tensor(rewards, dtype=torch.float32)

    avg_embed_time = sum(times) / len(times)
    print(f"Average embedding latency per sample: {avg_embed_time * 1000:.2f} ms")

    # 4.4 sample 10-shot (if less than 10 records, use all)
    N = len(records)
    num_shots = min(10, N)
    idx = list(range(N))
    random.shuffle(idx)
    shot_idx = idx[:num_shots]

    X_train = X_all[shot_idx]
    y_train = y_all[shot_idx]

    print(f"Using {num_shots} shots for LoRe training")

    # 4.5 create LoRe model and train; measure training time
    lore_model = LoReSingleUser(feature_dim=feature_dim, rank=8, learn_V=False, device=device)

    train_steps = 300
    lr = 0.02

    print("Start LoRe few-shot training...")
    t_train_start = time.perf_counter()
    train_single_user(lore_model, X_train, y_train, lr=lr, steps=train_steps)
    t_train_end = time.perf_counter()
    train_time = t_train_end - t_train_start
    print(f"LoRe training time (steps={train_steps}, shots={num_shots}): {train_time:.3f} s")

    # 4.6 measure inference latency (scoring new interventions)
    lore_model.eval()

    test_text = "Try a 1-minute stretch and 3 deep breaths to relax your shoulders after screen time."
    # embedding latency
    t_e0 = time.perf_counter()
    test_emb = encoder.encode(test_text, convert_to_tensor=True)
    t_e1 = time.perf_counter()

    # LoRe forward latency (single sample)
    test_emb = test_emb.unsqueeze(0)   # [1, d]
    with torch.no_grad():
        t_r0 = time.perf_counter()
        score = lore_model(test_emb)
        t_r1 = time.perf_counter()

    print(f"Test intervention text: {test_text}")
    print(f"Predicted reward: {float(score.item()):.4f}")
    print(f"Embedding latency (one sample): {(t_e1 - t_e0) * 1000:.2f} ms")
    print(f"LoRe reward forward latency (one sample): {(t_r1 - t_r0) * 1000:.4f} ms")

    # 4.7 optional: run multiple times to get stable forward latency
    num_repeats = 50
    forward_times = []
    with torch.no_grad():
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            _ = lore_model(test_emb)
            t1 = time.perf_counter()
            forward_times.append(t1 - t0)
    avg_forward = sum(forward_times) / len(forward_times)
    print(f"Average LoRe forward latency over {num_repeats} runs: {avg_forward * 1000:.4f} ms")


if __name__ == "__main__":
    main()
