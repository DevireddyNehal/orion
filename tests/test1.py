from time import perf_counter
from kokoro import KModel

t = perf_counter()
KModel(repo_id="hexgrad/Kokoro-82M")
print(f"KModel: {perf_counter()-t:.2f}s")