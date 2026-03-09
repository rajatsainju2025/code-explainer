import time
from code_explainer.security import ContentFilter

code = """
password = 'secret'
api_key = 'abcd'
for i in range(100):
    pass
"""

start = time.time()
for _ in range(1000):
    f = ContentFilter()
    f.scan(code)
end = time.time()
print(f"Instantiated and scanned ContentFilter 1000x in {end-start:.4f}s")
