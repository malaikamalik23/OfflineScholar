# exam_engine (Member 2)

Place Member 2's code here as a Python package. The backend discovers it automatically if
`exam_engine/api.py` exists and exposes:

```python
def analyze_subject(subject_id: int, db: "sqlalchemy.orm.Session") -> dict: ...
def recommend_revision(subject_id: int, db: "sqlalchemy.orm.Session") -> dict: ...
```

See `docs/INTEGRATION.md` for the shared schema and the reusable `ai_engine` functions to call.
Until this package exists, `/api/analysis/*` and `/api/revision/*` return HTTP 501.
