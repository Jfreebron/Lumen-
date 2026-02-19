from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
import psycopg2
import os
import re

app = FastAPI()

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    with open(html_path, "r") as f:
        return f.read()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cache schema on cold start
SCHEMA = None

def get_schema():
    global SCHEMA
    if SCHEMA:
        return SCHEMA

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    rows = cur.fetchall()
    conn.close()

    tables = {}
    for table, col, dtype in rows:
        if table not in tables:
            tables[table] = []
        tables[table].append(f"{col}:{dtype}")

    SCHEMA = "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items())
    return SCHEMA


def generate_sql(question: str) -> str:
    schema = get_schema()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are a SQL expert. Given this schema:
{schema}

Write a PostgreSQL query for: {question}

Rules:
- Return ONLY the SQL, no explanation
- Use lowercase for SQL keywords
- Always limit to 1000 rows"""
        }],
        max_tokens=500,
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql?\n?", "", sql).replace("```", "").strip()
    return sql


# Security: block dangerous queries
BLOCKED = ['insert', 'update', 'delete', 'drop', 'alter', 'create', 'truncate', 'grant', 'revoke']

def is_safe(sql: str) -> bool:
    sql_lower = sql.lower()
    return not any(word in sql_lower for word in BLOCKED)


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question:
        raise HTTPException(400, "Question required")

    sql = generate_sql(req.question)

    if not is_safe(sql):
        raise HTTPException(400, "Only SELECT queries allowed")

    try:
        conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        conn.close()

        # Convert to JSON-serializable format
        rows = [list(row) for row in rows]
        for row in rows:
            for i, val in enumerate(row):
                if not isinstance(val, (str, int, float, bool, type(None))):
                    row[i] = str(val)

        return {
            "sql": sql,
            "columns": columns,
            "rows": rows
        }
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")


@app.get("/api/schema")
def schema():
    return {"schema": get_schema()}
