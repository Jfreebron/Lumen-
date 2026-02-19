from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from typing import Optional, List, Dict, Any
import psycopg2
import os
import re
import ast
import requests
import base64
from collections import defaultdict

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


def get_schema(db_url: str) -> str:
    conn = psycopg2.connect(db_url)
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

    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items())


def generate_sql(question: str, db_url: str) -> str:
    schema = get_schema(db_url)
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
- Always limit to 1000 rows
- Use double quotes for table and column names if they contain special characters
- You CANNOT dynamically iterate over tables or use dynamic SQL
- If asked about "empty tables" or "tables without data", you cannot answer that with SQL - just return: SELECT 'Cannot check all tables dynamically' as message
- If asked to compare across all tables, pick the most relevant tables based on the question
- For questions about users, customers, orders, etc. query those specific tables"""
        }],
        max_tokens=500,
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql?\n?", "", sql).replace("```", "").strip()
    return sql


# Security: block dangerous queries (check for SQL commands, not column names)
def is_safe(sql: str) -> bool:
    sql_lower = sql.lower().strip()

    # Must start with SELECT or WITH (for CTEs)
    if not (sql_lower.startswith('select') or sql_lower.startswith('with')):
        return False

    # Block dangerous statements - check for command patterns, not just words
    # These patterns look for the command at word boundaries
    dangerous_patterns = [
        r'\binsert\s+into\b',
        r'\bupdate\s+\w+\s+set\b',
        r'\bdelete\s+from\b',
        r'\bdrop\s+(table|database|index|view)\b',
        r'\balter\s+(table|database)\b',
        r'\bcreate\s+(table|database|index|view)\b',
        r'\btruncate\s+',
        r'\bgrant\s+',
        r'\brevoke\s+',
        r'\bexec(ute)?\s*\(',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, sql_lower):
            return False

    return True


class QueryRequest(BaseModel):
    question: str
    db_url: str


class TablesRequest(BaseModel):
    db_url: str


@app.post("/api/query")
def query(req: QueryRequest):
    if not req.question:
        raise HTTPException(400, "Question required")
    if not req.db_url:
        raise HTTPException(400, "Database URL required")

    sql = generate_sql(req.question, req.db_url)

    if not is_safe(sql):
        raise HTTPException(400, "Only SELECT queries allowed")

    try:
        conn = psycopg2.connect(req.db_url)
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
    except psycopg2.OperationalError as e:
        raise HTTPException(400, f"Connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Query failed: {str(e)}")


@app.post("/api/tables")
def tables(req: TablesRequest):
    if not req.db_url:
        raise HTTPException(400, "Database URL required")

    try:
        conn = psycopg2.connect(req.db_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        rows = cur.fetchall()
        conn.close()
        return {"tables": [r[0] for r in rows]}
    except psycopg2.OperationalError as e:
        raise HTTPException(400, f"Connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Failed to get tables: {str(e)}")


class SchemaDetailsRequest(BaseModel):
    db_url: str
    table: str


@app.post("/api/table-stats")
def table_stats(req: TablesRequest):
    """Get row counts for all tables - useful for finding empty tables"""
    if not req.db_url:
        raise HTTPException(400, "Database URL required")

    try:
        conn = psycopg2.connect(req.db_url)
        cur = conn.cursor()

        # Get all tables
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]

        # Get row count for each table
        stats = []
        for table in tables:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                count = cur.fetchone()[0]
                stats.append({"table": table, "rows": count})
            except:
                stats.append({"table": table, "rows": -1})

        conn.close()
        return {"stats": stats}
    except psycopg2.OperationalError as e:
        raise HTTPException(400, f"Connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Failed to get stats: {str(e)}")


@app.post("/api/schema-details")
def schema_details(req: SchemaDetailsRequest):
    if not req.db_url:
        raise HTTPException(400, "Database URL required")
    if not req.table:
        raise HTTPException(400, "Table name required")

    try:
        conn = psycopg2.connect(req.db_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = %s
            ORDER BY ordinal_position
        """, (req.table,))
        rows = cur.fetchall()
        conn.close()
        return {"columns": [{"name": r[0], "type": r[1]} for r in rows]}
    except psycopg2.OperationalError as e:
        raise HTTPException(400, f"Connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Failed to get schema: {str(e)}")


# ============== INSIGHTS AGENT ==============

@app.post("/api/insights")
def get_insights(req: TablesRequest):
    """
    Comprehensive database insights agent that analyzes:
    - Foreign key relationships
    - Primary keys
    - Data quality (NULL rates, empty tables)
    - Orphaned records
    - Suggested questions
    - Overall health score
    """
    if not req.db_url:
        raise HTTPException(400, "Database URL required")

    try:
        conn = psycopg2.connect(req.db_url)
        cur = conn.cursor()

        insights = {
            "relationships": [],
            "orphaned_tables": [],
            "data_quality": [],
            "key_tables": [],
            "suggested_questions": [],
            "health_score": 0,
            "summary": {}
        }

        # 1. Get all tables
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        all_tables = [r[0] for r in cur.fetchall()]

        # 2. Get foreign key relationships
        cur.execute("""
            SELECT
                tc.table_name as from_table,
                kcu.column_name as from_column,
                ccu.table_name as to_table,
                ccu.column_name as to_column,
                tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
        """)
        fk_rows = cur.fetchall()

        relationships = []
        tables_with_fk = set()
        referenced_tables = set()

        for from_table, from_col, to_table, to_col, constraint in fk_rows:
            tables_with_fk.add(from_table)
            referenced_tables.add(to_table)
            relationships.append({
                "from_table": from_table,
                "from_column": from_col,
                "to_table": to_table,
                "to_column": to_col,
                "description": f"{from_table} → {to_table} (via {from_col})"
            })

        insights["relationships"] = relationships

        # 3. Get primary keys
        cur.execute("""
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = 'public'
        """)
        pk_rows = cur.fetchall()
        tables_with_pk = set(r[0] for r in pk_rows)

        # 4. Find orphaned tables (no FK relationships)
        orphaned = []
        for table in all_tables:
            if table not in tables_with_fk and table not in referenced_tables:
                orphaned.append(table)
        insights["orphaned_tables"] = orphaned

        # 5. Identify key/central tables (most referenced)
        reference_count = defaultdict(int)
        for rel in relationships:
            reference_count[rel["to_table"]] += 1

        key_tables = sorted(reference_count.items(), key=lambda x: -x[1])[:10]
        insights["key_tables"] = [{"table": t, "references": c} for t, c in key_tables]

        # 6. Data quality analysis (sample tables)
        data_quality = []
        tables_to_check = all_tables[:30]  # Limit for performance

        for table in tables_to_check:
            try:
                # Get row count
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                row_count = cur.fetchone()[0]

                # Get columns
                cur.execute("""
                    SELECT column_name, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """, (table,))
                columns = cur.fetchall()

                # Check NULL rates for nullable columns (sample)
                null_issues = []
                if row_count > 0:
                    for col_name, is_nullable in columns[:10]:  # Limit columns
                        try:
                            cur.execute(f'''
                                SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL
                            ''')
                            null_count = cur.fetchone()[0]
                            if null_count > 0:
                                null_pct = round((null_count / row_count) * 100, 1)
                                if null_pct > 50:
                                    null_issues.append({
                                        "column": col_name,
                                        "null_percentage": null_pct
                                    })
                        except:
                            pass

                quality_item = {
                    "table": table,
                    "row_count": row_count,
                    "column_count": len(columns),
                    "has_primary_key": table in tables_with_pk,
                    "null_issues": null_issues[:5]  # Top 5 issues
                }

                if row_count == 0:
                    quality_item["issue"] = "empty"
                elif not quality_item["has_primary_key"]:
                    quality_item["issue"] = "no_primary_key"
                elif len(null_issues) > 0:
                    quality_item["issue"] = "high_nulls"

                data_quality.append(quality_item)
            except Exception as e:
                pass

        insights["data_quality"] = data_quality

        # 7. Generate smart suggested questions based on schema
        suggestions = []

        # Find tables with common patterns
        user_tables = [t for t in all_tables if 'user' in t.lower()]
        order_tables = [t for t in all_tables if 'order' in t.lower() or 'deal' in t.lower()]
        company_tables = [t for t in all_tables if 'compan' in t.lower()]
        product_tables = [t for t in all_tables if 'product' in t.lower()]
        contact_tables = [t for t in all_tables if 'contact' in t.lower()]
        log_tables = [t for t in all_tables if 'log' in t.lower() or 'audit' in t.lower()]

        if user_tables:
            suggestions.append({"question": "How many users signed up this month?", "category": "Users"})
            suggestions.append({"question": "Show me the most active users", "category": "Users"})

        if order_tables or 'closed_deals' in all_tables:
            suggestions.append({"question": "What is our total revenue?", "category": "Sales"})
            suggestions.append({"question": "Show deals closed this quarter", "category": "Sales"})
            suggestions.append({"question": "Who are our top customers by revenue?", "category": "Sales"})

        if company_tables:
            suggestions.append({"question": "How many companies do we have?", "category": "Companies"})
            suggestions.append({"question": "Show companies added recently", "category": "Companies"})

        if contact_tables:
            suggestions.append({"question": "How many contacts per company on average?", "category": "Contacts"})

        if product_tables:
            suggestions.append({"question": "What are our best selling products?", "category": "Products"})

        # Relationship-based suggestions
        if key_tables:
            top_table = key_tables[0][0]
            suggestions.append({
                "question": f"Show me a breakdown by {top_table.replace('_', ' ')}",
                "category": "Analysis"
            })

        if log_tables:
            suggestions.append({"question": "Show recent activity logs", "category": "Activity"})

        # Add generic suggestions
        suggestions.append({"question": "Which tables have the most data?", "category": "Overview"})
        suggestions.append({"question": "Show me a summary of our key metrics", "category": "Overview"})

        insights["suggested_questions"] = suggestions[:12]  # Limit to 12

        # 8. Calculate health score (0-100)
        score = 100
        total_tables = len(all_tables)

        if total_tables == 0:
            score = 0
        else:
            # Deduct for empty tables
            empty_tables = len([q for q in data_quality if q.get("row_count", 0) == 0])
            empty_pct = (empty_tables / total_tables) * 100
            score -= min(empty_pct * 0.3, 20)  # Max 20 points off

            # Deduct for orphaned tables
            orphan_pct = (len(orphaned) / total_tables) * 100
            score -= min(orphan_pct * 0.2, 15)  # Max 15 points off

            # Deduct for missing PKs
            missing_pk = total_tables - len(tables_with_pk)
            pk_pct = (missing_pk / total_tables) * 100
            score -= min(pk_pct * 0.3, 20)  # Max 20 points off

            # Deduct for high null rates
            tables_with_nulls = len([q for q in data_quality if len(q.get("null_issues", [])) > 0])
            null_pct = (tables_with_nulls / max(len(data_quality), 1)) * 100
            score -= min(null_pct * 0.15, 15)  # Max 15 points off

        insights["health_score"] = max(0, round(score))

        # 9. Summary
        insights["summary"] = {
            "total_tables": total_tables,
            "total_relationships": len(relationships),
            "tables_with_data": len([q for q in data_quality if q.get("row_count", 0) > 0]),
            "empty_tables": len([q for q in data_quality if q.get("row_count", 0) == 0]),
            "orphaned_tables": len(orphaned),
            "tables_without_pk": total_tables - len(tables_with_pk)
        }

        conn.close()

        # 10. Generate AI summary using Groq
        try:
            ai_summary = generate_ai_insights_summary(insights)
            insights["ai_summary"] = ai_summary
        except:
            insights["ai_summary"] = None

        return insights

    except psycopg2.OperationalError as e:
        raise HTTPException(400, f"Connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(400, f"Failed to generate insights: {str(e)}")


def generate_ai_insights_summary(insights: dict) -> str:
    """Use AI to generate a plain-English summary of the insights"""
    summary = insights.get("summary", {})
    relationships = insights.get("relationships", [])
    key_tables = insights.get("key_tables", [])
    health_score = insights.get("health_score", 0)

    prompt = f"""You are a friendly data analyst explaining a database to a non-technical person.

Database stats:
- {summary.get('total_tables', 0)} tables total
- {summary.get('tables_with_data', 0)} tables have data
- {summary.get('empty_tables', 0)} empty tables
- {summary.get('total_relationships', 0)} relationships between tables
- {summary.get('orphaned_tables', 0)} standalone tables (no connections)
- Health score: {health_score}/100

Key tables (most connected): {', '.join([t['table'] for t in key_tables[:5]])}

Sample relationships: {', '.join([r['description'] for r in relationships[:5]])}

Write a 2-3 sentence friendly summary explaining:
1. What kind of data this database seems to contain
2. How well organized it is
3. One actionable suggestion

Keep it simple and non-technical. No bullet points."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


# ============== DEAD CODE ANALYZER ==============

class DeadCodeRequest(BaseModel):
    owner: str
    repo: str
    branch: str = "main"
    token: Optional[str] = None


def get_github_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Get headers for GitHub API requests"""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Lumen-Code-Analyzer"
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_repo_files(owner: str, repo: str, branch: str, token: Optional[str] = None) -> List[Dict]:
    """Fetch list of files from a GitHub repository"""
    headers = get_github_headers(token)

    # Get the tree recursively
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    response = requests.get(url, headers=headers)

    if response.status_code == 404:
        raise HTTPException(404, "Repository or branch not found. Check the URL or make sure you have access.")
    elif response.status_code == 403:
        raise HTTPException(403, "Rate limited. Please provide a GitHub token for higher limits.")
    elif response.status_code != 200:
        raise HTTPException(response.status_code, f"GitHub API error: {response.text}")

    data = response.json()
    files = []

    # Filter for code files we can analyze
    analyzable_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
    skip_patterns = ['node_modules/', 'venv/', '.git/', '__pycache__/', 'dist/', 'build/', '.next/', 'vendor/']

    for item in data.get("tree", []):
        if item["type"] != "blob":
            continue

        path = item["path"]

        # Skip unwanted directories
        if any(skip in path for skip in skip_patterns):
            continue

        # Check extension
        ext = os.path.splitext(path)[1].lower()
        if ext in analyzable_extensions:
            files.append({
                "path": path,
                "sha": item["sha"],
                "size": item.get("size", 0),
                "language": "python" if ext == ".py" else "javascript"
            })

    return files


def fetch_file_content(owner: str, repo: str, sha: str, token: Optional[str] = None) -> str:
    """Fetch content of a specific file from GitHub"""
    headers = get_github_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs/{sha}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return ""

    data = response.json()
    content = data.get("content", "")
    encoding = data.get("encoding", "base64")

    if encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except:
            return ""
    return content


def analyze_python_code(content: str, filepath: str) -> List[Dict]:
    """Analyze Python code for dead code patterns"""
    findings = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return findings

    # Collect all definitions
    definitions = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private/magic methods
            if not node.name.startswith('_'):
                definitions.append({
                    "type": "function",
                    "name": node.name,
                    "line": node.lineno
                })
        elif isinstance(node, ast.ClassDef):
            definitions.append({
                "type": "class",
                "name": node.name,
                "line": node.lineno
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append({
                    "name": name,
                    "module": alias.name,
                    "line": node.lineno
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append({
                    "name": name,
                    "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                    "line": node.lineno
                })

    # Check for unused definitions (simple heuristic: count occurrences in code)
    for defn in definitions:
        # Count occurrences (excluding the definition itself)
        pattern = r'\b' + re.escape(defn["name"]) + r'\b'
        matches = list(re.finditer(pattern, content))

        # If only appears once (the definition), it might be unused
        if len(matches) <= 1:
            findings.append({
                "type": f"unused_{defn['type']}",
                "name": defn["name"],
                "file": filepath,
                "line": defn["line"],
                "confidence": "medium",
                "reason": f"Function '{defn['name']}' appears to have no calls within this file"
            })

    # Check for unused imports
    for imp in imports:
        pattern = r'\b' + re.escape(imp["name"]) + r'\b'
        matches = list(re.finditer(pattern, content))

        # Import statement is one match, so if only 1 match, it's unused
        if len(matches) <= 1:
            findings.append({
                "type": "unused_import",
                "name": imp["name"],
                "file": filepath,
                "line": imp["line"],
                "confidence": "high",
                "reason": f"Import '{imp['name']}' is never used in this file"
            })

    return findings


def analyze_javascript_code(content: str, filepath: str) -> List[Dict]:
    """Analyze JavaScript/TypeScript code for dead code patterns"""
    findings = []
    lines = content.split('\n')

    # Patterns for function definitions
    function_patterns = [
        r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',  # function declarations
        r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\(',  # arrow functions
        r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?function',  # function expressions
        r'(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',  # exported functions
    ]

    # Patterns for imports
    import_patterns = [
        r'import\s+{([^}]+)}\s+from',  # named imports
        r'import\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s+from',  # default imports
        r'import\s+\*\s+as\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s+from',  # namespace imports
    ]

    # Find all function definitions
    functions = []
    for i, line in enumerate(lines, 1):
        for pattern in function_patterns:
            match = re.search(pattern, line)
            if match:
                name = match.group(1)
                # Skip common patterns that are typically used
                if name not in ['render', 'constructor', 'componentDidMount', 'useEffect', 'useState']:
                    functions.append({"name": name, "line": i})

    # Find all imports
    imports = []
    for i, line in enumerate(lines, 1):
        for pattern in import_patterns:
            match = re.search(pattern, line)
            if match:
                # Handle named imports (comma-separated)
                names_str = match.group(1)
                if ',' in names_str or '{' not in line:
                    for name in re.findall(r'([a-zA-Z_$][a-zA-Z0-9_$]*)', names_str):
                        if name not in ['as', 'from']:
                            imports.append({"name": name, "line": i})
                else:
                    imports.append({"name": names_str.strip(), "line": i})

    # Check for unused functions
    for func in functions:
        pattern = r'\b' + re.escape(func["name"]) + r'\b'
        matches = list(re.finditer(pattern, content))

        if len(matches) <= 1:
            findings.append({
                "type": "unused_function",
                "name": func["name"],
                "file": filepath,
                "line": func["line"],
                "confidence": "medium",
                "reason": f"Function '{func['name']}' appears to have no calls within this file"
            })

    # Check for unused imports
    for imp in imports:
        pattern = r'\b' + re.escape(imp["name"]) + r'\b'
        matches = list(re.finditer(pattern, content))

        if len(matches) <= 1:
            findings.append({
                "type": "unused_import",
                "name": imp["name"],
                "file": filepath,
                "line": imp["line"],
                "confidence": "high",
                "reason": f"Import '{imp['name']}' is never used in this file"
            })

    return findings


def analyze_with_ai(content: str, filepath: str, language: str, static_findings: List[Dict]) -> List[Dict]:
    """Use AI to validate and enhance findings"""
    if len(content) > 10000:  # Skip very large files
        return static_findings

    # Build context from static findings
    static_summary = "\n".join([f"- {f['type']}: {f['name']} at line {f['line']}" for f in static_findings[:10]])

    prompt = f"""Analyze this {language} code for dead/unused code. Focus on:
1. Validating these potential issues found by static analysis:
{static_summary if static_findings else "No static findings yet"}

2. Finding additional issues like:
- Unused variables
- Unreachable code
- Functions defined but never exported or called

Code ({filepath}):
```{language}
{content[:5000]}
```

Return ONLY a JSON array of findings. Each finding must have:
- type: "unused_function", "unused_import", "unused_variable", or "unreachable_code"
- name: the identifier name
- line: approximate line number
- confidence: "high", "medium", or "low"
- reason: brief explanation

If no issues found, return an empty array: []
Return ONLY valid JSON, no explanation."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
        if result.startswith('['):
            import json
            ai_findings = json.loads(result)

            # Add file path to each finding
            for f in ai_findings:
                f["file"] = filepath

            return ai_findings
    except:
        pass

    return static_findings


@app.post("/api/deadcode/analyze")
def analyze_deadcode(req: DeadCodeRequest):
    """Analyze a GitHub repository for dead code"""

    try:
        # Fetch file list
        files = fetch_repo_files(req.owner, req.repo, req.branch, req.token)

        if not files:
            return {
                "repository": {"name": f"{req.owner}/{req.repo}"},
                "findings": [],
                "summary": {
                    "files_analyzed": 0,
                    "total_issues": 0,
                    "unused_functions": 0,
                    "unused_imports": 0,
                    "unused_variables": 0
                }
            }

        # Limit files to analyze (for timeout reasons)
        files_to_analyze = files[:30]  # Analyze up to 30 files

        all_findings = []

        for file_info in files_to_analyze:
            # Fetch file content
            content = fetch_file_content(req.owner, req.repo, file_info["sha"], req.token)

            if not content or len(content) < 10:
                continue

            # Analyze based on language
            if file_info["language"] == "python":
                findings = analyze_python_code(content, file_info["path"])
            else:
                findings = analyze_javascript_code(content, file_info["path"])

            all_findings.extend(findings)

        # Use AI to validate top findings (limit to avoid timeout)
        if all_findings and len(files_to_analyze) <= 10:
            # Pick a few files with findings for AI analysis
            files_with_findings = list(set(f["file"] for f in all_findings))[:3]

            for filepath in files_with_findings:
                file_info = next((f for f in files_to_analyze if f["path"] == filepath), None)
                if file_info:
                    content = fetch_file_content(req.owner, req.repo, file_info["sha"], req.token)
                    file_findings = [f for f in all_findings if f["file"] == filepath]

                    ai_findings = analyze_with_ai(content, filepath, file_info["language"], file_findings)

                    # Replace static findings with AI findings for this file
                    all_findings = [f for f in all_findings if f["file"] != filepath]
                    all_findings.extend(ai_findings)

        # Calculate summary
        summary = {
            "files_analyzed": len(files_to_analyze),
            "total_issues": len(all_findings),
            "unused_functions": len([f for f in all_findings if f["type"] == "unused_function"]),
            "unused_imports": len([f for f in all_findings if f["type"] == "unused_import"]),
            "unused_variables": len([f for f in all_findings if f["type"] == "unused_variable"])
        }

        return {
            "repository": {
                "name": f"{req.owner}/{req.repo}",
                "branch": req.branch,
                "total_files": len(files),
                "files_analyzed": len(files_to_analyze)
            },
            "findings": all_findings,
            "summary": summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
