from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from typing import Optional, List, Dict, Any
import psycopg2
import os
import re
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
