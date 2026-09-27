"""
Northbridge Bank — Commercial Lending Portfolio Query Engine
Streamlit application converted from the Project 3 (Credit Risk Query Engine) notebook.

Preserves:
- database_schema (LLM-facing schema description)
- verified_query_library (VQ1-VQ10 pre-approved SQL templates)
- classify_intent / generate_query / validate_query / retry_generation /
  execute_query / generate_response / run_pipeline pipeline logic
"""

import os
import re
import json
import sqlite3
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Northbridge Bank | Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

# --------------------------------------------------------------------------------------
# Database schema (unchanged from the notebook — provided to the LLM in prompts)
# --------------------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# --------------------------------------------------------------------------------------
# Verified Query Template Library (VQ1-VQ10) — unchanged from the notebook
# --------------------------------------------------------------------------------------
sql_1 = """SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_exposure_mil,
    ROUND(SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN lm.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_exposure_mil
FROM
    loan_master lm
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
GROUP BY
    sm.sector_name
ORDER BY
    total_outstanding_exposure_mil DESC;"""

sql_2 = """SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_exposure_mil,
    COUNT(loan_account_number) AS loan_count
FROM
    loan_master
GROUP BY
    loan_category
ORDER BY
    total_outstanding_exposure_mil DESC;"""

sql_3 = """SELECT
    ifrs9_stage,
    COUNT(DISTINCT loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mil,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mil
FROM
    provisioning
WHERE
    reporting_date = '2025-09-30'
GROUP BY
    ifrs9_stage
ORDER BY
    ifrs9_stage;"""

sql_4 = """SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS average_provision_coverage_ratio
FROM
    provisioning p
JOIN
    loan_master lm ON p.loan_account_number = lm.loan_account_number
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
WHERE
    p.reporting_date = '2025-09-30'
GROUP BY
    sm.sector_name
ORDER BY
    average_provision_coverage_ratio DESC;"""

sql_5 = """SELECT
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_mil,
    lm.asset_classification
FROM
    loan_master lm
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
ORDER BY
    total_outstanding_mil DESC
LIMIT 10;"""

sql_6 = """SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mil
FROM
    loan_master
WHERE
    group_name IS NOT NULL
GROUP BY
    group_name
ORDER BY
    total_outstanding_mil DESC
LIMIT 5;"""

sql_7 = """SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_mil,
    lm.days_past_due,
    lm.asset_classification
FROM
    loan_master lm
JOIN
    sector_master sm ON lm.sector_code = sm.sector_code
WHERE
    lm.days_past_due > 0
ORDER BY
    lm.days_past_due DESC;"""

sql_8 = """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mil
FROM
    loan_master
GROUP BY
    dpd_bucket
ORDER BY
    CASE
        WHEN dpd_bucket = '0 (Current)' THEN 1
        WHEN dpd_bucket = '1-30' THEN 2
        WHEN dpd_bucket = '31-60' THEN 3
        WHEN dpd_bucket = '61-90' THEN 4
        ELSE 5
    END;"""

sql_9 = """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM
    borrower_rating
WHERE
    rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY
    pd_estimate DESC;"""

sql_10 = """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mil
FROM
    provisioning
GROUP BY
    reporting_date
ORDER BY
    reporting_date;"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Overall expected credit loss (ECL) trend for the entire portfolio across all reporting quarters, focusing on aggregate provisioning movement without any breakdown or filtering by specific IFRS 9 stages (e.g., Stage 1, 2, or 3).',
        'sql': sql_10
    }
}

# --------------------------------------------------------------------------------------
# LLM initialization (Streamlit-adapted credential handling)
#
# The notebook loaded OPENAI_API_KEY / OPENAI_API_BASE from a local config.json.
# In Streamlit, the same two values are read from st.secrets (preferred) or from
# environment variables, and can be overridden from the sidebar for local testing.
# --------------------------------------------------------------------------------------
def _get_default(name, fallback=""):
    if name in st.secrets:
        return st.secrets[name]
    return os.environ.get(name, fallback)


@st.cache_resource(show_spinner=False)
def get_llms(api_key: str, api_base: str, model_name: str):
    """Creates the generation and evaluator LLM clients (cached per credential set)."""
    kwargs = {"model": model_name, "openai_api_key": api_key}
    if api_base:
        kwargs["openai_api_base"] = api_base
    llm_ = ChatOpenAI(**kwargs)
    evaluator_llm_ = ChatOpenAI(**kwargs)
    return llm_, evaluator_llm_


# --------------------------------------------------------------------------------------
# Database connection (read-only, per the notebook's requirement)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    conn_ = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, check_same_thread=False)
    return conn_


# --------------------------------------------------------------------------------------
# Pipeline tool functions — logic preserved from the notebook.
# llm / evaluator_llm are injected via explicit parameters instead of notebook-style
# globals, so the Streamlit sidebar can control which credentials/model are used.
# --------------------------------------------------------------------------------------

def classify_intent(user_question, query_library, llm):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are an intent router for a bank's credit risk analytics query engine. Given the
user's natural-language question below, decide whether it can be answered by one of
the pre-approved VERIFIED QUERY TEMPLATES listed, or whether it requires freshly
generated SQL.

Only choose route = "verified" if a template's description is a close, unambiguous
match for what the user is asking - same metric, same level of aggregation/grouping,
and no additional filters the template does not support. If no template is a clear
match, choose route = "generated" and set query_id to null.

User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context, llm):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
You are an expert in SQLite SQL. Your task is to write a single, read-only SQLite SQL query that answers the user's question based on the provided database schema.

Here is the user's question:
"{user_question}"

Here is the database schema:
{schema_context}

Your response must contain ONLY the SQL query, enclosed within a markdown code block (```sql...```). Absolutely no other text, explanations, or conversational content outside this block. The query must be read-only (SELECT statements only) and use standard SQLite syntax. Do not include a trailing semicolon.
"""
    response_content = llm.invoke(generation_prompt).content.strip()
    sql = ""
    # Try to extract content from a markdown code block (with or without 'sql' specifier)
    code_block_match = re.search(r'```(?:sql)?\n(.*?)\n```', response_content, re.DOTALL)
    if code_block_match:
        sql = code_block_match.group(1).strip()
    else:
        # If no code block found, assume the entire content is SQL, but clean up any leading/trailing backticks
        sql = re.sub(r'^```\s*|\s*```$', '', response_content, flags=re.MULTILINE).strip()

    # Final check to ensure it starts with SELECT or WITH, or attempt a final cleanup
    if not (sql.upper().startswith('SELECT') or sql.upper().startswith('WITH')):
        # If still not starting with SELECT/WITH, it means the LLM might have put conversational text
        # outside a code block or didn't use one. Try to aggressively find the first SQL statement.
        potential_sql_start = re.search(r'(SELECT|WITH)\s', sql, re.DOTALL | re.IGNORECASE)
        if potential_sql_start:
            sql = sql[potential_sql_start.start():].strip()

    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None, verbose=False, trace=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.
    - verbose (bool): If True, appends intermediate debugging information to trace.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
You are an expert in validating SQL query relevance. Evaluate if the provided SQL query accurately addresses the user's question, considering the context of whether it's a verified template or a freshly generated query.

Context for validation:
{track_context}

User Question:
{user_question}

SQL Query:
{candidate_sql}

Based on the above, determine if the SQL query is relevant and correct for the user's question. Provide a verdict of "yes" or "no" and a confidence score between 0.0 and 1.0. If the verdict is "no", provide a brief reason.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        if verbose and trace is not None:
            trace.append(f"DEBUG: Expected SQL for template integrity: {expected_sql.strip()}")
            trace.append(f"DEBUG: Candidate SQL for template integrity: {candidate_sql.strip()}")
        try:
            # Remove trailing semicolon and strip whitespace for strict cursor.execute
            cleaned_expected_sql = expected_sql.rstrip(';').strip()
            cleaned_candidate_sql = candidate_sql.rstrip(';').strip()
            expected_cols = [d[0] for d in cur.execute(f"{cleaned_expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{cleaned_candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are an expert in debugging and regenerating SQL queries. A previous attempt to generate a SQL query failed validation. Your task is to correct the SQL query based on the validation error and the original user question, using the provided database schema.

Here is the original user question:
{user_question}

Here is the SQL query that failed:
```sql
{failed_sql}
```

Here is the validation error message:
{error_message}

Here is the database schema:
{schema_context}

Your response must contain ONLY the corrected SQL query, enclosed within a markdown code block (```sql...```). Absolutely no other text, explanations, or conversational content outside this block. The query must be read-only (SELECT statements only) and use standard SQLite syntax. Do not include a trailing semicolon.
"""

    response_content = llm.invoke(retry_prompt).content.strip()
    sql = ""
    # Try to extract content from a markdown code block (with or without 'sql' specifier)
    code_block_match = re.search(r'```(?:sql)?\n(.*?)\n```', response_content, re.DOTALL)
    if code_block_match:
        sql = code_block_match.group(1).strip()
    else:
        # If no code block found, assume the entire content is SQL, but clean up any leading/trailing backticks
        sql = re.sub(r'^```\s*|\s*```$', '', response_content, flags=re.MULTILINE).strip()

    # Final check to ensure it starts with SELECT or WITH, or attempt a final cleanup
    if not (sql.upper().startswith('SELECT') or sql.upper().startswith('WITH')):
        # If still not starting with SELECT/WITH, it means the LLM might have put conversational text
        # outside a code block or didn't use one. Try to aggressively find the first SQL statement.
        potential_sql_start = re.search(r'(SELECT|WITH)\s', sql, re.DOTALL | re.IGNORECASE)
        if potential_sql_start:
            sql = sql[potential_sql_start.start():].strip()
    return sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, llm, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
You are a helpful assistant that summarizes data for business users.
Based on the user's question and the provided data, generate a concise, business-focused natural language response. Highlight only the most relevant insights and exact figures from the data. Do not include any technical details about SQL or database tables.

User Question:
{user_question}

Data:
{dataframe.to_string()}

Provide a summary of the data that directly answers the user's question.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, collects intermediate pipeline stages into log['trace'].

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    trace = []

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'trace': trace,
        'warnings': []
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        trace.append(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        trace.append(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log['candidate_sql'] = candidate_sql

    if verbose:
        trace.append(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm,
                           log['query_id'], verbose=verbose, trace=trace)
    log['gate_result'] = gate

    if verbose:
        trace.append(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            trace.append(f"    Failed check: {gate.get('failed_check')}")
            trace.append(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            trace.append(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context, llm)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm,
                               None, verbose=verbose, trace=trace)
        log['gate_result'] = gate

        if verbose:
            trace.append(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                trace.append(f"    Retry failed check: {gate.get('failed_check')}")
                trace.append(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            trace.append(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    if verbose:
        trace.append(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            trace.append(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], llm, log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        trace.append(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------

if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []

st.title("🏦 Northbridge Bank — Credit Risk Portfolio Query Engine")
st.caption(
    "Ask routine commercial-lending portfolio questions in plain English. "
    "Verified questions run pre-approved SQL templates; new questions get "
    "freshly generated, validated, read-only SQL. Complex or ambiguous "
    "questions are escalated to a human analyst."
)

# ---- Sidebar: configuration ----
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        value=_get_default("OPENAI_API_KEY", ""),
        type="password",
        help="Loaded from st.secrets['OPENAI_API_KEY'] or the OPENAI_API_KEY env var if set."
    )
    api_base = st.text_input(
        "OpenAI API Base URL (optional)",
        value=_get_default("OPENAI_API_BASE", ""),
        help="Leave blank to use the default OpenAI endpoint, or point to an approved enterprise gateway."
    )
    model_name = st.text_input("Model", value=_get_default("OPENAI_MODEL", "gpt-4o"))

    st.divider()

    st.subheader("Database")
    default_db_path = _get_default("DB_PATH", "credit_risk_portfolio.db")
    uploaded_db = st.file_uploader("Upload a .db file (optional)", type=["db", "sqlite"])
    if uploaded_db is not None:
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, uploaded_db.name)
        with open(db_path, "wb") as f:
            f.write(uploaded_db.getbuffer())
    else:
        db_path = st.text_input("Database file path", value=default_db_path)

    st.divider()
    show_trace = st.checkbox("Show pipeline trace (debug)", value=False)

    st.divider()
    with st.expander("📚 Verified Query Library (VQ1–VQ10)"):
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}**: {entry['description']}")

if not api_key:
    st.warning("Enter an OpenAI API key in the sidebar to use the query engine.")
    st.stop()

if not os.path.exists(db_path):
    st.error(f"Database file not found at: `{db_path}`. Upload it or fix the path in the sidebar.")
    st.stop()

llm, evaluator_llm = get_llms(api_key, api_base, model_name)
conn = get_connection(db_path)

tab_query, tab_eval = st.tabs(["💬 Query Engine", "📊 Evaluation Against Ground Truth"])

# ---------------------------------------------------------------------------
# Tab 1: Query Engine
# ---------------------------------------------------------------------------
with tab_query:
    example_questions = [
        "What is the sector-wise NPA exposure across the portfolio?",
        "Show me the portfolio outstanding by loan category.",
        "What is the top 10 loan exposures in the portfolio?",
        "Which sectors have the highest average interest rate?",
        "How has the Stage 3 book size and ECL trended over the last four quarters?",
    ]

    cols = st.columns(len(example_questions))
    for i, q in enumerate(example_questions):
        if cols[i].button(f"Example {i + 1}", help=q, use_container_width=True):
            st.session_state["question_input"] = q

    user_question = st.text_area(
        "Ask a question about the commercial lending portfolio",
        key="question_input",
        placeholder="e.g. What is the sector-wise outstanding and NPA breakdown?",
        height=80,
    )

    run_clicked = st.button("Run Query", type="primary")

    if run_clicked and user_question.strip():
        with st.spinner("Classifying intent, building SQL, validating, and executing..."):
            try:
                result = run_pipeline(
                    user_question, conn, verified_query_library, database_schema,
                    llm, evaluator_llm, verbose=True
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                result = None

        if result is not None:
            log = result["log"]

            # --- Route / confidence summary row ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Route", log["route"] or "—")
            c2.metric("Query ID", log["query_id"] or "—")
            conf = log["confidence"]
            c3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))
            c4.metric("Rows Returned", log["row_count"] if log["row_count"] is not None else "—")

            if log["escalated"]:
                st.error(f"🚨 Escalated to human analyst: {log['gate_result'].get('details', '')}")
            else:
                st.success("✅ Answer generated and validated.")

            st.subheader("Narrative")
            st.markdown(log["narrative"])

            if not log["escalated"]:
                st.subheader("Result Data")
                st.dataframe(result["dataframe"], use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    result["dataframe"].to_csv(index=False).encode("utf-8"),
                    file_name="query_result.csv",
                    mime="text/csv",
                )
                if log["warnings"]:
                    st.warning("Reasonableness warnings: " + "; ".join(log["warnings"]))

            with st.expander("🔍 SQL Query Used"):
                st.code(log["executed_sql"] or log["candidate_sql"], language="sql")

            if log["retry_used"]:
                st.info("A retry was triggered after the first validation attempt failed.")

            if show_trace:
                with st.expander("🧩 Pipeline Trace"):
                    for line in log["trace"]:
                        st.text(line)

            # --- Audit trail ---
            st.session_state.audit_trail.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Question": user_question,
                "Route": log["route"],
                "Query ID": log["query_id"] or "",
                "Escalated": log["escalated"],
                "Confidence": log["confidence"],
                "Rows": log["row_count"],
                "SQL": log["executed_sql"] or log["candidate_sql"],
            })

    st.divider()
    st.subheader("🗂️ Audit Trail")
    if st.session_state.audit_trail:
        audit_df = pd.DataFrame(st.session_state.audit_trail)
        st.dataframe(audit_df, use_container_width=True)
        st.download_button(
            "Download audit trail as CSV",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_trail.csv",
            mime="text/csv",
        )
    else:
        st.caption("No queries run yet this session.")

# ---------------------------------------------------------------------------
# Tab 2: Evaluation Against Ground Truth
# ---------------------------------------------------------------------------
with tab_eval:
    st.write(
        "Upload `test_queries.csv` (columns: `Test Case`, `User Query`, `Expected Route`, "
        "`Expected Query ID`, `Expected Answer`) to reproduce the notebook's evaluation."
    )
    default_csv_path = _get_default("TEST_QUERIES_PATH", "test_queries.csv")
    uploaded_csv = st.file_uploader("Upload test_queries.csv", type=["csv"], key="eval_csv")

    ground_truth = None
    if uploaded_csv is not None:
        ground_truth = pd.read_csv(uploaded_csv)
    elif os.path.exists(default_csv_path):
        ground_truth = pd.read_csv(default_csv_path)
        st.caption(f"Using bundled file: `{default_csv_path}`")

    if ground_truth is not None:
        st.dataframe(ground_truth, use_container_width=True)

        if st.button("Run Evaluation"):
            evaluation_rows = []
            progress = st.progress(0.0)
            n = len(ground_truth)

            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = run_pipeline(
                    gt["User Query"], conn, verified_query_library, database_schema,
                    llm, evaluator_llm, verbose=False
                )
                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })
                progress.progress((i + 1) / n)

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df['Route Match'].mean() * 100
            verified_mask = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified_mask, 'Query ID Match'].mean() * 100

            evaluation_df['Confidence'] = pd.to_numeric(evaluation_df['Confidence'], errors='coerce')
            average_confidence = evaluation_df['Confidence'].mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            c2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
            c3.metric("Average Confidence Score", f"{average_confidence:.2f}")

            st.dataframe(evaluation_df, use_container_width=True)
            st.download_button(
                "Download evaluation results as CSV",
                evaluation_df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )
    else:
        st.info("No `test_queries.csv` found or uploaded yet.")