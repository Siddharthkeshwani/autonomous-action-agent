from sqlglot import Dialect, exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import TokenType


POSTGRES = Dialect.get_or_raise("postgres")


def validate_sql(sql: str) -> str:
    """Return a single read-only PostgreSQL SELECT with a default row limit."""
    try:
        tokens = POSTGRES.tokenize(sql)
        if not tokens or tokens[0].token_type not in (TokenType.SELECT, TokenType.WITH):
            raise ValueError("Only one read-only SELECT statement is allowed.")
        if any(token.token_type == TokenType.SEMICOLON for token in tokens[:-1]):
            raise ValueError("Only one read-only SELECT statement is allowed.")
        if tokens[-1].token_type == TokenType.SEMICOLON:
            tokens.pop()
        statements = POSTGRES.parser().parse(tokens, sql)
    except (ParseError, TokenError, RecursionError):
        raise ValueError("Invalid PostgreSQL query.") from None

    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("Only one read-only SELECT statement is allowed.")

    query = statements[0]
    if query.find(exp.DML, exp.DDL, exp.Command, exp.Into, exp.Lock):
        raise ValueError("Only read-only SELECT statements are allowed.")
    if any(not isinstance(cte.this, exp.Query) for cte in query.find_all(exp.CTE)):
        raise ValueError("Only read-only queries are allowed in CTEs.")

    if query.args.get("limit") is None:
        query = query.limit(500)
    return query.sql(dialect="postgres", comments=False)
