# PhiFlow Error Recovery Guide

## E001_UNEXPECTED_TOKEN
**What it means:** The parser found a token that is not valid at the current expression or statement boundary.
**Common causes:** Starting a statement with a keyword that only works in another context, or malformed expression ordering.
**Fix:** Replace the invalid token with a legal statement starter (`let`, function call, `resonate`) or correct the surrounding syntax.
**Example:**

```phi
// before
intention "x" {
    state something
}

// after
intention "x" {
    let value = some_function()
    resonate value
}
```

## E002_UNEXPECTED_EOF
**What it means:** The parser reached end-of-file before completing an open block or construct.
**Common causes:** Missing `}` at the end of `intention`, `witness`, `if`, or function blocks.
**Fix:** Add the missing closing brace/token where the block started.
**Example:**

```phi
// before
intention "name" {
    resonate 1.0

// after
intention "name" {
    resonate 1.0
}
```

## E003_EXPECTED_TOKEN
**What it means:** A required token was missing, and the parser reports what it expected.
**Common causes:** Missing punctuation such as `:`, `)`, `->`, `,`, or braces in declarations.
**Fix:** Insert the expected token near the reported location.
**Example:**

```phi
// before
function f(x Number) -> Number {
    return x
}

// after
function f(x: Number) -> Number {
    return x
}
```

## E004_UNEXPECTED_CHAR
**What it means:** The lexer encountered a character that is not part of PhiFlow syntax.
**Common causes:** Typing unsupported operators or stray symbols.
**Fix:** Remove the character or replace it with a supported operator.
**Example:**

```phi
// before
let x = 5 @ 3

// after
let x = 5.0 + 3.0
```

## E005_UNDECLARED_VARIABLE
**What it means:** Runtime/evaluation attempted to use a variable name that was never declared in scope.
**Common causes:** Typo in identifier, missing `let` binding, or scope mismatch.
**Fix:** Declare the variable before use in the same or parent scope.
**Example:**

```phi
// before
resonate y

// after
let y = 1.618
resonate y
```
