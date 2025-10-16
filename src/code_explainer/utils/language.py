"""Language detection utilities."""


def detect_language(code: str) -> str:
    """Very simple language detector for code snippets.
    Returns one of: python, javascript, java, cpp.
    """
    code_l = code.lower()
    if "#include" in code_l or "std::" in code or ";" in code and "using namespace" in code_l:
        return "cpp"
    if "public static void main" in code_l or "class " in code and "system.out" in code_l:
        return "java"
    if "function " in code_l or "=>" in code or "console.log" in code_l:
        return "javascript"
    return "python"