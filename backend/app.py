import re
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sympy import symbols, diff, latex, simplify, trigsimp, sec, csc, cot, E, pi, log, asin, acos, atan
from sympy import acot, asec, acsc, Abs, sinh, cosh, tanh, sech, csch, coth, floor, ceiling
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

app = Flask(__name__)
CORS(app)

x = symbols("x")

local_dict = {
    "x": x,
    "e": E,
    "E": E,
    "pi": pi,
    "Pi": pi,
    "ln": log,
    "log": log,
    "sec": sec,
    "csc": csc,
    "cosec": csc,
    "cot": cot,
    "abs": Abs,
    "Abs": Abs,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
    "sech": sech,
    "csch": csch,
    "coth": coth,
    "floor": floor,
    "ceil": ceiling,
    "ceiling": ceiling,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "arcsin": asin,
    "arccos": acos,
    "arctan": atan,
    "acot": acot,
    "arccot": acot,
    "asec": asec,
    "arcsec": asec,
    "acsc": acsc,
    "arccsc": acsc,
}

transformations = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)


def normalize_expression(expr: str) -> str:
    expr = expr.strip()
    # Absolute value input support: |x| -> abs(x)
    expr = re.sub(r'\|\s*([^|]+?)\s*\|', r'abs(\1)', expr)
    expr = re.sub(r"(?i)cosec", "csc", expr)
    # Treat short inverse trig names the same as arc- names.
    expr = re.sub(r"(?i)(?<!arc)acos", "arccos", expr)
    expr = re.sub(r"(?i)(?<!arc)asin", "arcsin", expr)
    expr = re.sub(r"(?i)(?<!arc)atan", "arctan", expr)
    expr = re.sub(r"(?i)(?<!arc)acot", "arccot", expr)
    expr = re.sub(r"(?i)(?<!arc)asec", "arcsec", expr)
    expr = re.sub(r"(?i)(?<!arc)acsc", "arccsc", expr)

    # Protect inverse trig names so the generic sin/cos/tan regexes do not
    # split arccos into arc*cos, arcsin into arc*sin, etc.
    expr = re.sub(r"(?i)arccos", "ZZQ1", expr)
    expr = re.sub(r"(?i)arcsin", "ZZQ2", expr)
    expr = re.sub(r"(?i)arctan", "ZZQ3", expr)
    expr = re.sub(r"(?i)arcsec", "ZZQ4", expr)
    expr = re.sub(r"(?i)arccsc", "ZZQ5", expr)
    expr = re.sub(r"(?i)arccot", "ZZQ6", expr)
    # Handle protected inverse trig names attached to variables/numbers.
    expr = re.sub(r"(ZZQ1)([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ2)([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ3)([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ4)([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ5)([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ6)([a-zA-Z0-9]+)", r"\1(\2)", expr)
    # Support protected inverse trig powers: arccos^3x -> arccos(x)^3
    expr = re.sub(r"(ZZQ[1-6])\^(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)
    expr = re.sub(r"(ZZQ[1-6])(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)

    # Support trig powers without brackets: sin^2x -> sin(x)^2
    expr = re.sub(r"(?i)\b(sin|cos|tan|sec|csc|cot)\^(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)
    expr = re.sub(r"(?i)\b(sin|cos|tan|sec|csc|cot)(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)
    # Support inverse trig powers without brackets: arcsin^2x -> arcsin(x)^2
    expr = re.sub(r"(?i)\b(arcsin|arccos|arctan|arccot|arcsec|arccsc|asin|acos|atan|acot|asec|acsc)\^(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)
    expr = re.sub(r"(?i)\b(arcsin|arccos|arctan|arccot|arcsec|arccsc|asin|acos|atan|acot|asec|acsc)(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)
    # Support log/ln powers only when ^ is explicitly used: ln^2x -> ln(x)^2, log^3x -> log(x)^3
    expr = re.sub(r"(?i)\b(ln|log)\^(\d+)\s*([a-zA-Z0-9]+)", r"\1(\3)^\2", expr)
    # xsinx -> x*sinx, 2cosx -> 2*cosx, xlnx -> x*lnx
    expr = re.sub(r"(?i)([0-9a-zA-Z\)])(arcsin|arccos|arctan|arccot|arcsec|arccsc|asin|acos|atan|acot|asec|acsc|sinh|cosh|tanh|sech|csch|coth|sin|cos|tan|sec|csc|cosec|cot|abs|floor|ceiling|ceil|log|ln|exp|sqrt)", r"\1*\2", expr)
    expr = re.sub(r"(?i)([0-9a-zA-Z\)])(arccot|arcsec|arccsc|acot|asec|acsc)", r"\1*\2", expr)
    expr = re.sub(r"(?i)\bsqrt\s*([a-zA-Z0-9]+)", r"sqrt(\1)", expr)
    # sinx -> sin(x), lnx -> ln(x), cos2x -> cos(2x)
    # Do not change calls that already use parentheses like sin(x)
    expr = re.sub(r"(?i)\b(arcsin|arccos|arctan|arccot|arcsec|arccsc|asin|acos|atan|acot|asec|acsc|sinh|cosh|tanh|sech|csch|coth|sin|cos|tan|sec|csc|cosec|cot|abs|floor|ceiling|ceil|log|ln|exp|sqrt)\s*([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(?i)\b(arccot|arcsec|arccsc|acot|asec|acsc)\s*([a-zA-Z0-9]+)", r"\1(\2)", expr)
    # Handle protected inverse trig names followed by a space and an argument.
    expr = re.sub(r"(ZZQ1)\s+([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ2)\s+([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ3)\s+([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ4)\s+([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ5)\s+([a-zA-Z0-9]+)", r"\1(\2)", expr)
    expr = re.sub(r"(ZZQ6)\s+([a-zA-Z0-9]+)", r"\1(\2)", expr)

    # Restore protected inverse trig names.
    expr = expr.replace("ZZQ1", "arccos")
    expr = expr.replace("ZZQ2", "arcsin")
    expr = expr.replace("ZZQ3", "arctan")
    expr = expr.replace("ZZQ4", "arcsec")
    expr = expr.replace("ZZQ5", "arccsc")
    expr = expr.replace("ZZQ6", "arccot")
    return expr

def convert_log_base_syntax(expr: str):
    expr = re.sub(r'(?i)log_([0-9A-Za-z]+)\s*\(([^()]+)\)', r'log(\2,\1)', expr)
    expr = re.sub(r'(?i)log([0-9]+)\s*\(([^()]+)\)', r'log(\2,\1)', expr)
    expr = re.sub(r'(?i)log_([0-9A-Za-z]+)\s+([A-Za-z0-9]+)', r'log(\2,\1)', expr)
    expr = re.sub(r'(?i)log([0-9]+)\s+([A-Za-z0-9]+)', r'log(\2,\1)', expr)
    expr = re.sub(r'(?i)log_([0-9]+)([A-Za-z])', r'log(\2,\1)', expr)
    expr = re.sub(r'(?i)log([0-9]+)([A-Za-z0-9]+)', r'log(\2,\1)', expr)

    return expr

def format_input_latex(expr: str) -> str:
    latex_expr = expr.strip()
    if not latex_expr:
        return "x"

    latex_expr = latex_expr.replace("\\", r"\textbackslash ")
    latex_expr = re.sub(r'([{}#%&])', r'\\\1', latex_expr)
    latex_expr = latex_expr.replace("*", " ")
    latex_expr = re.sub(r'(?i)\bcosec', 'csc', latex_expr)

    # Display absolute value nicely in LaTeX.
    latex_expr = re.sub(r'\|\s*([^|]+?)\s*\|', r'\\left|\1\\right|', latex_expr)

    # Display hyperbolic and common utility functions nicely in LaTeX.
    latex_expr = re.sub(r'(?i)\bsinh\s*\(([^()]+)\)', r'\\sinh\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcosh\s*\(([^()]+)\)', r'\\cosh\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\btanh\s*\(([^()]+)\)', r'\\tanh\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bsech\s*\(([^()]+)\)', r'\\operatorname{sech}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcsch\s*\(([^()]+)\)', r'\\operatorname{csch}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcoth\s*\(([^()]+)\)', r'\\coth\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bfloor\s*\(([^()]+)\)', r'\\lfloor \1 \\rfloor', latex_expr)
    latex_expr = re.sub(r'(?i)\b(?:ceil|ceiling)\s*\(([^()]+)\)', r'\\lceil \1 \\rceil', latex_expr)

    latex_expr = re.sub(r'(?i)\bsinh\s*([A-Za-z0-9]+)', r'\\sinh\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcosh\s*([A-Za-z0-9]+)', r'\\cosh\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\btanh\s*([A-Za-z0-9]+)', r'\\tanh\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bsech\s*([A-Za-z0-9]+)', r'\\operatorname{sech}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcsch\s*([A-Za-z0-9]+)', r'\\operatorname{csch}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcoth\s*([A-Za-z0-9]+)', r'\\coth\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bfloor\s*([A-Za-z0-9]+)', r'\\lfloor \1 \\rfloor', latex_expr)
    latex_expr = re.sub(r'(?i)\b(?:ceil|ceiling)\s*([A-Za-z0-9]+)', r'\\lceil \1 \\rceil', latex_expr)

    # Display trig functions nicely in LaTeX.
    latex_expr = re.sub(r'(?i)\barccos\s*\(([^()]+)\)', r'\\arccos\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barcsin\s*\(([^()]+)\)', r'\\arcsin\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barctan\s*\(([^()]+)\)', r'\\arctan\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barccot\s*\(([^()]+)\)', r'\\operatorname{arccot}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barcsec\s*\(([^()]+)\)', r'\\operatorname{arcsec}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barccsc\s*\(([^()]+)\)', r'\\operatorname{arccsc}\\left(\1\\right)', latex_expr)

    latex_expr = re.sub(r'(?i)\bsin(?!h)\s*\(([^()]+)\)', r'\\sin\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcos(?!h)\s*\(([^()]+)\)', r'\\cos\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\btan(?!h)\s*\(([^()]+)\)', r'\\tan\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bsec(?!h)\s*\(([^()]+)\)', r'\\sec\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcsc(?!h)\s*\(([^()]+)\)', r'\\csc\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcot(?!h)\s*\(([^()]+)\)', r'\\cot\\left(\1\\right)', latex_expr)

    latex_expr = re.sub(r'(?i)\barccos\s*([A-Za-z0-9]+)', r'\\arccos\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barcsin\s*([A-Za-z0-9]+)', r'\\arcsin\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barctan\s*([A-Za-z0-9]+)', r'\\arctan\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barccot\s*([A-Za-z0-9]+)', r'\\operatorname{arccot}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barcsec\s*([A-Za-z0-9]+)', r'\\operatorname{arcsec}\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\barccsc\s*([A-Za-z0-9]+)', r'\\operatorname{arccsc}\\left(\1\\right)', latex_expr)

    latex_expr = re.sub(r'(?i)\bsin(?!h)\s*([A-Za-z0-9]+)', r'\\sin\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcos(?!h)\s*([A-Za-z0-9]+)', r'\\cos\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\btan(?!h)\s*([A-Za-z0-9]+)', r'\\tan\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bsec(?!h)\s*([A-Za-z0-9]+)', r'\\sec\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcsc(?!h)\s*([A-Za-z0-9]+)', r'\\csc\\left(\1\\right)', latex_expr)
    latex_expr = re.sub(r'(?i)\bcot(?!h)\s*([A-Za-z0-9]+)', r'\\cot\\left(\1\\right)', latex_expr)

    # log_4(x) -> log base 4
    latex_expr = re.sub(
        r'(?i)log_([0-9A-Za-z]+)\s*\(([^()]+)\)',
        r'\\log_{\1}\\left(\2\\right)',
        latex_expr
    )

    # log_4x -> log base 4 of x
    latex_expr = re.sub(
        r'(?i)log_([0-9]+)([A-Za-z])',
        r'\\log_{\1}\\left(\2\\right)',
        latex_expr
    )

    # log_4 x -> log base 4 of x
    latex_expr = re.sub(
        r'(?i)log_([0-9A-Za-z]+)\s+([A-Za-z0-9]+)',
        r'\\log_{\1}\\left(\2\\right)',
        latex_expr
    )

    # lnx -> \ln(x)
    latex_expr = re.sub(
        r'(?i)\bln\s*([A-Za-z0-9]+)',
        r'\\ln\\left(\1\\right)',
        latex_expr
    )

    # plain log4x = ln(4x), NOT base log
    latex_expr = re.sub(
        r'(?i)\blog([0-9]+)([A-Za-z])',
        r'\\ln\\left(\1\2\\right)',
        latex_expr
    )

    # plain log(x) = ln(x)
    latex_expr = re.sub(
        r'(?i)\blog\s*\(([^()]+)\)',
        r'\\ln\\left(\1\\right)',
        latex_expr
    )

    # plain logx = ln(x)
    latex_expr = re.sub(
        r'(?i)\blog\s*([A-Za-z0-9]+)',
        r'\\ln\\left(\1\\right)',
        latex_expr
    )

    return latex_expr
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Derivative API is running"})


@app.route("/preview", methods=["POST"])
def preview():
    try:
        data = request.get_json()
        expr = data.get("expression", "").strip()

        if not expr:
            return jsonify({"latex": "x"})

        display_expr = expr
        expr = convert_log_base_syntax(normalize_expression(expr))
        function = parse_expr(expr, transformations=transformations, local_dict=local_dict)

        # Use parsed LaTeX for a reliable live preview.
        # Keep custom formatting only for user-entered base-log notation.
        if re.search(r'(?i)log_[0-9A-Za-z]+', display_expr):
            preview_latex = format_input_latex(display_expr)
        else:
            preview_latex = latex(function).replace(r"\log", r"\ln")

        return jsonify({"latex": preview_latex})
    except Exception:
        raw = request.get_json().get("expression", "x") if request.get_json() else "x"
        return jsonify({"latex": raw.strip() or "x"})

@app.route("/derivative", methods=["POST"])
def derivative():
    try:
        data = request.get_json()
        expr = data.get("expression", "").strip()
        display_expr = expr
        expr = convert_log_base_syntax(normalize_expression(expr))

        if not expr:
            return jsonify({"error": "Expression is required"}), 400

        function = parse_expr(expr, transformations=transformations, local_dict=local_dict)
        result = diff(function, x)

        expression_latex = format_input_latex(display_expr)
        display_result = trigsimp(result).replace(
            lambda e: e.is_Pow and e.base.func == sec and e.exp == 2,
            lambda e: e
        ).replace(
            lambda e: e.is_Pow and e.base.func.__name__ == "cos" and e.exp == -2,
            lambda e: sec(e.base.args[0])**2
        ).replace(
            lambda e: e.is_Pow and e.base.func == csc and e.exp == 2,
            lambda e: e
        ).replace(
            lambda e: e.is_Pow and e.base.func.__name__ == "sin" and e.exp == -2,
            lambda e: csc(e.base.args[0])**2
        ).replace(
            lambda e: e.is_Pow and e.base.func == sech and e.exp == 2,
            lambda e: e
        ).replace(
            lambda e: e.is_Pow and e.base.func.__name__ == "cosh" and e.exp == -2,
            lambda e: sech(e.base.args[0])**2
        ).replace(
            lambda e: e.is_Pow and e.base.func == csch and e.exp == 2,
            lambda e: e
        ).replace(
            lambda e: e.is_Pow and e.base.func.__name__ == "sinh" and e.exp == -2,
            lambda e: csch(e.base.args[0])**2
        )
        derivative_latex = latex(display_result).replace(r"\log", r"\ln")
        derivative_latex = derivative_latex.replace(r"\csc", r"\operatorname{cosec}")
        derivative_latex = derivative_latex.replace(r"\csch", r"\operatorname{cosech}")

        return jsonify({
            "expression": str(function),
            "derivative": str(result),
            "expression_latex": expression_latex,
            "derivative_latex": derivative_latex
        })
    except Exception:
        return jsonify({"error": "Invalid expression"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)