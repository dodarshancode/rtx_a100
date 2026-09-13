#!/usr/bin/env python3
"""
check_azure_model.py - find out exactly which model sits behind an Azure OpenAI
deployment and which request features it accepts.

    pip install "openai>=1.60"
    export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
    export AZURE_OPENAI_API_KEY="..."
    python check_azure_model.py <deployment-name> [<deployment-name> ...]

Sends 4 tiny requests per deployment (no document content).
"""
import json
import os
import sys

from openai import AzureOpenAI

API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=API_VERSION,
)

MSG = [{"role": "user", "content": "Return the value 1342,00 V converted to kV as a number."}]
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "probe",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"value_kv": {"type": "number"}, "unit": {"type": "string"}},
            "required": ["value_kv", "unit"],
            "additionalProperties": False,
        },
    },
}


def attempt(label, deployment, **kw):
    try:
        r = client.chat.completions.create(model=deployment, messages=MSG,
                                           max_completion_tokens=2000, **kw)
        txt = (r.choices[0].message.content or "").strip().replace("\n", " ")
        print(f"  [OK ] {label:28} model={r.model!r}  -> {txt[:80]}")
        return r
    except Exception as e:  # report the API's own error text
        msg = getattr(e, "message", str(e))
        print(f"  [ERR] {label:28} {msg[:160]}")
        return None


for dep in sys.argv[1:] or [os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")]:
    print(f"\nDeployment: {dep}   (api_version={API_VERSION})")
    attempt("plain", dep)
    attempt("temperature=0", dep, temperature=0)
    r = attempt("json_schema strict", dep, response_format=SCHEMA)
    if r is not None:
        try:
            json.loads(r.choices[0].message.content)
            print("        json_schema output parsed OK")
        except Exception:
            print("        json_schema output did NOT parse")
    attempt("reasoning_effort=low", dep, reasoning_effort="low")

print("\nReading the result: 'model=' shows the real model+version behind the deployment.\n"
      "Reasoning models (gpt-5, gpt-5-mini, gpt-5.x) reject temperature but accept reasoning_effort;\n"
      "non-reasoning chat models do the opposite. json_schema [OK] means structured outputs work.")
