"""DeepSeek JSON-mode adapter. No credentials or provider bodies enter logs."""
import json
import os
from urllib.parse import urlparse
import requests
from pydantic import ValidationError
from . import prompts
from .state import Understanding, Assessment, Answer, AnswerRevision
from .errors import schema_issues


class ModelError(RuntimeError):
    def __init__(self, message, code="model_error", **details):
        super().__init__(message)
        self.diagnostic = {"code": code, **details}


class DeepSeek:
    def __init__(self, settings, api_key=None, session=None):
        self.settings = settings
        self.key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.key:
            raise ModelError("缺少 DEEPSEEK_API_KEY 环境变量")
        parsed = urlparse(settings.base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ModelError("模型接口必须使用不含凭据的 HTTPS 地址")
        self.session = session or requests.Session()
        self.requests_used = 0

    def reset_budget(self):
        self.requests_used = 0

    def call(self, stage, payload):
        self.last_call_diagnostic = {}
        try:
            return self._call_once(stage, payload)
        except ModelError as first:
            issues = first.diagnostic.get("issues", [])
            # One repair budget per node invocation, shared with node-level
            # citation/route recovery. Never retry transport errors or coerce facts.
            def allowed(issue):
                loc = issue['location']
                return (issue['type'] == 'json_invalid' and not loc or
                        stage == 'assess' and issue['type'] == 'literal_error'
                        and len(loc) == 3 and loc[0] == 'checks' and loc[2] == 'status' or
                        issue['type'] == 'string_too_short' and bool(loc) and loc[-1] == 'quote')
            repairable = (first.diagnostic.get('code') == 'schema_invalid'
                          and bool(issues) and all(allowed(i) for i in issues)
                          and not payload.get('citation_repair') and not payload.get('routing_repair'))
            if not repairable:
                raise
            repair = {"attempts": 1, "first_failure": first.diagnostic}
            retry_payload = {**payload, "schema_repair": {
                "issues": issues,
                "instruction": "依据原输入重新输出合法JSON。checks.status只能为supported、knowledge_gap、user_gap、unavailable或not_applicable。普通quote仍必须至少4个非空白字符，逐字复制原文。短表格单元格使用给定table_rows的table_row_id，省略quote和span_id；不要拼接标签、改写或编造证据，不把格式错误当资料不足。"}}
            try:
                result = self._call_once(stage, retry_payload)
            except ModelError as second:
                self.last_call_diagnostic = {**repair, "status": "failed", "repair_failure": second.diagnostic}
                raise ModelError("模型结构修复失败", "schema_repair_failed",
                                 **self.last_call_diagnostic) from None
            self.last_call_diagnostic = {**repair, "status": "validated"}
            return result

    def _call_once(self, stage, payload):
        stages = {
            "understand": (Understanding, prompts.UNDERSTAND),
            "assess": (Assessment, prompts.ASSESS),
            "generate": (Answer, prompts.GENERATE),
            "review_answer": (AnswerRevision, prompts.REVIEW_ANSWER),
        }
        if stage.startswith('c_'):
            from .aspect_schema import AspectPlan, CoverageReport, OutputAudit
            from . import aspect_prompts
            stages.update(c_plan=(AspectPlan, aspect_prompts.PLAN),
                          c_cover=(CoverageReport, aspect_prompts.COVERAGE),
                          c_generate=(Answer, aspect_prompts.GENERATE_C),
                          c_audit=(OutputAudit, aspect_prompts.AUDIT))
        schema, prompt = stages[stage]
        output_schema = schema.model_json_schema()
        evidence_ids = [d["chunk_id"] for d in payload.get("evidence", [])]
        if stage in {"generate", "review_answer"}:
            output_schema["$defs"]["Citation"]["properties"]["evidence_id"]["enum"] = evidence_ids
        elif stage == "assess":
            output_schema["properties"]["evidence_ids"]["items"]["enum"] = evidence_ids
            output_schema["required"] += ["checks", "critical_conditions"]
            output_schema["properties"]["checks"]["minItems"] = 1
            output_schema["$defs"]["EvidenceCheck"]["required"].append("required_for_answer")
            output_schema["$defs"]["Citation"]["properties"]["evidence_id"]["enum"] = evidence_ids
        if stage in {"assess", "generate", "review_answer"}:
            span_ids = list(dict.fromkeys(s['span_id'] for d in payload.get('evidence', [])
                                          for s in d.get('citable_spans', []) if s.get('span_id')))
            if span_ids:
                citation_schema = output_schema['$defs']['Citation']
                citation_schema['properties']['span_id'] = {
                    'anyOf': [{'type': 'string', 'enum': span_ids}, {'type': 'null'}],
                    'description': 'Optional; omit for uniquely matching quotations. Never calculate offsets.'}
        if stage.startswith('c_') and 'Citation' in output_schema.get('$defs', {}):
            output_schema['$defs']['Citation']['properties']['evidence_id']['enum'] = evidence_ids
        if 'Citation' in output_schema.get('$defs', {}):
            row_ids = [r['table_row_id'] for d in payload.get('evidence', []) for r in d.get('table_rows', [])]
            row_schema = {'anyOf': [{'type': 'string', 'enum': row_ids}, {'type': 'null'}]} if row_ids else {'type': 'null'}
            output_schema['$defs']['Citation']['properties']['table_row_id'] = row_schema
            output_schema['$defs']['Citation']['oneOf'] = [
                {'required': ['quote'], 'properties': {'quote': {'type': 'string'}, 'table_row_id': {'type': 'null'}}},
                {'required': ['table_row_id'], 'properties': {'table_row_id': {'type': 'string'},
                    'quote': {'type': 'null'}, 'span_id': {'type': 'null'}}}]
        if self.requests_used >= self.settings.max_model_requests:
            raise ModelError('本轮模型请求预算已耗尽', 'model_request_budget_exhausted')
        self.requests_used += 1
        try:
            result = self.session.post(
                self.settings.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + self.key},
                json={"model": self.settings.model, "stream": False,
                      "thinking": {"type": "disabled"},
                      "max_tokens": 4096, "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": prompt +
                                    "\n严格使用以下 JSON Schema 的英文字段名和类型，不翻译字段名，不增加其他字段：\n" +
                                    json.dumps(output_schema, ensure_ascii=False)},
                                   {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
                timeout=(10, self.settings.timeout), allow_redirects=False)
            if result.status_code != 200:
                raise ModelError(f"模型接口 HTTP {result.status_code}", "provider_http",
                                 http_status=int(result.status_code))
            choice = result.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ModelError("模型输出未正常结束", "output_not_finished")
            validated = schema.model_validate_json(choice["message"]["content"]).model_dump(exclude_none=True)
            if stage == "assess" and not validated["checks"]:
                raise ModelError("缺少逐项证据检查", "empty_evidence_checks")
            return validated
        except ModelError:
            raise
        except ValidationError as exc:
            content = choice['message']['content']
            issues = schema_issues(exc)
            metadata = {'content_chars': len(content), 'finish_reason': choice['finish_reason']}
            try:
                parsed = json.loads(content)
                # Record lengths only at known validation paths, never values.
                lengths = []
                for issue in issues:
                    if issue['type'] == 'string_too_short' and issue['location'][-1:] == ['quote']:
                        value = parsed
                        for part in issue['location']:
                            value = value[part]
                        if isinstance(value, str):
                            lengths.append({'location': issue['location'], 'raw_chars': len(value),
                                            'nonwhitespace_chars': len(''.join(value.split()))})
                if lengths:
                    metadata['quote_lengths'] = lengths
            except json.JSONDecodeError as error:
                metadata['json_position'] = {'offset': error.pos, 'line': error.lineno, 'column': error.colno}
            except (TypeError, KeyError, IndexError):
                pass
            raise ModelError("模型结构校验失败", "schema_invalid",
                             issues=issues, **metadata) from None
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelError("模型调用或响应解析失败", "provider_or_response_error",
                             exception_type=type(exc).__name__) from None
