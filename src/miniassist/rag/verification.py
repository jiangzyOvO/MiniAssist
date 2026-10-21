"""Narrow, auditable query facets from literal user text, without QA labels.

These are development-tested Chinese heuristics, not a general intent parser.
They reserve queries inside the existing one-round/two-query budget. Product
names remain in applicability slots; action queries use the device category to
avoid repeatedly ranking only FAQ chunks containing the product name.
"""
import re
from .context import user_context

ACTION = re.compile(r'(短按|长按|单击|双击|按住|点击)([^，。？！?！；;\n]{1,16}?)(?:没有|未能|没能|不能|却没|没)([^，。？！?！；;\n]{1,10})')
DOCUMENT_VERSION = re.compile(
    r'(?<![A-Za-z0-9_.])(?:(?P<before>v\d+(?:\.\d+)+)\s*(?:的\s*)?(?:用户)?(?:手册|说明书)|'
    r'(?:手册|说明书)\s*(?:版本[为是]?\s*)?(?P<after>v\d+(?:\.\d+)+))(?![A-Za-z0-9_.])', re.I)


def verification_queries(state):
    messages = user_context(state)['utterances']
    text = '\n'.join(messages)
    slots = state.get('slots', {})
    # Never establish applicability from an invented slot value.
    def stated(name):
        value = re.sub(r'^DJI\s+', '', slots.get(name, ''), flags=re.I).strip()
        return value if value and value.casefold() in text.casefold() else ''

    plans = []
    match = ACTION.search(text)
    if match:
        verb, target, goal = match.groups()
        category = ('遥控器' if '遥控器' in text or stated('controller') else
                    '飞行器' if '飞行器' in text or stated('aircraft') else '')
        action = verb + target.strip()
        # Conventional synonym, not an answer or an expected numerical value.
        target = target.strip().replace('电源键', '电源按键')
        plans += [{'purpose': '核对用户已执行动作本身的作用', 'trigger': match.group(0),
                   'query': f'{category} {action} 作用 操作说明'.strip()},
                  {'purpose': '核对达到用户所述目标的完整操作', 'trigger': match.group(0),
                   'query': f'{category} {verb} {target} {goal.strip()}'.strip()}]
    # Bind the version and document noun within the same clause. A firmware/App
    # version elsewhere in the question is not a manual-version assertion.
    for clause in re.split(r'[，。？！?！；;\n]', text):
        if not re.search(r'手册|说明书', clause):
            continue
        version = DOCUMENT_VERSION.search(clause)
        if version and not re.search(r'(?:App|固件|软件)\s*$', clause[:version.start()], re.I):
            plans.append({'purpose': '核对所引文档的正文版本依据', 'trigger': clause,
                          'query': f"{stated('aircraft') or stated('controller')} {version.group('before') or version.group('after')} 用户手册 封面 版本".strip()})
            break
    return plans[:2]
