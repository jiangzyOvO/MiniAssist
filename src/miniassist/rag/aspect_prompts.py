"""Policy-level C prompts. No question IDs, gold answers or DJI fact constants."""
from .prompts import COMMON, GENERATE

PLAN = COMMON + '''
只读用户原问题，建立回答政策必需的aspect清单；此时没有检索文本。
每个aspect对应一个可以独立核验的子问题，使用稳定、简短的aspect_id。
按实际问题采用以下方面，不要机械地全部添加：
- 操作与预期不符：用户已做动作的正常效果、达到目标的正确操作、能否据此判故障；
  只有用户要求定位个体原因且确实阻塞时，才增加需要用户提供的排查信息。
- 版本/文档差异：所引资料自身的版本/日期依据、各自记载的结论、该条目的使用前提、
  是否存在改变结论的支持记录；不臆造新旧因果，不要求文档预测未来。
- 型号：完整型号和适用组合、相似名称的区分；未知设备需追问，已知组合先核对资料。
- 数值/联合条件：每个用户值、对应完整阈值、逐项比较及能否执行；不能把失败原因的反面当充分条件。
- 多问：保留每一问的独立回答义务；不要只拆用户前提，漏了实际提问。
category=boundary用于判断能否保证/诊断、未来计划、维修价格/保修或设备私有记录等信息边界，
不用为了说“不能据此判断”而要求官方逐字写出用户个案；该类由系统边界说明承接。
category=user_info只用于必须向用户追问的信息，不是让检索系统查出个体状态。
其他category需要可检索的官方证据支持。结论可由证据和用户条件合理推导，不能要求原文照抄问句。
required=true表示回答原问题必需；可选后续诊断不要设为必需。不能为了凑数量加入完整教程或无关规格。
不要在requirement内写你猜测的答案、阈值、日期或未给出的型号。reason说明该方面为何是用户需求。
最多10项，通常3到6项。boundary选最贴合原问题的信息边界；可答规则不因unknown_fault而阻断。
'''

COVERAGE = COMMON + '''
对冻结的aspect_plan逐项标覆盖，原样返回全部aspect_id且各出现一次，不增删或弱化requirement。
supported必须附真正支撑该方面的逐字引文，校对对象/型号、状态、完整条件和版本范围。
只有步骤没有单个动作效果、只有功能没有使用前提、只有单个阈值没有联合要求，都不算对应方面完整。
版本元数据/章节标签只是定位提示，不能代替正文版本证据。没有出现某型号不等于禁止兼容。
引文可来自不同块共同支持推理；category=conclusion/numeric可以结合用户给定值比较，detail说明推理。
存在冲突且没有足够范围依据消解时为conflict，不能从日期元数据猜先后。
missing是资料库可能有但当前尚未找到的事实；conflict是仍须查明适用范围的冲突。
对每个required的missing/conflict，若remaining_queries>0，给出一条仅针对该缺口的自然语言query。
查询只使用用户信息、已给原文和待查概念，不猜答案/阈值；相近缺口允许使用相同query以合并。
只有missing/conflict可填query；supported、user_gap、unavailable的query必须为空。
category=user_info且用户未提供必要个体信息时用user_gap并给具体question_to_user，不能补查代替问用户。
category=boundary用unavailable说明不能由公开文档确定的范围；这不是强制整题拒答。
若用户信息aspect实际已知，在detail指出用户已提供、用unavailable且不追问，不伪造文档引文。
其他事实不得仅以“没有写、无法保证”标supported；应有直接引文或可核查推理前提。
所有引文仅复制citable_spans连续文本，不引用元数据，不拼接跨单元格文字，不计算span_id；唯一引文可省略span_id。
预算耗尽仍保持missing/conflict；不要为消除缺口而编造支持。detail不是新的事实来源。
'''

GENERATE_C = GENERATE + '''
本策略没有自动附加critical_conditions；所有必要限定应在claims正文中准确写出。
required_aspects是用户回答义务，aspect_coverage是模型检查结果而非事实来源。
对每个supported的必需aspect都要在正文明确回答并引用原文，不只留在来源脚注。
对missing/conflict不编造答案；有据部分先答，明确尚未核实的具体方面。
boundary、follow_up由系统呈现，不为信息边界制造引文。不要添加与问题无关的设备说明。
若output_audit非空，这是一次受预算限制的修订：补上有证据但漏写的必需项，纠正无支撑/矛盾/无关claim；
保持原本正确的部分，不引入新事实或扩大查询，不因删掉问题句而假装已回答必需aspect。
'''

AUDIT = COMMON + '''
你检查即将发布的完整claims，不能参考标准答案或宣称人审通过。
逐一返回每个claim的0基索引且恰好一次；逐一返回冻结aspect_plan的全部aspect_id且恰好一次。
claim supported需要其实际引用支撑全句：引文真实还不够，对象/型号、版本、日期、单位、上下界、
联合条件、因果关系和用途都要一致。合理推论可使用用户已给条件，但不能推断未经核实的个体状态。
直接有反证为contradicted；引文未充分支撑为unsupported；即使真实但和原问题无关为irrelevant。
自动条件若有也必须同样检查。不能因同词（短按/电量等）就把遥控器条件用于飞行器。
仅有来源元数据、列表未提及、失败条件反面、同次发布组合，都不构成额外能力/最低兼容/可执行保证。
aspect addressed须正文直接覆盖该需求、claim_indices列出承载的全部句子；有证据但没写仍为missing。
若supported的必需aspect只写“无法确认”或仅出现在引用脚注，判missing。
appropriately_limited仅用于原始coverage确为missing/conflict/user_gap/unavailable且答案/系统边界明确说明限制；
不能把原本可回答的aspect改成“无法确认”来通过检查。
boundary_text、follow_up、missing_notice也随答案发布；可用于核对边界/追问方面，不能代替事实证据。
不要修订文本，只输出每项结论及具体理由；系统按检查结果决定一次修订或降级。
'''
