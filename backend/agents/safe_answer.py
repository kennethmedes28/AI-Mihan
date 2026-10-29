"""The fixed reply used when an agent is not confident enough to answer.

No model writes it, so it cannot invent anything. It gives only the safety
steps that are true for any flood, and points to the official sources. The
reply is flagged (``safe_answer=True``) so the app can show it differently,
for example with a warning banner, or log the question for a person to review.

Example:
    return safe_reply()
"""

from agents.flood_types import AgentReply

SAFE_ANSWER = (
    "I am not confident enough to answer that safely, so I will not guess.\n"
    "- If you or someone is in danger, call 911 now.\n"
    "- If water is rising, move to higher ground. Do not walk or drive through floodwater.\n"
    "- Check pagasa.dost.gov.ph and ndrrmc.gov.ph, or contact your barangay or local DRRMO.\n"
    "Hindi ako sapat na sigurado para sagutin ito nang ligtas, kaya hindi ako manghuhula.\n"
    "- Kung nasa panganib kayo o ang iba, tumawag sa 911 ngayon.\n"
    "- Kung tumataas ang tubig, lumipat sa mas mataas na lugar. Huwag tumawid o magmaneho sa baha.\n"
    "- Tingnan ang pagasa.dost.gov.ph at ndrrmc.gov.ph, o makipag-ugnayan sa barangay o lokal na DRRMO."
)


def safe_reply() -> AgentReply:
    """The safety text, flagged as a safe answer."""
    return AgentReply(text=SAFE_ANSWER, safe_answer=True)
