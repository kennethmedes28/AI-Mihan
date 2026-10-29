"""Flood media agent: checks a photo or video of a street for flood water.

Uses SAM 3.1 (see ``video/``) to find flood water, people and vehicles, then writes a short
bilingual summary with fixed wording. No chat model is used, so the same picture always gives the
same answer and a made-up detail cannot slip in.

What it can and cannot say:

    * It reports how much of the picture is covered by water and how many people or vehicles stand in it.
    * It never says a road is safe or gives a water depth. A picture can be old, edited or from
      another place, so the result is unverified evidence and never overrides PAGASA or NDRRMC.

Needs the ``MUSE_API`` environment variable (the same key as the chat models).

Example:
    reply = answer_flood_media_question(ImageInput(path="street.jpg"), ChatContext(location=place))
    print(reply.text)
"""

import logging
import os

from agents.flood_types import AgentReply, ChatContext
from video.flood_scene import LEVEL_NONE, LEVEL_SOME, FloodSceneReport, analyze_flood_media
from video.sam_client import ImageInput, SamError, VideoInput

logger = logging.getLogger(__name__)

NO_KEY_ANSWER = (
    "I cannot check pictures right now.\n"
    "Hindi ko ma-check ang mga larawan sa ngayon.\n"
    "If someone is in danger, call 911."
)
FAILED_ANSWER = (
    "I could not check that picture or video. Please try again, or describe what you see.\n"
    "Hindi ko ma-check ang larawan o video. Subukan ulit, o ilarawan ang nakikita mo.\n"
    "If someone is in danger, call 911."
)
SAFETY_TEXT = (
    "This is an automatic check of the picture or video only. It cannot show how deep the water is or whether a road is "
    "passable, and the picture may be old. Do not walk or drive through flood water. If someone is in danger, call 911.\n"
    "Awtomatikong pagsusuri lang ito ng larawan o video. Hindi nito matutukoy ang lalim ng baha o kung madadaanan ang kalsada, "
    "at maaaring luma ang larawan. Huwag lumusong o magmaneho sa baha. Kung may nasa panganib, tumawag sa 911."
)


def answer_flood_media_question(
    media: ImageInput | VideoInput, context: ChatContext, api_key: str | None = None
) -> AgentReply:
    """Analyse a photo or video and describe the flood water in it."""
    api_key = api_key or os.getenv("MUSE_API")
    if not api_key:
        return AgentReply(text=NO_KEY_ANSWER)

    try:
        report = analyze_flood_media(media, api_key)
    except (SamError, OSError, ValueError) as error:  # OSError covers network problems and unreadable files
        logger.error("Flood media check failed: %s", error)
        return AgentReply(text=FAILED_ANSWER)

    return AgentReply(text=format_report(report, context))


def format_report(report: FloodSceneReport, context: ChatContext) -> str:
    """Write the report as English and Filipino text for the user."""
    kind = "video" if report.media_kind == "video" else "picture"
    kind_fil = "video" if report.media_kind == "video" else "larawan"

    if report.level == LEVEL_NONE:
        english = f"I did not find clear flood water in the {kind}."
        filipino = f"Walang malinaw na baha sa {kind_fil}."
    else:
        extent = "some" if report.level == LEVEL_SOME else "a large part"
        extent_fil = "bahagi" if report.level == LEVEL_SOME else "malaking bahagi"
        english = f"Flood water covers {extent} of the {kind} (about {report.water_coverage:.0%})."
        filipino = f"May baha sa {extent_fil} ng {kind_fil} (mga {report.water_coverage:.0%})."

    english_lines, filipino_lines = [english], [filipino]
    if report.people:
        english_lines.append(f"{report.people} people are visible, {report.people_in_water} of them standing in water.")
        filipino_lines.append(f"May {report.people} taong nakikita, {report.people_in_water} sa kanila ay nasa baha.")
    if report.vehicles:
        english_lines.append(f"{report.vehicles} vehicles are visible, {report.vehicles_in_water} of them in water.")
        filipino_lines.append(f"May {report.vehicles} sasakyang nakikita, {report.vehicles_in_water} sa kanila ay nasa baha.")
    if report.media_kind == "video":
        english_lines.append(f"Water was seen in {report.frames_with_water:.0%} of the {report.frames_checked} frames checked.")
        filipino_lines.append(f"Nakita ang baha sa {report.frames_with_water:.0%} ng {report.frames_checked} frame na sinuri.")

    place = ""
    if context.location is not None:
        place = f"Place given by you: {context.location.name} (not confirmed from the {kind}).\n"

    return place + " ".join(english_lines) + "\n" + " ".join(filipino_lines) + "\n\n" + SAFETY_TEXT
